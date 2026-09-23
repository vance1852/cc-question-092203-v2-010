"""命令行接口。

提供完整的风电场机位布局评估和优化流程。
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from .config import WindFarmConfig, create_sample_config
from .core.turbine import Turbine
from .core.wind_resource import WindResource
from .core.wake import WakeModel
from .constraints.boundary import SiteBoundary
from .constraints.spacing import compute_min_spacing_from_diameters
from .farm.aep import AEPCalculator, FarmResult
from .geo import (
    GeoJSONError,
    ImportedSite,
    export_boundary_geojson,
    export_layout_geojson,
    import_site,
)
from .optimization.baseline import generate_grid_layout
from .optimization.ga import GeneticAlgorithm, GAConfig
from .optimization.pso import ParticleSwarmOptimizer, PSOConfig
from .economy.costs import (
    EconomicAnalyzer,
    EconomicResult,
    get_default_turbine_cost,
    get_default_farm_cost,
)
from .visualization.plotting import (
    plot_farm_layout,
    plot_wind_rose,
    plot_convergence,
    plot_aep_vs_turbines,
    plot_turbine_loss_bar,
    plot_comparison,
    plot_wake_heatmap,
)


class WindFarmOptimizerCLI:
    """风电场优化命令行接口主类。"""

    def __init__(self, config: WindFarmConfig) -> None:
        self.config = config
        self._setup_output_dir()

        self.turbines = config.create_turbines()
        self.wind_resource = config.create_wind_resource()
        self.wake_model = config.create_wake_model()

        # GeoJSON 导入（可选）：一旦提供场界文件，边界改由投影后的
        # GeoJSON 提供；未提供时保持原有本地米制配置不变。
        self.geo_site: Optional[ImportedSite] = None
        if config.geo.layout_geojson and not config.geo.boundary_geojson:
            raise GeoJSONError(
                "提供了机位布局 GeoJSON 但未提供场界 GeoJSON；"
                "导入布局必须同时给出 --boundary-geojson 以便执行边界检查"
            )
        if config.geo.boundary_geojson:
            self._import_geo_site()
        else:
            self.boundary = config.create_boundary()

        if self.geo_site is not None and self.geo_site.turbines:
            # 机位数以导入布局为准，保证编号与位置一一对应（数量冲突已在
            # _import_geo_site 打印成功信息之前拒绝）
            config.n_turbines = len(self.geo_site.turbines)
            self.turbine_ids = self.geo_site.turbine_ids
            self.turbines = config.create_turbines()
        else:
            self.turbine_ids = [str(i) for i in range(config.n_turbines)]

        self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
        self.rated_powers = np.array([t.rated_power for t in self.turbines])
        self.thrust_coefficients = np.array([t.thrust_coefficient for t in self.turbines])

        self.aep_calc = AEPCalculator(
            turbines=self.turbines,
            wind_resource=self.wind_resource,
            wake_model=self.wake_model,
            wake_superposition=config.superposition_method,
        )

        self.baseline_positions: Optional[np.ndarray] = None
        self.baseline_result: Optional[FarmResult] = None
        self.optimized_positions: Optional[np.ndarray] = None
        self.optimized_result: Optional[FarmResult] = None
        self.optimize_result = None
        self.economic_result: Optional[EconomicResult] = None
        self.sweep_results: Optional[dict] = None

    def _import_geo_site(self) -> None:
        """从 GeoJSON 导入场界与可选机位布局，建立投影链并完成校验。"""
        geo_cfg = self.config.geo
        min_spacing = compute_min_spacing_from_diameters(
            np.array([t.rotor_diameter for t in self.turbines]),
            self.config.optimization.min_spacing_multiple,
        )
        self.geo_site = import_site(
            boundary_path=geo_cfg.boundary_geojson,
            layout_path=geo_cfg.layout_geojson,
            source_crs=geo_cfg.source_crs,
            target_crs=geo_cfg.target_crs,
            min_spacing_m=min_spacing if geo_cfg.layout_geojson else None,
            boundary_tolerance_m=geo_cfg.boundary_tolerance_m,
            roundtrip_tolerance_m=geo_cfg.roundtrip_tolerance_m,
            use_local_frame=geo_cfg.use_local_frame,
            origin_mode=geo_cfg.origin_mode,
            allow_cross_zone=geo_cfg.allow_cross_zone,
        )
        self.boundary = self.geo_site.boundary

        # 显式台数与导入机位数量冲突时，在打印成功信息前拒绝
        if self.geo_site.turbines and getattr(self.config, "_n_turbines_explicit", False):
            imported_n = len(self.geo_site.turbines)
            if self.config.n_turbines != imported_n:
                raise GeoJSONError(
                    f"命令行要求 {self.config.n_turbines} 台风机，但 GeoJSON 布局"
                    f"包含 {imported_n} 台机位；导入布局时机位数量以文件为准，"
                    "请删除 --n-turbines 或更换布局文件"
                )

        chain = self.geo_site.chain
        tgt_code = chain.target_crs.to_epsg() or chain.target_crs.name
        print("已导入 GeoJSON 场址：")
        print(f"  源坐标系:     {chain.source_crs.to_epsg() or chain.source_crs.name}"
              f"（声明来源 {self.geo_site.source_crs_declared_from}）")
        print(f"  米制计算坐标系: {tgt_code}（{chain.target_selection}）")
        print(f"  本地原点(投影m): ({chain.local_origin[0]:.1f}, {chain.local_origin[1]:.1f})")
        print(f"  往返最大误差:  {self.geo_site.roundtrip_summary['max_error_m']:.2e} m"
              f"（容差 {geo_cfg.roundtrip_tolerance_m} m）")
        if self.geo_site.turbines:
            print(f"  导入机位:      {len(self.geo_site.turbines)} 台，"
                  f"边界与间距检查通过，编号顺序将在优化与导出中保持")

    def _setup_output_dir(self) -> None:
        """创建输出目录。"""
        output_dir = self.config.visualization.save_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"输出目录: {os.path.abspath(output_dir)}")

    def _print_header(self, title: str) -> None:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def _print_result_summary(self, result: FarmResult, label: str = "") -> None:
        """打印计算结果摘要。"""
        print(f"\n--- {label} 结果 ---")
        print(f"  装机容量:    {result.total_installed_capacity:.2f} MW")
        print(f"  理论AEP:     {result.gross_aep/1e3:.2f} GWh/年")
        print(f"  净AEP:       {result.net_aep/1e3:.2f} GWh/年")
        print(f"  尾流损失:    {result.total_wake_loss/1e3:.2f} GWh/年 ({result.wake_loss_pct:.2f}%)")
        print(f"  容量系数:    {result.capacity_factor:.2f}%")
        print(f"  风机台数:    {len(result.turbine_results)}")

        max_loss_turb = max(result.turbine_results, key=lambda x: x.wake_loss_pct)
        print(f"  最大损失风机: #{max_loss_turb.turbine_idx} ({max_loss_turb.wake_loss_pct:.2f}%)")
        if max_loss_turb.dominant_wake_source is not None:
            print(f"    主要影响源: #{max_loss_turb.dominant_wake_source}")

    def run_baseline(self) -> None:
        """运行基线（规则网格布局或导入布局）评估。"""
        self._print_header("步骤 1/6: 生成并评估基线布局")

        rng = np.random.default_rng(self.config.optimization.seed)

        if self.geo_site is not None and self.geo_site.turbines:
            # 导入的设计方案本身作为基线，编号顺序与源文件一致
            self.baseline_positions = self.geo_site.local_positions.copy()
            print(f"以 GeoJSON 导入的 {self.config.n_turbines} 台机位布局作为基线")
        else:
            self.baseline_positions = generate_grid_layout(
                boundary=self.boundary,
                n_turbines=self.config.n_turbines,
                rotor_diameters=self.rotor_diameters,
                min_multiple=self.config.optimization.min_spacing_multiple,
                rng=rng,
            )
            print(f"已生成 {self.config.n_turbines} 台风机的网格布局")

        self.baseline_result = self.aep_calc.compute_farm_aep(self.baseline_positions)
        self._print_result_summary(self.baseline_result, "基线布局")

    def run_optimization(self) -> None:
        """运行机位优化。"""
        self._print_header("步骤 2/6: 执行机位布局优化")

        fit_fn = self.aep_calc.evaluate_layout

        algo = self.config.optimization.algorithm.lower()

        # GeoJSON 导入的设计方案作为优化种子，保证方案可在原编号基础上改进
        seed_positions = (
            self.geo_site.local_positions
            if self.geo_site is not None and self.geo_site.turbines
            else None
        )

        if algo == "ga":
            ga_config = GAConfig(
                population_size=self.config.optimization.population_size,
                max_generations=self.config.optimization.max_iterations,
                min_spacing_multiple=self.config.optimization.min_spacing_multiple,
                seed=self.config.optimization.seed,
            )
            optimizer = GeneticAlgorithm(
                n_turbines=self.config.n_turbines,
                rotor_diameters=self.rotor_diameters,
                boundary=self.boundary,
                fitness_fn=fit_fn,
                config=ga_config,
                initial_positions=seed_positions,
            )
        elif algo == "pso":
            pso_config = PSOConfig(
                swarm_size=self.config.optimization.population_size,
                max_iterations=self.config.optimization.max_iterations,
                min_spacing_multiple=self.config.optimization.min_spacing_multiple,
                seed=self.config.optimization.seed,
            )
            optimizer = ParticleSwarmOptimizer(
                n_turbines=self.config.n_turbines,
                rotor_diameters=self.rotor_diameters,
                boundary=self.boundary,
                fitness_fn=fit_fn,
                config=pso_config,
                initial_positions=seed_positions,
            )
        else:
            raise ValueError(f"未知的优化算法: {algo}")

        print(f"使用优化算法: {algo.upper()}")
        self.optimize_result = optimizer.optimize(verbose=True)

        self.optimized_positions = self.optimize_result.best_positions
        self.optimized_result = self.aep_calc.compute_farm_aep(self.optimized_positions)

        print("\n--- 优化后结果 ---")
        self._print_result_summary(self.optimized_result, "优化后布局")

        if self.baseline_result is not None:
            improvement = (
                (self.optimized_result.net_aep - self.baseline_result.net_aep)
                / self.baseline_result.net_aep
                * 100
            )
            loss_reduction = (
                (self.baseline_result.wake_loss_pct - self.optimized_result.wake_loss_pct)
                / self.baseline_result.wake_loss_pct
                * 100
            )
            print(f"\n--- 优化提升 ---")
            print(f"  发电量提升:   {improvement:+.2f}%")
            print(f"  尾流损失减少: {loss_reduction:+.2f}%")
            print(f"  额外发电量:   {(self.optimized_result.net_aep - self.baseline_result.net_aep)/1e3:+.2f} GWh/年")

    def run_economic_analysis(self) -> None:
        """运行经济性分析。"""
        if not self.config.economic.enable_analysis:
            return

        self._print_header("步骤 3/6: 经济性分析")

        if self.optimized_result is None:
            print("警告: 未进行优化，使用基线布局进行经济性分析")
            result = self.baseline_result
        else:
            result = self.optimized_result

        turbine_cost = get_default_turbine_cost(self.config.turbine_model)
        farm_cost = get_default_farm_cost()
        farm_cost.discount_rate = self.config.economic.discount_rate

        analyzer = EconomicAnalyzer(
            turbine_cost=turbine_cost,
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
        )

        rated_power_MW = self.turbines[0].rated_power / 1e3
        self.economic_result = analyzer.analyze(
            n_turbines=self.config.n_turbines,
            rated_power_per_turbine_MW=rated_power_MW,
            net_aep_GWh=result.net_aep / 1e3,
        )

        print(f"\n--- 经济性分析结果（基于优化后布局） ---")
        print(f"  上网电价:      {self.config.economic.electricity_price:.2f} 元/kWh")
        print(f"  折现率:        {self.config.economic.discount_rate*100:.1f}%")
        print(f"  初始投资:      {self.economic_result.total_capital_cost/1e4:.2f} 亿元")
        print(f"  年运维费用:    {self.economic_result.total_om_cost_annual:.1f} 万元/年")
        print(f"  年发电收益:    {self.economic_result.annual_revenue:.1f} 万元/年")
        print(f"  度电成本:      {self.economic_result.lcoe:.3f} 元/kWh")

        if self.economic_result.npv is not None:
            print(f"  净现值(NPV):   {self.economic_result.npv/1e4:+.2f} 亿元")
        if self.economic_result.irr is not None:
            print(f"  内部收益率:    {self.economic_result.irr:.2f}%")
        if self.economic_result.payback_period is not None:
            print(f"  投资回收期:    {self.economic_result.payback_period:.1f} 年")

        print(f"\n  成本构成:")
        for item, cost in self.economic_result.cost_breakdown.items():
            pct = cost / self.economic_result.total_capital_cost * 100
            print(f"    {item}: {cost/1e4:.2f} 亿元 ({pct:.1f}%)")

    def run_turbine_sweep(self, min_turbines: int = 5, max_turbines: int = 25, step: int = 2) -> None:
        """运行风机台数扫描分析。"""
        self._print_header("步骤 4/6: 风机台数扫描分析")

        print(f"扫描范围: {min_turbines} ~ {max_turbines} 台，步长 {step}")
        print("此分析将为不同台数快速优化布局并评估经济性")

        sweep_data = {
            "n_turbines": [],
            "aep": [],
            "lcoe": [],
        }

        rng = np.random.default_rng(self.config.optimization.seed)
        original_n = self.config.n_turbines

        turbine_cost = get_default_turbine_cost(self.config.turbine_model)
        farm_cost = get_default_farm_cost()
        analyzer = EconomicAnalyzer(
            turbine_cost=turbine_cost,
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
        )

        for n in range(min_turbines, max_turbines + 1, step):
            print(f"\n  分析 {n} 台风机...")
            self.config.n_turbines = n
            self.turbines = [self.turbines[0] for _ in range(n)]
            self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
            self.rated_powers = np.array([t.rated_power for t in self.turbines])

            self.aep_calc = AEPCalculator(
                turbines=self.turbines,
                wind_resource=self.wind_resource,
                wake_model=self.wake_model,
                wake_superposition=self.config.superposition_method,
            )

            try:
                positions = generate_grid_layout(
                    boundary=self.boundary,
                    n_turbines=n,
                    rotor_diameters=self.rotor_diameters,
                    min_multiple=self.config.optimization.min_spacing_multiple,
                    rng=rng,
                )

                result = self.aep_calc.compute_farm_aep(positions)

                rated_power_MW = self.turbines[0].rated_power / 1e3
                econ_result = analyzer.analyze(
                    n_turbines=n,
                    rated_power_per_turbine_MW=rated_power_MW,
                    net_aep_GWh=result.net_aep / 1e3,
                )

                sweep_data["n_turbines"].append(n)
                sweep_data["aep"].append(result.net_aep)
                sweep_data["lcoe"].append(econ_result.lcoe)

                print(f"    净AEP: {result.net_aep/1e3:.1f} GWh, LCOE: {econ_result.lcoe:.3f} 元/kWh")
            except Exception as e:
                print(f"    跳过: {e}")

        self.sweep_results = sweep_data
        self.config.n_turbines = original_n

    def run_visualization(self) -> None:
        """生成所有可视化图表。"""
        self._print_header("步骤 5/6: 生成可视化图表")

        save_dir = self.config.visualization.save_dir
        save = self.config.visualization.save_plots
        show = self.config.visualization.show_plots

        if save:
            print("图表将保存到:", os.path.abspath(save_dir))

        plot_wind_rose(
            wind_resource=self.wind_resource,
            title="项目场址风玫瑰图",
            save_path=os.path.join(save_dir, "wind_rose.png") if save else None,
            show=show,
        )

        if self.baseline_positions is not None and self.baseline_result is not None:
            baseline_losses = np.array([tr.wake_loss_pct for tr in self.baseline_result.turbine_results])
            plot_farm_layout(
                positions=self.baseline_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=baseline_losses,
                turbine_names=self.turbine_ids[:len(self.baseline_positions)],
                title="基线布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "baseline_layout.png") if save else None,
                show=show,
            )

            plot_turbine_loss_bar(
                farm_result=self.baseline_result,
                title="基线布局 - 各风机尾流损失",
                save_path=os.path.join(save_dir, "baseline_losses.png") if save else None,
                show=show,
            )

        if self.optimized_positions is not None and self.optimized_result is not None:
            opt_losses = np.array([tr.wake_loss_pct for tr in self.optimized_result.turbine_results])
            plot_farm_layout(
                positions=self.optimized_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=opt_losses,
                turbine_names=self.turbine_ids[:len(self.optimized_positions)],
                title="优化后布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "optimized_layout.png") if save else None,
                show=show,
            )

            plot_turbine_loss_bar(
                farm_result=self.optimized_result,
                title="优化后布局 - 各风机尾流损失",
                save_path=os.path.join(save_dir, "optimized_losses.png") if save else None,
                show=show,
            )

        if self.optimize_result is not None and self.baseline_result is not None:
            plot_convergence(
                optimize_result=self.optimize_result,
                baseline_aep=self.baseline_result.net_aep,
                title="优化收敛曲线",
                save_path=os.path.join(save_dir, "convergence.png") if save else None,
                show=show,
            )

        if self.baseline_result is not None and self.optimized_result is not None:
            plot_comparison(
                baseline_result=self.baseline_result,
                optimized_result=self.optimized_result,
                title="优化前后关键指标对比",
                save_path=os.path.join(save_dir, "comparison.png") if save else None,
                show=show,
            )

        if self.sweep_results is not None:
            plot_aep_vs_turbines(
                n_turbines_list=self.sweep_results["n_turbines"],
                aep_list=self.sweep_results["aep"],
                lcoe_list=self.sweep_results["lcoe"],
                title="风机台数优化分析",
                save_path=os.path.join(save_dir, "aep_vs_turbines.png") if save else None,
                show=show,
            )

        if self.config.visualization.plot_wake_heatmap and self.optimized_positions is not None:
            dominant_dir = self.wind_resource.directions[np.argmax(self.wind_resource.frequencies)]
            plot_wake_heatmap(
                positions=self.optimized_positions,
                boundary=self.boundary,
                wake_model=self.wake_model,
                wind_direction=dominant_dir,
                rotor_diameters=self.rotor_diameters,
                thrust_coefficients=self.thrust_coefficients,
                title=f"主风向({dominant_dir:.0f}°)尾流速度亏损分布",
                save_path=os.path.join(save_dir, "wake_heatmap.png") if save else None,
                show=show,
            )

    def save_results(self) -> None:
        """保存所有结果到JSON文件。"""
        self._print_header("步骤 6/6: 保存结果数据")

        output_dir = self.config.visualization.save_dir

        results = {
            "config": {
                "n_turbines": self.config.n_turbines,
                "turbine_model": self.config.turbine_model,
                "wake_model": self.config.wake_model,
                "min_spacing_multiple": self.config.optimization.min_spacing_multiple,
            },
            "site": {
                "area_km2": float(self.boundary.area / 1e6),
                "mean_wind_speed": float(self.wind_resource.overall_mean_speed),
            },
        }

        if self.geo_site is not None:
            results["geo"] = self.geo_site.provenance_dict()

        if self.baseline_result is not None:
            results["baseline"] = {
                "positions": self.baseline_positions.tolist() if self.baseline_positions is not None else None,
                "gross_aep_gwh": float(self.baseline_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.baseline_result.net_aep / 1e3),
                "wake_loss_pct": float(self.baseline_result.wake_loss_pct),
                "capacity_factor": float(self.baseline_result.capacity_factor),
                "turbine_losses": [
                    {
                        "idx": tr.turbine_idx,
                        "wake_loss_pct": float(tr.wake_loss_pct),
                        "dominant_source": tr.dominant_wake_source,
                    }
                    for tr in self.baseline_result.turbine_results
                ],
            }

        if self.optimized_result is not None:
            results["optimized"] = {
                "positions": self.optimized_positions.tolist() if self.optimized_positions is not None else None,
                "gross_aep_gwh": float(self.optimized_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.optimized_result.net_aep / 1e3),
                "wake_loss_pct": float(self.optimized_result.wake_loss_pct),
                "capacity_factor": float(self.optimized_result.capacity_factor),
                "turbine_losses": [
                    {
                        "idx": tr.turbine_idx,
                        "wake_loss_pct": float(tr.wake_loss_pct),
                        "dominant_source": tr.dominant_wake_source,
                    }
                    for tr in self.optimized_result.turbine_results
                ],
            }

        if self.economic_result is not None:
            results["economic"] = {
                "total_capital_cost_yiyuan": float(self.economic_result.total_capital_cost / 1e4),
                "annual_revenue_wanyuan": float(self.economic_result.annual_revenue),
                "lcoe_yuan_per_kwh": float(self.economic_result.lcoe),
                "npv_yiyuan": float(self.economic_result.npv / 1e4) if self.economic_result.npv is not None else None,
                "irr_pct": float(self.economic_result.irr) if self.economic_result.irr is not None else None,
                "payback_years": float(self.economic_result.payback_period) if self.economic_result.payback_period is not None else None,
            }

        if self.baseline_result is not None and self.optimized_result is not None:
            results["improvement"] = {
                "aep_improvement_pct": float(
                    (self.optimized_result.net_aep - self.baseline_result.net_aep)
                    / self.baseline_result.net_aep * 100
                ),
                "additional_aep_gwh": float(
                    (self.optimized_result.net_aep - self.baseline_result.net_aep) / 1e3
                ),
                "loss_reduction_pct": float(
                    (self.baseline_result.wake_loss_pct - self.optimized_result.wake_loss_pct)
                    / self.baseline_result.wake_loss_pct * 100
                ),
            }

        if self.sweep_results is not None:
            results["turbine_sweep"] = {
                "n_turbines": self.sweep_results["n_turbines"],
                "aep_mwh": self.sweep_results["aep"],
                "lcoe_yuan_per_kwh": self.sweep_results["lcoe"],
            }

        results_path = os.path.join(output_dir, "results.json")

        if self.geo_site is not None and self.config.geo.export_geojson:
            self._export_geo_files(output_dir, results)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        config_path = os.path.join(output_dir, "config.json")
        self.config.to_json(config_path)

        print(f"结果已保存到: {os.path.abspath(results_path)}")
        print(f"配置已保存到: {os.path.abspath(config_path)}")

    def _computed_attributes(self, farm_result: FarmResult) -> list[dict]:
        """把每台风机的计算结果整理为导出属性，尾流来源映射为机位编号。"""
        attrs: list[dict] = []
        for tr in farm_result.turbine_results:
            source_id = (
                self.turbine_ids[tr.dominant_wake_source]
                if tr.dominant_wake_source is not None
                and 0 <= tr.dominant_wake_source < len(self.turbine_ids)
                else None
            )
            attrs.append(
                {
                    "net_aep_mwh": float(tr.net_aep),
                    "gross_aep_mwh": float(tr.gross_aep),
                    "wake_loss_pct": float(tr.wake_loss_pct),
                    "capacity_factor_pct": float(tr.capacity_factor),
                    "dominant_wake_source_id": source_id,
                }
            )
        return attrs

    def _export_geo_files(self, output_dir: str, results: dict) -> None:
        """把场界与布局反变换导出为源坐标系的 GeoJSON。

        导出文件的绝对路径会写入 ``results['geo']['exported_files']``，
        由调用方随后统一写入 results.json。
        """
        site = self.geo_site
        metadata = {
            "source_files": {
                "boundary": site.boundary_path,
                "layout": site.layout_path,
            },
            "projection_summary": site.chain.as_dict(),
            "roundtrip_verification": site.roundtrip_summary,
        }

        boundary_out = os.path.join(output_dir, "boundary_export.geojson")
        export_boundary_geojson(
            site.chain,
            self.boundary.vertices,
            boundary_out,
            properties=site.boundary_properties,
            metadata=metadata,
        )
        print(f"场界 GeoJSON 已导出: {os.path.abspath(boundary_out)}")

        exported = {"boundary": os.path.abspath(boundary_out)}

        # 优先导出优化后布局；未优化时导出基线布局（导入方案原样往返）
        if self.optimized_positions is not None and self.optimized_result is not None:
            layout_out = os.path.join(output_dir, "layout_optimized.geojson")
            computed = self._computed_attributes(self.optimized_result)
            export_layout_geojson(
                site.chain,
                self.optimized_positions,
                self.turbine_ids,
                layout_out,
                source_attributes=[t.attributes for t in site.turbines]
                if site.turbines
                else None,
                computed_attributes=computed,
                metadata=metadata,
            )
            print(f"优化布局 GeoJSON 已导出: {os.path.abspath(layout_out)}")
            exported["optimized_layout"] = os.path.abspath(layout_out)
        elif self.baseline_positions is not None and self.baseline_result is not None:
            layout_out = os.path.join(output_dir, "layout_baseline.geojson")
            computed = self._computed_attributes(self.baseline_result)
            export_layout_geojson(
                site.chain,
                self.baseline_positions,
                self.turbine_ids,
                layout_out,
                source_attributes=[t.attributes for t in site.turbines]
                if site.turbines
                else None,
                computed_attributes=computed,
                metadata=metadata,
            )
            print(f"基线布局 GeoJSON 已导出: {os.path.abspath(layout_out)}")
            exported["baseline_layout"] = os.path.abspath(layout_out)

        results["geo"]["exported_files"] = exported

    def run_full_analysis(
        self,
        run_baseline: bool = True,
        run_opt: bool = True,
        run_econ: bool = True,
        run_sweep: bool = False,
        run_viz: bool = True,
        save: bool = True,
    ) -> None:
        """运行完整分析流程。"""
        start_time = time.time()

        self._print_header("风电场机位布局优化分析")
        print(f"  风机: {self.config.turbine_model} x {self.config.n_turbines} 台")
        print(f"  尾流模型: {self.config.wake_model}")
        print(f"  平均风速: {self.wind_resource.overall_mean_speed:.2f} m/s")
        print(f"  场地面积: {self.boundary.area / 1e6:.2f} km²")

        if run_baseline:
            self.run_baseline()

        if run_opt:
            self.run_optimization()

        if run_econ:
            self.run_economic_analysis()

        if run_sweep:
            if self.geo_site is not None and self.geo_site.turbines:
                print("\n提示: 已从 GeoJSON 导入固定机位布局，跳过风机台数扫描")
            else:
                self.run_turbine_sweep(
                    min_turbines=getattr(self, '_min_turbines', 5),
                    max_turbines=getattr(self, '_max_turbines', 25),
                )

        if run_viz:
            self.run_visualization()

        if save:
            self.save_results()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  全部分析完成! 耗时: {elapsed:.1f} 秒")
        print(f"{'='*60}\n")


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="风电场机位布局优化工具 - 尾流计算、布局优化、经济性评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置运行完整分析
  python -m wind_farm_opt

  # 从配置文件运行
  python -m wind_farm_opt --config my_config.json

  # 自定义参数运行
  python -m wind_farm_opt --n-turbines 20 --turbine V164-9.5MW --wake-model gaussian

  # 仅评估不优化
  python -m wind_farm_opt --no-optimization

  # 启用风机台数扫描
  python -m wind_farm_opt --sweep --min-turbines 10 --max-turbines 30
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径(JSON格式)",
    )

    parser.add_argument(
        "--n-turbines",
        type=int,
        default=None,
        help="风机台数",
    )

    parser.add_argument(
        "--turbine",
        type=str,
        default=None,
        choices=["V126-3.45MW", "V164-9.5MW"],
        help="风机型号",
    )

    parser.add_argument(
        "--wake-model",
        type=str,
        default=None,
        choices=["jensen", "gaussian"],
        help="尾流模型: jensen 或 gaussian",
    )

    parser.add_argument(
        "--wake-decay",
        type=float,
        default=None,
        help="尾流衰减系数 (Jensen模型)",
    )

    parser.add_argument(
        "--boundary",
        type=str,
        default=None,
        choices=["rectangular", "hexagonal", "irregular"],
        help="场地边界类型",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="矩形场地宽度 (m)",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="矩形场地高度 (m)",
    )

    parser.add_argument(
        "--min-spacing",
        type=float,
        default=None,
        help="最小间距倍数（转子直径倍数）",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["ga", "pso"],
        help="优化算法: ga(遗传算法) 或 pso(粒子群)",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="种群/粒子群大小",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="最大迭代代数",
    )

    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="仅评估基线布局，不执行优化",
    )

    parser.add_argument(
        "--no-economic",
        action="store_true",
        help="跳过经济性分析",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="启用风机台数扫描分析",
    )

    parser.add_argument(
        "--min-turbines",
        type=int,
        default=5,
        help="台数扫描最小值",
    )

    parser.add_argument(
        "--max-turbines",
        type=int,
        default=25,
        help="台数扫描最大值",
    )

    parser.add_argument(
        "--electricity-price",
        type=float,
        default=None,
        help="上网电价 (元/kWh)",
    )

    parser.add_argument(
        "--discount-rate",
        type=float,
        default=None,
        help="折现率 (0-1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成图表",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="显示图表窗口",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子",
    )

    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        help="生成示例配置文件并退出",
    )

    # ---- GeoJSON 地理数据导入导出 ----
    parser.add_argument(
        "--boundary-geojson",
        type=str,
        default=None,
        help="场界 GeoJSON 文件路径（必须带坐标参考系或同时给出 --source-crs）",
    )

    parser.add_argument(
        "--layout-geojson",
        type=str,
        default=None,
        help="机位布局 GeoJSON 文件路径；导入前会执行场界包含与最小间距检查，"
             "并作为优化的初始种子",
    )

    parser.add_argument(
        "--source-crs",
        type=str,
        default=None,
        help="源坐标参考系，如 EPSG:4326(WGS84)、EPSG:4490(CGCS2000)；"
             "与 GeoJSON 内嵌 crs 至少给出一个",
    )

    parser.add_argument(
        "--target-crs",
        type=str,
        default=None,
        help="米制计算坐标参考系，如 EPSG:32650；缺省时按场址自动选择 UTM 带",
    )

    parser.add_argument(
        "--roundtrip-tolerance",
        type=float,
        default=None,
        help="坐标正反变换往返误差上限（米，默认 0.05）",
    )

    parser.add_argument(
        "--allow-cross-zone",
        action="store_true",
        help="允许明显跨越 UTM 投影带的场址（默认拒绝）",
    )

    parser.add_argument(
        "--no-geojson-export",
        action="store_true",
        help="优化后不导出源坐标系的 GeoJSON（默认导出）",
    )

    return parser


def main() -> int:
    """主函数入口。"""
    parser = build_argparser()
    args = parser.parse_args()

    if args.generate_config:
        config = create_sample_config()
        config.to_json(args.generate_config)
        print(f"示例配置已生成: {os.path.abspath(args.generate_config)}")
        return 0

    if args.config:
        config = WindFarmConfig.from_json(args.config)
    else:
        config = create_sample_config()

    if args.n_turbines is not None:
        config.n_turbines = args.n_turbines
        config._n_turbines_explicit = True
    if args.turbine is not None:
        config.turbine_model = args.turbine
    if args.wake_model is not None:
        config.wake_model = args.wake_model
    if args.wake_decay is not None:
        config.wake_decay = args.wake_decay
    if args.boundary is not None:
        config.boundary_type = args.boundary
    if args.width is not None:
        config.boundary_params["width"] = args.width
    if args.height is not None:
        config.boundary_params["height"] = args.height
    if args.min_spacing is not None:
        config.optimization.min_spacing_multiple = args.min_spacing
    if args.algorithm is not None:
        config.optimization.algorithm = args.algorithm
    if args.population is not None:
        config.optimization.population_size = args.population
    if args.iterations is not None:
        config.optimization.max_iterations = args.iterations
    if args.seed is not None:
        config.optimization.seed = args.seed
    if args.electricity_price is not None:
        config.economic.electricity_price = args.electricity_price
    if args.discount_rate is not None:
        config.economic.discount_rate = args.discount_rate
    if args.output_dir is not None:
        config.visualization.save_dir = args.output_dir
    if args.no_plots:
        config.visualization.save_plots = False
    if args.show_plots:
        config.visualization.show_plots = True
    if args.no_economic:
        config.economic.enable_analysis = False

    if args.boundary_geojson is not None:
        config.geo.boundary_geojson = args.boundary_geojson
    if args.layout_geojson is not None:
        config.geo.layout_geojson = args.layout_geojson
    if args.source_crs is not None:
        config.geo.source_crs = args.source_crs
    if args.target_crs is not None:
        config.geo.target_crs = args.target_crs
    if args.roundtrip_tolerance is not None:
        config.geo.roundtrip_tolerance_m = args.roundtrip_tolerance
    if args.allow_cross_zone:
        config.geo.allow_cross_zone = True
    if args.no_geojson_export:
        config.geo.export_geojson = False

    cli = WindFarmOptimizerCLI(config)
    cli._min_turbines = args.min_turbines
    cli._max_turbines = args.max_turbines

    try:
        cli.run_full_analysis(
            run_baseline=True,
            run_opt=not args.no_optimization,
            run_econ=not args.no_economic,
            run_sweep=args.sweep,
            run_viz=not args.no_plots,
            save=True,
        )
        return 0
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
