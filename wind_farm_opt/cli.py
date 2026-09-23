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
from .core.turbine import Turbine, create_default_turbine
from .core.wind_resource import WindResource
from .core.wake import WakeModel
from .constraints.boundary import SiteBoundary
from .constraints.spacing import compute_min_spacing_from_diameters
from .farm.aep import AEPCalculator, FarmResult
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
from .geospatial import (
    ImportedSite,
    export_geojson,
    import_geojson,
)


class WindFarmOptimizerCLI:
    """风电场优化命令行接口主类。"""

    def __init__(
        self,
        config: WindFarmConfig,
        imported_site: Optional[ImportedSite] = None,
        export_geojson_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.imported_site = imported_site
        self.export_geojson_path = export_geojson_path
        self._setup_output_dir()

        if imported_site is not None:
            # GeoJSON 导入模式：场界、机位数量与初始布局来自源文件，
            # 全部已在本地米制计算坐标系中；配置中的本地坐标参数不参与。
            self.boundary = imported_site.boundary
            n = len(imported_site.turbine_ids)
            self.config.n_turbines = n
            self.turbine_ids = list(imported_site.turbine_ids)
            self.turbines = config.create_turbines()
            self.imported_positions = imported_site.positions
            # 图面显示原点：场界西南角，仅用于把投影带大坐标显示为局部小数字。
            self.display_origin = np.array(
                [self.boundary.x_min, self.boundary.y_min], dtype=np.float64
            )
        else:
            # 原有本地坐标模式，行为完全保持不变。
            self.turbines = config.create_turbines()
            self.boundary = config.create_boundary()
            self.turbine_ids = [str(i) for i in range(config.n_turbines)]
            self.imported_positions = None
            self.display_origin = None

        self.wind_resource = config.create_wind_resource()
        self.wake_model = config.create_wake_model()

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
        """运行基线评估。

        GeoJSON 导入模式下，基线即勘测/设计交回的原始布局（已通过边界与
        间距检查），用于与优化结果对比；本地坐标模式仍生成规则网格。
        """
        if self.imported_positions is not None:
            self._print_header("步骤 1/6: 评估导入的原始机位布局")
            self.baseline_positions = self.imported_positions.copy()
            print(f"已载入 {self.config.n_turbines} 台风机的 GeoJSON 原始布局")
        else:
            self._print_header("步骤 1/6: 生成并评估基线网格布局")

            rng = np.random.default_rng(self.config.optimization.seed)
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
                initial_positions=self.imported_positions,
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
                initial_positions=self.imported_positions,
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
            baseline_title = (
                "导入原始布局 - 尾流损失分布"
                if self.imported_site is not None
                else "基线网格布局 - 尾流损失分布"
            )
            plot_farm_layout(
                positions=self.baseline_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=baseline_losses,
                turbine_names=list(self.turbine_ids),
                title=baseline_title,
                save_path=os.path.join(save_dir, "baseline_layout.png") if save else None,
                show=show,
                display_origin=self.display_origin,
            )

            plot_turbine_loss_bar(
                farm_result=self.baseline_result,
                title="基线布局 - 各风机尾流损失",
                save_path=os.path.join(save_dir, "baseline_losses.png") if save else None,
                show=show,
                turbine_labels=list(self.turbine_ids),
            )

        if self.optimized_positions is not None and self.optimized_result is not None:
            opt_losses = np.array([tr.wake_loss_pct for tr in self.optimized_result.turbine_results])
            plot_farm_layout(
                positions=self.optimized_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=opt_losses,
                turbine_names=list(self.turbine_ids),
                title="优化后布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "optimized_layout.png") if save else None,
                show=show,
                display_origin=self.display_origin,
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
                display_origin=self.display_origin,
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

        if self.imported_site is not None:
            summary = self.imported_site.summary
            results["coordinate_transform"] = {
                "source_file": getattr(self, "import_source_path", None),
                "source_crs": summary.source_crs,
                "source_crs_kind": summary.source_crs_kind,
                "source_crs_declared_in": summary.source_crs_declared_in,
                "source_crs_assumed": summary.source_crs_assumed,
                "target_metric_crs": summary.metric_crs,
                "metric_crs_family": summary.metric_crs_family,
                "zone_label": summary.zone_label,
                "source_bounds_lonlat": summary.source_bounds_lonlat,
                "forward_chain": summary.forward_chain,
                "inverse_chain": summary.inverse_chain,
                "import_roundtrip_max_error_m": summary.roundtrip_max_error_m,
                "roundtrip_tolerance_m": summary.roundtrip_tolerance_m,
                "n_check_points": summary.n_check_points,
                "boundary_tolerance_m": self.imported_site.boundary_tolerance_m,
                "near_boundary_turbine_ids": self.imported_site.near_boundary_ids,
                "source_dimension": self.imported_site.source_dimension,
                "datum_note": summary.datum_note,
                "exported_geojson": None,
            }

        if self.baseline_result is not None:
            results["baseline"] = {
                "positions": self.baseline_positions.tolist() if self.baseline_positions is not None else None,
                "turbine_ids": list(self.turbine_ids),
                "gross_aep_gwh": float(self.baseline_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.baseline_result.net_aep / 1e3),
                "wake_loss_pct": float(self.baseline_result.wake_loss_pct),
                "capacity_factor": float(self.baseline_result.capacity_factor),
                "turbine_losses": [
                    {
                        "idx": tr.turbine_idx,
                        "turbine_id": self.turbine_ids[tr.turbine_idx]
                        if tr.turbine_idx < len(self.turbine_ids) else str(tr.turbine_idx),
                        "wake_loss_pct": float(tr.wake_loss_pct),
                        "dominant_source": tr.dominant_wake_source,
                    }
                    for tr in self.baseline_result.turbine_results
                ],
            }

        if self.optimized_result is not None:
            results["optimized"] = {
                "positions": self.optimized_positions.tolist() if self.optimized_positions is not None else None,
                "turbine_ids": list(self.turbine_ids),
                "gross_aep_gwh": float(self.optimized_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.optimized_result.net_aep / 1e3),
                "wake_loss_pct": float(self.optimized_result.wake_loss_pct),
                "capacity_factor": float(self.optimized_result.capacity_factor),
                "turbine_losses": [
                    {
                        "idx": tr.turbine_idx,
                        "turbine_id": self.turbine_ids[tr.turbine_idx]
                        if tr.turbine_idx < len(self.turbine_ids) else str(tr.turbine_idx),
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
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        config_path = os.path.join(output_dir, "config.json")
        self.config.to_json(config_path)

        print(f"结果已保存到: {os.path.abspath(results_path)}")
        print(f"配置已保存到: {os.path.abspath(config_path)}")

        if self.imported_site is not None:
            self._export_optimized_geojson(output_dir, results)
            # 回写导出文件路径到 results.json
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    def _export_optimized_geojson(self, output_dir: str, results: dict) -> None:
        """把优化后布局逆变换回源坐标系并导出 GeoJSON。"""
        if self.optimized_positions is not None:
            positions = self.optimized_positions
            stage = "optimized"
            farm_result = self.optimized_result
            filename = "optimized_layout.geojson"
        elif self.baseline_positions is not None:
            # --no-optimization 时导出评估后的原始布局
            positions = self.baseline_positions
            stage = "imported"
            farm_result = self.baseline_result
            filename = "evaluated_layout.geojson"
        else:
            return

        out_path = self.export_geojson_path or os.path.join(output_dir, filename)

        attrs: dict[str, dict] = {}
        if farm_result is not None:
            for tr in farm_result.turbine_results:
                tid = (
                    self.turbine_ids[tr.turbine_idx]
                    if tr.turbine_idx < len(self.turbine_ids)
                    else str(tr.turbine_idx)
                )
                attrs[tid] = {
                    "net_aep_mwh": float(tr.net_aep),
                    "wake_loss_pct": float(tr.wake_loss_pct),
                    "capacity_factor_pct": float(tr.capacity_factor),
                }

        export_geojson(
            site=self.imported_site,
            metric_positions=positions,
            output_path=out_path,
            turbine_ids=list(self.turbine_ids),
            turbine_attributes=attrs,
            layout_stage=stage,
            extra_boundary_properties={
                "turbine_count": len(self.turbine_ids),
                "turbine_model": self.config.turbine_model,
            },
        )
        results["coordinate_transform"]["exported_geojson"] = os.path.abspath(out_path)
        print(f"GeoJSON 布局已导出（源坐标系）: {os.path.abspath(out_path)}")

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

    # ---- GeoJSON 导入导出 ----
    parser.add_argument(
        "--import-geojson",
        type=str,
        default=None,
        help="导入场界/机位 GeoJSON 文件路径；提供后忽略配置中的本地坐标边界",
    )
    parser.add_argument(
        "--source-crs",
        type=str,
        default=None,
        help="源坐标参考系，覆盖 GeoJSON crs 成员，如 EPSG:4326、EPSG:32650",
    )
    parser.add_argument(
        "--metric-crs",
        type=str,
        default=None,
        help="显式指定米制计算坐标系，如 EPSG:32650；默认按场址自动选带",
    )
    parser.add_argument(
        "--metric-crs-preference",
        type=str,
        default="auto",
        choices=["auto", "cgcs2000", "utm"],
        help="自动选带偏好：auto（中国境内 CGCS2000 三度带，境外 UTM）/ cgcs2000 / utm",
    )
    parser.add_argument(
        "--export-geojson",
        type=str,
        default=None,
        help="优化后 GeoJSON 导出路径（默认写入输出目录 optimized_layout.geojson）",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=0.5,
        help="导入机位边界判定容差（米），默认 0.5",
    )
    parser.add_argument(
        "--roundtrip-tolerance",
        type=float,
        default=1e-3,
        help="坐标正反变换往返残差容差（米），默认 0.001",
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

    imported_site = None
    if args.import_geojson:
        # GeoJSON 导入模式：机位数量由文件决定，命令行 --n-turbines 不适用。
        # 间距检查使用当前配置的最小间距倍数对应的米数。
        probe_turbine = create_default_turbine(config.turbine_model)
        min_spacing_m = compute_min_spacing_from_diameters(
            np.array([probe_turbine.rotor_diameter]),
            config.optimization.min_spacing_multiple,
        )
        try:
            imported_site = import_geojson(
                path=args.import_geojson,
                source_crs=args.source_crs,
                metric_crs=args.metric_crs,
                metric_preference=args.metric_crs_preference,
                min_spacing_m=min_spacing_m,
                boundary_tolerance_m=args.boundary_tolerance,
                roundtrip_tolerance_m=args.roundtrip_tolerance,
                require_turbines=True,
            )
        except Exception as e:
            print(f"\nGeoJSON 导入失败: {e}", file=sys.stderr)
            return 2

        s = imported_site.summary
        print("=" * 60)
        print("  GeoJSON 场址导入")
        print("=" * 60)
        print(f"  源坐标系:     {s.source_crs}（{s.source_crs_kind}，"
              f"{'声明' if not s.source_crs_assumed else 'RFC7946 默认'}）")
        print(f"  米制坐标系:   {s.metric_crs}（{s.zone_label}）")
        print(f"  转换链路:     {s.forward_chain}")
        print(f"  往返最大残差: {s.roundtrip_max_error_m:.3e} m"
              f"（容差 {s.roundtrip_tolerance_m:g} m）")
        print(f"  机位数量:     {len(imported_site.turbine_ids)}")
        if imported_site.near_boundary_ids:
            print(f"  贴边机位:     {', '.join(imported_site.near_boundary_ids)}")
        if s.datum_note:
            print(f"  基准提示:     {s.datum_note}")

        if args.sweep:
            print("提示: GeoJSON 导入模式机位数量固定，已跳过台数扫描。")

    cli = WindFarmOptimizerCLI(
        config,
        imported_site=imported_site,
        export_geojson_path=args.export_geojson,
    )
    if imported_site is not None:
        cli.import_source_path = os.path.abspath(args.import_geojson)
    cli._min_turbines = args.min_turbines
    cli._max_turbines = args.max_turbines

    try:
        cli.run_full_analysis(
            run_baseline=True,
            run_opt=not args.no_optimization,
            run_econ=not args.no_economic,
            run_sweep=args.sweep and imported_site is None,
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
