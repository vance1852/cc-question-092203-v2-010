"""GeoJSON 导入导出与 CLI / 配置层的集成测试。"""

import json
import subprocess
import sys

import numpy as np
import pytest

from wind_farm_opt.config import WindFarmConfig, create_sample_config
from wind_farm_opt.geo import import_site

from conftest import boundary_geojson, grid_points, layout_geojson


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8"
    )
    return str(path)


def test_config_geo_section_roundtrip(tmp_path):
    config = create_sample_config()
    # 原有本地坐标配置不受影响
    assert config.boundary_type == "rectangular"
    config.geo.boundary_geojson = "site.geojson"
    config.geo.source_crs = "EPSG:4326"
    config.geo.target_crs = "EPSG:32650"

    path = str(tmp_path / "cfg.json")
    config.to_json(path)
    loaded = WindFarmConfig.from_json(path)
    assert loaded.geo.boundary_geojson == "site.geojson"
    assert loaded.geo.source_crs == "EPSG:4326"
    assert loaded.geo.target_crs == "EPSG:32650"
    # 原配置字段保持
    assert loaded.boundary_type == "rectangular"
    assert loaded.boundary_params["width"] == 3500


def test_config_without_geo_section_still_loads():
    # 旧配置文件（无 geo 段）必须照常加载
    import os
    import tempfile

    minimal = {
        "n_turbines": 6,
        "boundary_type": "rectangular",
        "boundary_params": {"width": 2000, "height": 2000},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(minimal, f)
        tmppath = f.name
    try:
        loaded = WindFarmConfig.from_json(tmppath)
        assert loaded.n_turbines == 6
        assert loaded.geo.boundary_geojson is None
    finally:
        os.unlink(tmppath)


def test_local_config_unchanged_when_no_geojson():
    config = create_sample_config()
    boundary = config.create_boundary()
    # 默认矩形边界仍是以原点为中心的本地米制坐标
    assert boundary.x_min == -1750.0
    assert boundary.x_max == 1750.0
    assert config.geo.boundary_geojson is None


def test_layout_without_boundary_rejected(tmp_path):
    from wind_farm_opt.cli import WindFarmOptimizerCLI
    from wind_farm_opt.geo import GeoJSONError

    config = create_sample_config()
    config.geo.layout_geojson = "layout.geojson"
    config.visualization.save_dir = str(tmp_path / "out")
    config.visualization.save_plots = False
    config.economic.enable_analysis = False
    with pytest.raises(GeoJSONError, match="场界"):
        WindFarmOptimizerCLI(config)


def test_cli_end_to_end_with_geojson(tmp_path):
    bpath = _write(tmp_path, "boundary.geojson", boundary_geojson())
    lpath = _write(tmp_path, "layout.geojson",
                   layout_geojson(grid_points(),
                                  extra_props={"model": "V126-3.45MW"}))
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable, "-m", "wind_farm_opt",
        "--boundary-geojson", bpath,
        "--layout-geojson", lpath,
        "--iterations", "4",
        "--population", "8",
        "--no-economic",
        "--no-plots",
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, cwd="/workspace", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    results = json.loads((out_dir / "results.json").read_text())
    geo = results["geo"]
    assert geo["projection"]["source_crs"]["code"] == "EPSG:4326"
    assert geo["projection"]["target_crs"]["code"] == "EPSG:32650"
    assert geo["roundtrip_verification"]["passed"] is True
    assert geo["layout_import_validation"]["passed"] is True
    assert geo["turbine_ids"] == [f"WT-{i:02d}" for i in range(1, 13)]

    exported_layout = out_dir / "layout_optimized.geojson"
    exported_boundary = out_dir / "boundary_export.geojson"
    assert exported_layout.exists()
    assert exported_boundary.exists()

    doc = json.loads(exported_layout.read_text())
    assert [f["id"] for f in doc["features"]] == geo["turbine_ids"]
    assert doc["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::4326"
    # 原始属性保留
    assert doc["features"][0]["properties"]["model"] == "V126-3.45MW"
    # 计算属性随导出
    assert "wfo_wake_loss_pct" in doc["features"][0]["properties"]

    # 导出文件必须能再次通过导入校验（完整往返）
    re_site = import_site(str(exported_boundary),
                          layout_path=str(exported_layout),
                          min_spacing_m=630.0)
    assert re_site.validation.passed is True
    assert re_site.turbine_ids == geo["turbine_ids"]


def test_cli_rejects_missing_crs(tmp_path):
    bpath = _write(tmp_path, "b.geojson", boundary_geojson(crs=None))
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, "-m", "wind_farm_opt",
        "--boundary-geojson", bpath,
        "--no-plots", "--no-economic",
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, cwd="/workspace", capture_output=True, text=True)
    assert result.returncode == 1
    assert "坐标参考系" in result.stderr or "坐标参考系" in result.stdout


def test_cli_explicit_crs_arg_succeeds(tmp_path):
    bpath = _write(tmp_path, "b.geojson", boundary_geojson(crs=None))
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, "-m", "wind_farm_opt",
        "--boundary-geojson", bpath,
        "--source-crs", "EPSG:4326",
        "--n-turbines", "6",
        "--iterations", "2", "--population", "6",
        "--no-plots", "--no-economic",
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, cwd="/workspace", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    results = json.loads((out_dir / "results.json").read_text())
    assert results["geo"]["projection"]["source_crs"]["code"] == "EPSG:4326"


def test_optimizer_seed_uses_imported_layout():
    from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
    from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig
    from wind_farm_opt.constraints.boundary import create_rectangular_boundary

    boundary = create_rectangular_boundary(3000, 3000)
    diameters = np.full(4, 126.0)
    seed = np.array([[-800, -800], [800, -800], [800, 800], [-800, 800]], dtype=float)

    def fitness(pos):
        return 1000.0 - float(np.sum(pos ** 2)) * 1e-6

    ga = GeneticAlgorithm(
        n_turbines=4, rotor_diameters=diameters, boundary=boundary,
        fitness_fn=fitness,
        config=GAConfig(population_size=6, max_generations=2, seed=1),
        initial_positions=seed,
    )
    population = ga._initialize_population(6)
    # 第一个个体必须是种子布局
    np.testing.assert_allclose(population[0].reshape(4, 2), seed)

    pso = ParticleSwarmOptimizer(
        n_turbines=4, rotor_diameters=diameters, boundary=boundary,
        fitness_fn=fitness,
        config=PSOConfig(swarm_size=6, max_iterations=2, seed=1),
        initial_positions=seed,
    )
    positions, velocities = pso._initialize_swarm(6)
    np.testing.assert_allclose(positions[0].reshape(4, 2), seed)
    # 种子粒子初始速度为 0
    np.testing.assert_allclose(velocities[0], 0.0)

    # 错误形状的种子被拒绝
    with pytest.raises(ValueError):
        GeneticAlgorithm(
            n_turbines=4, rotor_diameters=diameters, boundary=boundary,
            fitness_fn=fitness, initial_positions=seed[:3],
        )
