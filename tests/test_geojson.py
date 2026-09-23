"""GeoJSON 导入导出、校验与往返一致性测试。"""

import json

import numpy as np
import pytest

from wind_farm_opt.geo import (
    CoordinateSystemError,
    GeoJSONError,
    export_boundary_geojson,
    export_layout_geojson,
    import_site,
)
from wind_farm_opt.geo.geojson import parse_layout

from conftest import boundary_geojson, grid_points, layout_geojson


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# 正常导入
# --------------------------------------------------------------------------


def test_import_boundary_only(site_files):
    site = import_site(site_files["boundary"])
    assert site.turbines == []
    assert site.chain.target_crs.to_epsg() == 32650
    # 本地场界是以场址为中心的千米级坐标，且面积为平方米量级
    assert site.boundary.area > 5e6
    assert np.abs(site.boundary.vertices).max() < 6000
    assert site.roundtrip_summary["passed"] is True
    assert site.validation is None


def test_import_boundary_with_explicit_source_crs(tmp_path):
    doc = boundary_geojson(crs=None)
    path = _write(tmp_path, "b.geojson", doc)
    site = import_site(path, source_crs="WGS84")
    assert site.chain.source_crs.to_epsg() == 4326
    assert site.source_crs_declared_from.startswith("argument")


def test_import_site_with_layout(site_files):
    site = import_site(site_files["boundary"], layout_path=site_files["layout"],
                       min_spacing_m=630.0)
    assert len(site.turbines) == 12
    assert site.turbine_ids[:3] == ["WT-01", "WT-02", "WT-03"]
    assert site.validation.passed is True
    # 本地坐标与网格生成坐标在同一投影下
    assert site.local_positions.shape == (12, 2)
    assert np.all(np.isfinite(site.local_positions))


def test_cgcs2000_source(site_files, tmp_path):
    doc = boundary_geojson(crs="EPSG:4490")
    path = _write(tmp_path, "cgcs.geojson", doc)
    site = import_site(path)
    assert site.chain.source_crs.to_epsg() == 4490
    assert site.chain.target_crs.to_epsg() == 32650
    assert site.roundtrip_summary["max_error_m"] < 0.01


# --------------------------------------------------------------------------
# 拒绝：缺失坐标系
# --------------------------------------------------------------------------


def test_missing_crs_boundary_rejected(tmp_path):
    path = _write(tmp_path, "b.geojson", boundary_geojson(crs=None))
    with pytest.raises(CoordinateSystemError, match="缺少坐标参考系"):
        import_site(path)


def test_missing_crs_layout_rejected(site_files, tmp_path):
    ldoc = layout_geojson(grid_points(), crs=None)
    lpath = _write(tmp_path, "l.geojson", ldoc)
    with pytest.raises(CoordinateSystemError, match="缺少坐标参考系"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


def test_conflicting_crs_between_files_rejected(site_files, tmp_path):
    ldoc = layout_geojson(grid_points(), crs="EPSG:4490")
    lpath = _write(tmp_path, "l4490.geojson", ldoc)
    with pytest.raises(CoordinateSystemError, match="不一致"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


# --------------------------------------------------------------------------
# 拒绝：非有限坐标 / 混合维度 / 几何问题
# --------------------------------------------------------------------------


def test_non_finite_coordinate_rejected(site_files, tmp_path):
    pts = grid_points()
    pts[3][1] = float("nan")
    ldoc = layout_geojson(pts)
    lpath = _write(tmp_path, "nan.geojson", ldoc)
    with pytest.raises(GeoJSONError, match="有限数"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


def test_mixed_dimensions_rejected(site_files, tmp_path):
    ldoc = layout_geojson(grid_points()[:2])
    ldoc["features"][1]["geometry"]["coordinates"].append(15.0)  # 加 Z
    lpath = _write(tmp_path, "mix.geojson", ldoc)
    with pytest.raises(GeoJSONError, match="维度不一致"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


def test_three_d_coordinates_rejected(site_files, tmp_path):
    bdoc = boundary_geojson()
    ring = bdoc["features"][0]["geometry"]["coordinates"][0]
    bdoc["features"][0]["geometry"]["coordinates"] = [[p + [0.0] for p in ring]]
    path = _write(tmp_path, "b3d.geojson", bdoc)
    with pytest.raises(GeoJSONError, match="第 3 维"):
        import_site(path)


def test_unclosed_polygon_rejected(tmp_path):
    ring = [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11],
            [114.68, 41.11]]  # 未回到起点
    doc = {"type": "FeatureCollection",
           "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
           "features": [{"type": "Feature", "properties": {},
                         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    path = _write(tmp_path, "open.geojson", doc)
    with pytest.raises(GeoJSONError, match="未闭合"):
        import_site(path)


def test_geographic_range_guards_against_lonlat_mistake(tmp_path):
    # 把投影米制大数当成经纬度 -> 必须拒绝，避免“把经纬度当平面距离”的反面
    ring = [[300000, 4500000], [304000, 4500000], [304000, 4504000],
            [300000, 4504000], [300000, 4500000]]
    doc = {"type": "FeatureCollection",
           "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
           "features": [{"type": "Feature", "properties": {},
                         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    path = _write(tmp_path, "big.geojson", doc)
    with pytest.raises(CoordinateSystemError, match="经度"):
        import_site(path)


# --------------------------------------------------------------------------
# 拒绝：重复编号 / 缺编号
# --------------------------------------------------------------------------


def test_duplicate_turbine_ids_rejected(site_files, tmp_path):
    ldoc = layout_geojson(grid_points()[:3])
    ldoc["features"][2]["properties"]["turbine_id"] = "WT-01"
    ldoc["features"][2]["id"] = "WT-01"
    lpath = _write(tmp_path, "dup.geojson", ldoc)
    with pytest.raises(GeoJSONError, match="重复"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


def test_missing_turbine_id_rejected(site_files, tmp_path):
    ldoc = layout_geojson(grid_points()[:2], with_ids=False)
    # 去掉 Feature.id 与 properties 编号
    for f in ldoc["features"]:
        f["id"] = None
        f["properties"] = {}
    lpath = _write(tmp_path, "noid.geojson", ldoc)
    with pytest.raises(GeoJSONError, match="缺少编号"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


def test_parse_layout_accepts_chinese_id_key(site_files):
    doc = layout_geojson(grid_points()[:1])
    for f in doc["features"]:
        f["properties"] = {"编号": "A-01"}
    records = parse_layout(doc)
    assert records[0].turbine_id == "A-01"


# --------------------------------------------------------------------------
# 拒绝：跨带
# --------------------------------------------------------------------------


def test_cross_zone_site_rejected(tmp_path):
    ring = [[100.0, 40.0], [107.5, 40.0], [107.5, 41.0], [100.0, 41.0],
            [100.0, 40.0]]
    doc = {"type": "FeatureCollection",
           "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
           "features": [{"type": "Feature", "properties": {},
                         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    path = _write(tmp_path, "wide.geojson", doc)
    with pytest.raises(CoordinateSystemError, match="跨带"):
        import_site(path)
    # 显式允许跨带时可以导入（仅记录风险）
    site = import_site(path, allow_cross_zone=True)
    assert site.chain.zone_check.cross_zone is True


# --------------------------------------------------------------------------
# 拒绝：布局边界 / 间距检查
# --------------------------------------------------------------------------


def test_layout_outside_boundary_rejected(site_files, tmp_path):
    pts = [[114.90, 41.30]]  # 远离场址
    ldoc = layout_geojson(pts)
    lpath = _write(tmp_path, "out.geojson", ldoc)
    with pytest.raises(GeoJSONError, match="场界之外"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=10.0)


def test_layout_spacing_violation_rejected(site_files, tmp_path):
    # 两台相距约 8 m 的机位
    pts = [[114.705, 41.092], [114.7051, 41.092]]
    ldoc = layout_geojson(pts)
    lpath = _write(tmp_path, "close.geojson", ldoc)
    with pytest.raises(GeoJSONError, match="间距不足"):
        import_site(site_files["boundary"], layout_path=lpath, min_spacing_m=630.0)


def test_layout_spacing_just_at_limit_passes(site_files):
    # 规则网格间距 700m > 630m，应当通过
    site = import_site(site_files["boundary"], layout_path=site_files["layout"],
                       min_spacing_m=630.0)
    assert len(site.validation.spacing_violations) == 0


# --------------------------------------------------------------------------
# 往返一致性：导出 -> 再导入
# --------------------------------------------------------------------------


def test_full_roundtrip_preserves_ids_and_positions(site_files, tmp_path):
    site = import_site(site_files["boundary"], layout_path=site_files["layout"],
                       min_spacing_m=630.0)
    b_out = str(tmp_path / "b_export.geojson")
    l_out = str(tmp_path / "l_export.geojson")

    export_boundary_geojson(site.chain, site.boundary.vertices, b_out)
    export_layout_geojson(
        site.chain, site.local_positions, site.turbine_ids, l_out,
        source_attributes=[t.attributes for t in site.turbines],
        computed_attributes=[{"net_aep_mwh": 1234.5} for _ in site.turbines],
    )

    site2 = import_site(b_out, layout_path=l_out, min_spacing_m=630.0)

    # 编号顺序保持
    assert site2.turbine_ids == site.turbine_ids
    # 本地米制坐标往返一致（9 位经纬度取整带来亚毫米误差，容差 5cm 内）
    np.testing.assert_allclose(
        site2.local_positions, site.local_positions, atol=0.05
    )
    # 原始属性保留，计算属性带 wfo_ 前缀
    doc = json.loads(tmp_path.joinpath("l_export.geojson").read_text())
    props = doc["features"][0]["properties"]
    assert props["turbine_id"] == "WT-01"
    assert props["wfo_net_aep_mwh"] == 1234.5
    # 导出 CRS 与源一致
    assert doc["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::4326"


def test_roundtrip_without_optimization_keeps_geographic_input(site_files, tmp_path):
    site = import_site(site_files["boundary"], layout_path=site_files["layout"],
                       min_spacing_m=630.0)
    l_out = str(tmp_path / "l_rt.geojson")
    export_layout_geojson(site.chain, site.local_positions, site.turbine_ids, l_out)

    original = np.asarray([t.source_coords for t in site.turbines])
    exported_doc = json.loads(tmp_path.joinpath("l_rt.geojson").read_text())
    exported = np.asarray(
        [f["geometry"]["coordinates"] for f in exported_doc["features"]]
    )
    # 经纬度层面的往返误差应远小于 1e-6 度（约 0.1m）
    np.testing.assert_allclose(exported, original, atol=1e-7)


def test_provenance_dict_contents(site_files):
    site = import_site(site_files["boundary"], layout_path=site_files["layout"],
                       min_spacing_m=630.0)
    prov = site.provenance_dict()
    assert prov["boundary_file"].endswith("boundary.geojson")
    assert prov["projection"]["source_crs"]["code"] == "EPSG:4326"
    assert prov["projection"]["target_crs"]["code"] == "EPSG:32650"
    assert prov["zone_check"]["cross_zone"] is False
    assert prov["roundtrip_verification"]["passed"] is True
    assert prov["layout_import_validation"]["passed"] is True
    assert prov["turbine_ids"] == [f"WT-{i:02d}" for i in range(1, 13)]


def test_exported_layout_is_valid_geojson_with_crs(site_files, tmp_path):
    site = import_site(site_files["boundary"], layout_path=site_files["layout"],
                       min_spacing_m=630.0)
    out = str(tmp_path / "out.geojson")
    export_layout_geojson(site.chain, site.local_positions, site.turbine_ids, out)
    doc = json.loads(tmp_path.joinpath("out.geojson").read_text())
    assert doc["type"] == "FeatureCollection"
    assert doc["crs"]["type"] == "name"
    assert len(doc["features"]) == 12
    for f in doc["features"]:
        assert f["geometry"]["type"] == "Point"
        assert len(f["geometry"]["coordinates"]) == 2
