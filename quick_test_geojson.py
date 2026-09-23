"""GeoJSON 导入导出快速测试。

覆盖：
- WGS84 经纬度源自动选择 CGCS2000 三度带米制坐标系；
- 多边形、机位编号与属性经导入→（模拟优化）→导出→再导入往返一致；
- 缺失坐标系、混合维度、非有限坐标、重复编号、跨带场址、
  界外机位、间距不足等输入被拒绝。

运行：
    PYTHONPATH=. python3 quick_test_geojson.py
"""

import json
import os
import tempfile

import numpy as np
from pyproj import Transformer

from wind_farm_opt.geospatial import (
    CRSMissingError,
    CrossZoneError,
    GeoJSONValidationError,
    LayoutConstraintError,
    build_export_geojson,
    export_geojson,
    import_geojson,
)

PASS = 0


def ok(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1
    print(f"  ✓ {msg}")


def ring(points):
    return points + [points[0]]


def make_site_geojson(lon0=114.9, lat0=41.0, half=1500.0, spacing=900.0):
    """在 CGCS2000 三度带中央经线114 附近构造 3x2 机位的 WGS84 GeoJSON。"""
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)
    inv = Transformer.from_crs("EPSG:4547", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon0, lat0)

    poly = [list(inv.transform(cx + dx, cy + dy))
            for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]]
    feats = [{
        "type": "Feature",
        "properties": {"sfp_role": "site_boundary", "site_name": "测试场址"},
        "geometry": {"type": "Polygon", "coordinates": [ring(poly)]},
    }]
    idx = 1
    for gx in (-spacing, 0.0, spacing):
        for gy in (-spacing * 0.6, spacing * 0.6):
            lon, lat = inv.transform(cx + gx, cy + gy)
            feats.append({
                "type": "Feature",
                "id": f"WT-{idx:03d}",
                "properties": {"model": "V126-3.45MW", "group": "A"},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            })
            idx += 1
    return {"type": "FeatureCollection", "features": feats}


def expect_error(exc_type, build_text, name, **kw):
    text = build_text if isinstance(build_text, str) else build_text()
    try:
        import_geojson(text=text, require_turbines=False, **kw)
    except exc_type as e:
        print(f"  ✓ 拒绝{name}: {type(e).__name__}")
        global PASS
        PASS += 1
        return
    raise AssertionError(f"未拒绝{name}")


def main() -> None:
    print("=" * 64)
    print("GeoJSON 导入导出测试")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmp:
        # 1) 正常往返
        print("\n1. 导入与自动选带")
        gj = make_site_geojson()
        src_path = os.path.join(tmp, "site.geojson")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump(gj, f)

        site = import_geojson(path=src_path, min_spacing_m=630.0)
        ok(site.summary.metric_crs == "EPSG:4547",
           f"中国境内自动选择 CGCS2000 三度带 ({site.summary.metric_crs})")
        ok(site.summary.source_crs == "EPSG:4326", "源坐标系识别为 EPSG:4326")
        ok(len(site.turbine_ids) == 6, f"读入 6 个机位编号 ({site.turbine_ids})")
        ok(abs(site.boundary.area / 1e6 - 9.0) < 1e-6,
           f"场界面积投影正确 {site.boundary.area/1e6:.4f} km²")
        ok(site.summary.roundtrip_max_error_m <= site.summary.roundtrip_tolerance_m,
           f"导入往返残差 {site.summary.roundtrip_max_error_m:.2e} m ≤ 1 mm")

        print("\n2. 优化后导出与再导入")
        rng = np.random.default_rng(0)
        moved = site.positions + rng.normal(0, 20.0, site.positions.shape)
        out_path = os.path.join(tmp, "opt.geojson")
        export_geojson(
            site, moved, out_path,
            turbine_attributes={tid: {"wake_loss_pct": 1.5} for tid in site.turbine_ids},
        )
        again = import_geojson(path=out_path, min_spacing_m=630.0)
        ok(again.turbine_ids == site.turbine_ids, "机位编号顺序保持一致")
        ok(float(np.abs(again.positions - moved).max()) < 1e-6,
           "导出→再导入地理位置往返一致 (<1e-6 m)")
        ok(again.turbine_properties[0].get("group") == "A", "自定义属性完整保留")
        ok(again.turbine_properties[0].get("wake_loss_pct") == 1.5,
           "优化结果属性已写入机位")
        with open(out_path, encoding="utf-8") as f:
            exported = json.load(f)
        ok(exported["crs"]["properties"]["name"] == "EPSG:4326",
           "导出文件声明源坐标系 crs 成员")
        ok("sfp_coordinate_transform" in exported, "导出文件附带转换摘要")

        # 2b) 转换摘要可序列化且字段齐全
        summary = exported["sfp_coordinate_transform"]
        for key in ("source_crs", "metric_crs", "forward_chain", "inverse_chain",
                    "roundtrip_max_error_m", "source_bounds_lonlat"):
            ok(key in summary, f"转换摘要包含 {key}")

        # 3) 拒绝路径
        print("\n3. 非法输入拒绝")
        poly = ring([[114.88, 40.99], [114.92, 40.99],
                     [114.92, 41.01], [114.88, 41.01]])

        expect_error(
            CRSMissingError,
            lambda: json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {}, "geometry":
                 {"type": "Polygon", "coordinates":
                  [ring([[500000, 4540000], [503000, 4540000],
                         [503000, 4543000], [500000, 4543000]])]}}]}),
            "缺失坐标系的投影坐标",
        )
        expect_error(
            GeoJSONValidationError,
            lambda: json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [poly]}},
                {"type": "Feature", "id": "T1", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [114.9, 41.0, 0.0]}},
            ]}),
            "混合 2D/3D 维度",
        )
        expect_error(
            GeoJSONValidationError,
            '{"type":"FeatureCollection","features":[{"type":"Feature",'
            '"properties":{},"geometry":{"type":"Polygon","coordinates":'
            '[[[114.88,40.99],[Infinity,40.99],[114.92,41.01],'
            '[114.88,41.01],[114.88,40.99]]]}}]}',
            "Infinity 坐标",
        )
        expect_error(
            GeoJSONValidationError,
            lambda: json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [poly]}},
                {"type": "Feature", "id": "T1", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [114.9, 41.0]}},
                {"type": "Feature", "id": "T1", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [114.901, 41.0]}},
            ]}),
            "重复机位编号",
        )
        expect_error(
            CrossZoneError,
            lambda: json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {}, "geometry":
                 {"type": "Polygon", "coordinates":
                  [ring([[112.3, 41.0], [112.7, 41.0],
                         [112.7, 41.2], [112.3, 41.2]])]}}]}),
            "跨越三度带边界场址",
        )
        expect_error(
            LayoutConstraintError,
            lambda: json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [poly]}},
                {"type": "Feature", "id": "T1", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [115.5, 41.0]}},
            ]}),
            "界外机位",
        )
        expect_error(
            LayoutConstraintError,
            lambda: json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [poly]}},
                {"type": "Feature", "id": "T1", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [114.895, 41.0]}},
                {"type": "Feature", "id": "T2", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [114.896, 41.0]}},
            ]}),
            "间距不足机位", min_spacing_m=630.0,
        )

    print("\n" + "=" * 64)
    print(f"GeoJSON 测试全部通过！共 {PASS} 项断言。")
    print("=" * 64)


if __name__ == "__main__":
    main()
