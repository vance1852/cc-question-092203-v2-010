"""pytest 固定装置：在临时目录中构造带 CRS 的 GeoJSON 测试数据。"""

import json

import pytest

# 河北张北附近（UTM 50N），约 3.5km x 3km 的不规则场界
CENTER_LON = 114.705
CENTER_LAT = 41.092

RING_LONLAT = [
    [CENTER_LON - 0.0215, CENTER_LAT - 0.0140],
    [CENTER_LON + 0.0200, CENTER_LAT - 0.0145],
    [CENTER_LON + 0.0220, CENTER_LAT + 0.0020],
    [CENTER_LON + 0.0120, CENTER_LAT + 0.0145],
    [CENTER_LON - 0.0110, CENTER_LAT + 0.0150],
    [CENTER_LON - 0.0225, CENTER_LAT + 0.0030],
]


def boundary_geojson(crs: str = "urn:ogc:def:crs:EPSG::4326") -> dict:
    ring = RING_LONLAT + [RING_LONLAT[0]]
    doc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"site_name": "测试场址"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    if crs is not None:
        doc["crs"] = {"type": "name", "properties": {"name": crs}}
    return doc


def layout_geojson(points_lonlat, crs: str = "urn:ogc:def:crs:EPSG::4326",
                   with_ids=True, extra_props=None) -> dict:
    features = []
    for i, (lon, lat) in enumerate(points_lonlat):
        props = dict(extra_props or {})
        if with_ids:
            props["turbine_id"] = f"WT-{i + 1:02d}"
        features.append(
            {
                "type": "Feature",
                "id": f"WT-{i + 1:02d}" if with_ids else None,
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    doc = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        doc["crs"] = {"type": "name", "properties": {"name": crs}}
    return doc


def grid_points(n_rows=3, n_cols=4, spacing_m=700.0):
    """在 UTM 50N 投影坐标系中生成规则网格机位并转回经纬度。"""
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(CENTER_LON, CENTER_LAT)
    points = []
    for r in range(n_rows):
        for c in range(n_cols):
            e = cx + (c - (n_cols - 1) / 2.0) * spacing_m
            n = cy + (r - (n_rows - 1) / 2.0) * spacing_m
            points.append(list(to_wgs.transform(e, n)))
    return points


@pytest.fixture
def site_files(tmp_path):
    bpath = tmp_path / "boundary.geojson"
    bpath.write_text(json.dumps(boundary_geojson()), encoding="utf-8")

    lpath = tmp_path / "layout.geojson"
    lpath.write_text(
        json.dumps(layout_geojson(grid_points())), encoding="utf-8"
    )

    empty_layout = tmp_path / "layout_empty.geojson"
    return {
        "tmp": tmp_path,
        "boundary": str(bpath),
        "layout": str(lpath),
        "empty_layout": str(empty_layout),
    }
