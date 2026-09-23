"""坐标参考系与投影链测试。"""

import math

import numpy as np
import pytest
from pyproj import CRS

from wind_farm_opt.geo.crs import (
    CoordinateSystemError,
    build_projection_chain,
    check_zone,
    parse_crs,
    resolve_source_crs,
    suggest_utm_crs,
    utm_zone_for_longitude,
)


def test_utm_zone_numbering():
    assert utm_zone_for_longitude(-180.0) == 1
    assert utm_zone_for_longitude(-174.1) == 1
    assert utm_zone_for_longitude(114.7) == 50
    assert utm_zone_for_longitude(179.9) == 60


def test_parse_crs_aliases():
    assert parse_crs("WGS84").to_epsg() == 4326
    assert parse_crs("wgs 84").to_epsg() == 4326
    assert parse_crs("EPSG:4326").to_epsg() == 4326
    assert parse_crs("urn:ogc:def:crs:EPSG::4490").to_epsg() == 4490
    assert parse_crs("CGCS2000").to_epsg() == 4490


def test_parse_crs_missing_rejected():
    with pytest.raises(CoordinateSystemError):
        parse_crs("")
    with pytest.raises(CoordinateSystemError):
        parse_crs("   ")


def test_parse_crs_unknown_rejected():
    with pytest.raises(CoordinateSystemError):
        parse_crs("EPSG:999999")


def test_resolve_source_crs_precedence(tmp_path):
    doc = {"crs": {"type": "name", "properties": {"name": "EPSG:4326"}}}
    crs, source = resolve_source_crs(doc)
    assert crs.to_epsg() == 4326
    assert source.startswith("geojson")

    # 显式参数优先，与内嵌一致时通过
    crs2, _ = resolve_source_crs(doc, "urn:ogc:def:crs:EPSG::4326")
    assert crs2.to_epsg() == 4326

    # 显式与内嵌不一致 -> 拒绝
    with pytest.raises(CoordinateSystemError):
        resolve_source_crs(doc, "EPSG:4490")

    # 两者都缺失 -> 拒绝
    with pytest.raises(CoordinateSystemError):
        resolve_source_crs({})


def test_suggest_utm_north_south():
    crs_n, zone_n, north_n = suggest_utm_crs(
        np.array([114.6, 114.8]), np.array([41.0, 41.2])
    )
    assert zone_n == 50 and north_n is True and crs_n.to_epsg() == 32650

    crs_s, zone_s, north_s = suggest_utm_crs(
        np.array([-58.5, -58.3]), np.array([-34.7, -34.5])
    )
    assert zone_s == 21 and north_s is False and crs_s.to_epsg() == 32721


def test_cross_zone_rejected():
    lons = np.array([100.0, 107.5])
    with pytest.raises(CoordinateSystemError, match="跨带"):
        check_zone(lons)
    result = check_zone(lons, allow_cross_zone=True)
    assert result.cross_zone is True
    assert result.utm_zones == [47, 48]


def test_chain_geographic_roundtrip_accuracy():
    src = CRS.from_epsg(4326)
    boundary = np.array(
        [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11], [114.68, 41.11]]
    )
    chain = build_projection_chain(src, boundary)
    assert chain.target_crs.to_epsg() == 32650
    assert chain.target_selection == "auto_utm"
    assert chain.zone == 50

    # 本地坐标以场址中心为原点，量级应为千米
    local = chain.to_local(boundary)
    assert np.abs(local).max() < 5000.0

    # 投影坐标 easting 应为数十万米的 UTM 坐标
    projected = chain.to_projected(boundary)
    assert 300000 < projected[:, 0].mean() < 400000

    summary = chain.verify_roundtrip(boundary, tolerance_m=0.05)
    assert summary["max_error_m"] < 1e-6
    assert summary["passed"] is True


def test_chain_roundtrip_failure_raises():
    src = CRS.from_epsg(4326)
    boundary = np.array([[114.68, 41.07], [114.73, 41.07], [114.73, 41.11]])
    chain = build_projection_chain(src, boundary)
    with pytest.raises(CoordinateSystemError, match="往返误差"):
        chain.verify_roundtrip(boundary, tolerance_m=-1.0)


def test_manual_target_crs_mismatch_rejected():
    src = CRS.from_epsg(4326)
    boundary = np.array(
        [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11], [114.68, 41.11]]
    )
    # 场址在 50 带，强行指定 49 带 -> 拒绝
    with pytest.raises(CoordinateSystemError, match="不匹配"):
        build_projection_chain(src, boundary, target_crs="EPSG:32649")


def test_target_crs_must_be_metric_projected():
    src = CRS.from_epsg(4326)
    boundary = np.array(
        [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11], [114.68, 41.11]]
    )
    with pytest.raises(CoordinateSystemError, match="投影坐标系"):
        build_projection_chain(src, boundary, target_crs="EPSG:4326")


def test_projected_source_uses_same_metric_crs():
    # 源已是 UTM 50N 投影坐标时，目标带应沿用源投影
    src = CRS.from_epsg(32650)
    boundary_geo = np.array(
        [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11], [114.68, 41.11]]
    )
    chain = build_projection_chain(src, boundary_geo)
    assert chain.target_crs.to_epsg() == 32650
    assert chain.target_selection == "source_projected"


def test_chain_summary_records_transforms():
    src = CRS.from_epsg(4326)
    boundary = np.array(
        [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11], [114.68, 41.11]]
    )
    chain = build_projection_chain(src, boundary)
    summary = chain.as_dict()
    assert summary["source_crs"]["code"] == "EPSG:4326"
    assert summary["target_crs"]["code"] == "EPSG:32650"
    assert "utm" in summary["target_crs"]["zone"].lower()
    assert summary["local_frame"]["enabled"] is True
    assert "proj4" in summary["forward_transform_proj4"].lower() or "+proj" in summary["forward_transform_proj4"]
    assert math.isfinite(summary["local_frame"]["origin_easting_m"])


def test_no_local_frame_keeps_projected_coordinates():
    src = CRS.from_epsg(4326)
    boundary = np.array(
        [[114.68, 41.07], [114.73, 41.07], [114.73, 41.11], [114.68, 41.11]]
    )
    chain = build_projection_chain(src, boundary, use_local_frame=False)
    local = chain.to_local(boundary)
    projected = chain.to_projected(boundary)
    np.testing.assert_allclose(local, projected)
    np.testing.assert_allclose(chain.local_origin, [0.0, 0.0])
