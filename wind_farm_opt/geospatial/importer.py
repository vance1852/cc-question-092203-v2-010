"""GeoJSON 场界与机位布局导入编排。

完整导入管线：
1. 解析、结构校验（validation）；
2. 解析/要求源坐标参考系（crs.resolve_source_crs）；
3. 选择场址适配的米制计算坐标系（自动选带或显式指定）；
4. 拒绝明显跨带场址（transform.CoordinateTransformer.verify_import）；
5. 正变换到本地米制坐标，构造 :class:`SiteBoundary`；
6. 导入布局必须先通过边界检查与最小间距检查；
7. 输出 :class:`ImportedSite`，携带变压器、编号、属性与转换摘要。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pyproj import CRS

from .crs import (
    MetricCRSRecommendation,
    build_crs,
    recommend_metric_crs,
    resolve_source_crs,
    validate_explicit_metric_crs,
)
from .transform import CoordinateTransformer, TransformSummary
from .validation import (
    BoundaryGeometry,
    GeoJSONValidationError,
    load_geojson_file,
    load_geojson_text,
    parse_geojson,
)
from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import check_min_spacing


class LayoutConstraintError(ValueError):
    """导入布局未通过边界或间距检查。"""


# 经纬度合法范围（用于无 CRS 时判断是否可按 RFC 7946 默认 WGS84）。
_LON_MAX, _LAT_MAX = 180.0, 90.0

# 边界检查容差（米）：正变换后的机位相对原多边形可能有亚毫米级数值偏移，
# 距边界小于该值视为在界内。
DEFAULT_BOUNDARY_TOLERANCE_M = 0.5


@dataclass
class ImportedSite:
    """GeoJSON 导入结果（本地米制坐标）。

    Attributes
    ----------
    boundary
        米制坐标系下的场地边界（含孔洞）。
    turbine_ids
        机位编号，顺序与 ``positions`` 一致。
    positions
        机位本地米制坐标 (N, 2)。
    turbine_properties
        各机位的原始属性（不含已提升为编号的字段）。
    boundary_properties
        场界要素属性。
    transformer
        正反变换变压器，供导出使用。
    summary
        转换摘要。
    source_dimension
        源数据坐标维度（2 或 3）。
    boundary_tolerance_m
        边界检查采用的容差。
    near_boundary_ids
        落在边界容差带内的机位编号（已接受但需记录）。
    """

    boundary: SiteBoundary
    turbine_ids: list[str]
    positions: np.ndarray
    turbine_properties: list[dict]
    boundary_properties: dict
    transformer: CoordinateTransformer
    summary: TransformSummary
    source_dimension: int
    boundary_tolerance_m: float
    near_boundary_ids: list[str] = field(default_factory=list)


def _looks_geographic(points: np.ndarray) -> bool:
    """判断坐标是否全部落在合法经纬度范围内。

    无 crs 成员时 RFC 7946 规定坐标即 WGS84 经纬度，因此只要全部坐标
    落在合法经纬度范围内就按默认处理；明显为投影量级的大坐标会越界，
    从而触发 CRSMissingError，要求显式声明坐标系。
    """
    if points is None or len(points) == 0:
        return False
    lon_ok = np.all((points[:, 0] >= -_LON_MAX) & (points[:, 0] <= _LON_MAX))
    lat_ok = np.all((points[:, 1] >= -_LAT_MAX) & (points[:, 1] <= _LAT_MAX))
    return bool(lon_ok and lat_ok)


def import_geojson(
    path: Optional[str] = None,
    *,
    text: Optional[str] = None,
    source_crs: Optional[str] = None,
    metric_crs: Optional[str] = None,
    metric_preference: str = "auto",
    min_spacing_m: Optional[float] = None,
    boundary_tolerance_m: float = DEFAULT_BOUNDARY_TOLERANCE_M,
    roundtrip_tolerance_m: float = 1e-3,
    require_turbines: bool = True,
) -> ImportedSite:
    """导入场界（可含机位布局）GeoJSON。

    Parameters
    ----------
    path
        GeoJSON 文件路径（与 ``text`` 二选一）。
    text
        GeoJSON 文本。
    source_crs
        显式源坐标系（覆盖文件 crs 成员），如 ``EPSG:4326``。
    metric_crs
        显式米制计算坐标系，如 ``EPSG:32650``；默认按场址自动选带。
    metric_preference
        自动选带偏好：auto / cgcs2000 / utm。
    min_spacing_m
        机位最小允许间距（米）。提供时对导入布局执行间距检查。
    boundary_tolerance_m
        边界判定容差（米）。
    roundtrip_tolerance_m
        坐标往返残差容差（米）。
    require_turbines
        为 True 时文件必须包含至少一个机位。
    """
    if path is not None:
        data = load_geojson_file(path)
    elif text is not None:
        data = load_geojson_text(text)
    else:
        raise ValueError("必须提供 path 或 text")

    parsed = parse_geojson(data)

    if parsed.boundary is None:
        raise GeoJSONValidationError("GeoJSON 中缺少场界多边形（Polygon/MultiPolygon）")
    if require_turbines and not parsed.turbines:
        raise GeoJSONValidationError("GeoJSON 中缺少机位要素（Point）")

    # 1) 源坐标系
    source = resolve_source_crs(
        crs_member=parsed.crs_member,
        cli_override=source_crs,
        looks_geographic=_looks_geographic(parsed.all_source_points),
    )

    # 场址由场界定义：自动选带与跨带判定只依据场界顶点，
    # 个别界外/错误机位不改变场址所属投影带。
    boundary_src_points = np.vstack(parsed.boundary.rings)

    # 2) 选米制计算坐标系：先把场界源点转到地理坐标求范围，再选带
    if metric_crs is not None:
        target = build_crs(metric_crs)
        validate_explicit_metric_crs(target)
        metric_obj: MetricCRSRecommendation | CRS = target
    else:
        boundary_geo_points = _to_lonlat(source.crs, boundary_src_points)
        from .crs import build_extent
        extent = build_extent(boundary_geo_points)
        metric_obj = recommend_metric_crs(extent, metric_preference)

    # 3) 变压器 + 跨带（仅场界）+ 往返校验（场界顶点和全部机位）
    transformer = CoordinateTransformer(
        source, metric_obj, roundtrip_tolerance_m=roundtrip_tolerance_m
    )
    transformer.verify_import(
        parsed.all_source_points,
        cross_zone_points=boundary_src_points,
        cross_zone_check=True,
    )

    # 4) 变换场界与机位
    boundary = _build_boundary(parsed.boundary, transformer)
    ids = [t.turbine_id for t in parsed.turbines]
    props = [dict(t.properties) for t in parsed.turbines]
    src_xy = np.vstack([t.source_xy for t in parsed.turbines]) if parsed.turbines \
        else np.zeros((0, 2), dtype=np.float64)
    positions = transformer.forward(src_xy) if len(src_xy) else src_xy.copy()

    # 5) 边界与间距检查
    near_boundary: list[str] = []
    if len(positions):
        near_boundary = _check_layout_inside(
            ids, positions, boundary, boundary_tolerance_m
        )
        if min_spacing_m is not None:
            _check_layout_spacing(ids, positions, float(min_spacing_m))

    return ImportedSite(
        boundary=boundary,
        turbine_ids=ids,
        positions=positions,
        turbine_properties=props,
        boundary_properties=dict(parsed.boundary.properties),
        transformer=transformer,
        summary=transformer.summary,
        source_dimension=parsed.dimension,
        boundary_tolerance_m=boundary_tolerance_m,
        near_boundary_ids=near_boundary,
    )


def _to_lonlat(crs: CRS, points: np.ndarray) -> np.ndarray:
    if crs.is_geographic:
        return np.asarray(points, dtype=np.float64)
    from pyproj import Transformer
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(points[:, 0], points[:, 1])
    out = np.column_stack([np.asarray(lon), np.asarray(lat)])
    if not np.all(np.isfinite(out)):
        raise GeoJSONValidationError("源投影坐标无法转换为经纬度（越界或坐标系不匹配）")
    return out


def _build_boundary(
    geom: BoundaryGeometry,
    transformer: CoordinateTransformer,
) -> SiteBoundary:
    outer = transformer.forward(geom.rings[0])
    holes = [transformer.forward(ring) for ring in geom.rings[1:]]
    return SiteBoundary(outer, holes=holes)


def _distance_to_boundary(boundary: SiteBoundary, point: np.ndarray) -> float:
    return float(np.linalg.norm(point - boundary.project_to_boundary(point)))


def _check_layout_inside(
    ids: list[str],
    positions: np.ndarray,
    boundary: SiteBoundary,
    tolerance_m: float,
) -> list[str]:
    """边界检查：拒绝界外机位，记录贴边机位。"""
    outside = []
    near = []
    for i, (tid, pt) in enumerate(zip(ids, positions)):
        if boundary.contains_point(pt):
            # 已在界内（含边），若非常贴近边界也记录下来。
            if _distance_to_boundary(boundary, pt) <= tolerance_m:
                near.append(tid)
            continue
        dist = _distance_to_boundary(boundary, pt)
        if dist <= tolerance_m:
            near.append(tid)
        else:
            outside.append((tid, i, dist))

    if outside:
        lines = [
            f"机位 {tid}(序号{idx}) 超出场界 {dist:.2f} m"
            for tid, idx, dist in outside[:10]
        ]
        more = "" if len(outside) <= 10 else f" 等共 {len(outside)} 台"
        raise LayoutConstraintError(
            "导入布局未通过边界检查:\n  " + ";\n  ".join(lines) + more
        )
    return near


def _check_layout_spacing(
    ids: list[str],
    positions: np.ndarray,
    min_spacing_m: float,
) -> None:
    valid, pairs = check_min_spacing(positions, min_spacing_m)
    if valid:
        return
    lines = []
    for i, j in pairs[:10]:
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        lines.append(
            f"机位 {ids[i]} 与 {ids[j]} 间距 {dist:.2f} m < {min_spacing_m:.2f} m"
        )
    more = "" if len(pairs) <= 10 else f" 等共 {len(pairs)} 对"
    raise LayoutConstraintError(
        "导入布局未通过最小间距检查:\n  " + ";\n  ".join(lines) + more
    )
