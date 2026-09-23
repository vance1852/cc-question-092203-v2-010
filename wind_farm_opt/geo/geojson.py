"""GeoJSON 场界与机位布局的导入导出。

本模块把勘测交付的 GeoJSON（经纬度或投影坐标 + 显式坐标参考系）校验后，
经 :mod:`wind_farm_opt.geo.crs` 的投影链转换为本地米制坐标，供原有
约束、优化与绘图模块使用；优化完成后再把机位编号、属性与计算结果
反变换导出为相同坐标参考系的 GeoJSON。

校验规则（任一不满足即拒绝并给出明确原因）：

* GeoJSON 必须携带或由参数显式给出坐标参考系；
* 坐标必须全部为有限数，且维度一致（不允许 2D/3D 混用）；
* 机位编号必须存在且唯一；
* 场界必须是单个闭合 Polygon（≥3 个不重复顶点）；
* 导入布局必须先通过场界包含检查与最小间距检查。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from .crs import (
    CoordinateSystemError,
    ProjectionChain,
    build_projection_chain,
    crs_code,
    resolve_source_crs,
)


class GeoJSONError(ValueError):
    """GeoJSON 结构、坐标或编号不合规。"""


# 机位编号候选属性名（按优先级）
_ID_KEYS = ("id", "turbine_id", "turbineId", "turbineID", "name", "编号", "机位编号")
# 导出时写入的编号属性名
_EXPORT_ID_KEY = "turbine_id"

# 坐标输出精度：地理坐标 9 位小数约 0.1 mm，投影坐标 4 位小数为 0.1 mm
_GEO_DECIMALS = 9
_PROJECTED_DECIMALS = 4


# --------------------------------------------------------------------------
# 基础读取与坐标校验
# --------------------------------------------------------------------------


def read_geojson(path: str) -> dict:
    """读取并解析 GeoJSON 文件（UTF-8）。"""
    if not os.path.exists(path):
        raise GeoJSONError(f"GeoJSON 文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise GeoJSONError(f"文件不是合法的 JSON/GeoJSON: {path}（{exc}）") from exc
    if not isinstance(doc, dict) or "type" not in doc:
        raise GeoJSONError(f"文件缺少 GeoJSON type 成员: {path}")
    return doc


def _validate_position(pos: Any, *, context: str) -> list[float]:
    """校验单个坐标位置：必须是全有限数的 2D 数组。"""
    if not isinstance(pos, (list, tuple)):
        raise GeoJSONError(f"{context} 的坐标必须是数组，实际为 {type(pos).__name__}")
    if len(pos) < 2:
        raise GeoJSONError(f"{context} 的坐标至少需要 2 个分量，得到 {len(pos)} 个")
    if len(pos) > 2:
        # 全三维与二维混用具“混合维度”错误统一在调用处统计；这里先拒绝带 Z
        raise GeoJSONError(
            f"{context} 的坐标包含第 3 维（疑似高程/海拔 {pos[2]!r}）；"
            "机位与场界只接受二维平面/经纬度坐标，请去除高程分量"
        )
    values = []
    for axis, v in zip(("x/经度", "y/纬度"), pos):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise GeoJSONError(f"{context} 的{axis}不是数值: {v!r}")
        if not math.isfinite(float(v)):
            raise GeoJSONError(f"{context} 的{axis}不是有限数: {v!r}")
        values.append(float(v))
    return values


def _validate_ring(ring: Any, *, context: str) -> np.ndarray:
    """校验 Polygon 环并返回去掉闭合点后的 (M, 2) 顶点数组。"""
    if not isinstance(ring, list):
        raise GeoJSONError(f"{context} 的环必须是坐标数组")
    if len(ring) < 4:
        raise GeoJSONError(
            f"{context} 的多边形环至少需要 4 个位置（含闭合点），得到 {len(ring)} 个"
        )

    dims = {len(p) if isinstance(p, (list, tuple)) else None for p in ring}
    if len(dims) > 1:
        raise GeoJSONError(f"{context} 的坐标维度不一致（2D/3D 混用）: {sorted(dims)}")

    points = [_validate_position(p, context=context) for p in ring]
    arr = np.asarray(points, dtype=np.float64)

    first, last = arr[0], arr[-1]
    if not np.allclose(first, last, rtol=0.0, atol=1e-9):
        raise GeoJSONError(
            f"{context} 的多边形环未闭合：首点 {first.tolist()} 与末点 "
            f"{last.tolist()} 不一致"
        )

    open_arr = arr[:-1]
    # 相邻重复顶点会破坏面积与射线判断
    for i in range(len(open_arr)):
        j = (i + 1) % len(open_arr)
        if np.allclose(open_arr[i], open_arr[j], rtol=0.0, atol=1e-9):
            raise GeoJSONError(f"{context} 的多边形环存在相邻重复顶点（第 {i} 个）")

    return open_arr


def _check_geographic_ranges(coords: np.ndarray, *, context: str) -> None:
    """对地理坐标做经纬度取值范围检查。"""
    lons, lats = coords[:, 0], coords[:, 1]
    if np.any(lons < -180.0) or np.any(lons > 180.0):
        bad = coords[(lons < -180.0) | (lons > 180.0)]
        raise CoordinateSystemError(
            f"{context} 出现超出 [-180, 180] 的经度值（如 {bad[0].tolist()}），"
            "坐标参考系可能被错误声明：投影米制坐标不能按经纬度读入"
        )
    if np.any(lats < -90.0) or np.any(lats > 90.0):
        bad = coords[(lats < -90.0) | (lats > 90.0)]
        raise CoordinateSystemError(
            f"{context} 出现超出 [-90, 90] 的纬度值（如 {bad[0].tolist()}），"
            "坐标参考系可能被错误声明"
        )


# --------------------------------------------------------------------------
# 场界解析
# --------------------------------------------------------------------------


def parse_boundary(doc: dict) -> tuple[np.ndarray, dict]:
    """从 GeoJSON 文档解析场界外环。

    接受 ``Polygon``（不允许带洞）或仅含一个 ``Polygon`` 的
    ``Feature`` / ``FeatureCollection``。

    Returns
    -------
    tuple[np.ndarray, dict]
        (源 CRS 下的外环顶点 (M, 2)，已去闭合点；属性字典)
    """
    geom, props = _extract_single_geometry(doc, expected=("Polygon",))

    rings = geom.get("coordinates")
    if not isinstance(rings, list) or not rings:
        raise GeoJSONError("Polygon 缺少 coordinates 环数组")
    if len(rings) > 1:
        raise GeoJSONError(
            f"场界 Polygon 含 {len(rings)} 个环（存在洞/内岛），当前场地模型"
            "只支持单一边界，请去除内环或分别处理"
        )

    vertices = _validate_ring(rings[0], context="场界 Polygon")
    if vertices.shape[0] < 3:
        raise GeoJSONError("场界多边形至少需要 3 个不重复顶点")
    return vertices, dict(props)


def _extract_single_geometry(
    doc: dict, *, expected: tuple[str, ...]
) -> tuple[dict, dict]:
    """从 FeatureCollection / Feature / 裸几何中取出唯一几何。"""
    doc_type = doc.get("type")

    if doc_type == "FeatureCollection":
        features = doc.get("features")
        if not isinstance(features, list) or not features:
            raise GeoJSONError("FeatureCollection 不包含任何 Feature")
        geom_features = [f for f in features if isinstance(f, dict) and f.get("geometry")]
        if len(geom_features) != 1:
            raise GeoJSONError(
                f"该文件期望包含恰好 1 个几何，实际有 {len(geom_features)} 个；"
                "场界文件请只放一个 Polygon"
            )
        feature = geom_features[0]
        geom = feature.get("geometry")
        props = feature.get("properties") or {}
        _ensure_geometry_type(geom, expected)
        return geom, props

    if doc_type == "Feature":
        geom = doc.get("geometry")
        if geom is None:
            raise GeoJSONError("Feature 缺少 geometry")
        _ensure_geometry_type(geom, expected)
        return geom, dict(doc.get("properties") or {})

    if doc_type in expected:
        return doc, {}

    raise GeoJSONError(
        f"不支持的 GeoJSON 类型 {doc_type!r}，期望 FeatureCollection/Feature/"
        f"{' 或 '.join(expected)}"
    )


def _ensure_geometry_type(geom: Any, expected: tuple[str, ...]) -> None:
    if not isinstance(geom, dict):
        raise GeoJSONError("geometry 必须是对象")
    if geom.get("type") not in expected:
        raise GeoJSONError(
            f"几何类型 {geom.get('type')!r} 不符合要求，期望 {'/'.join(expected)}"
        )


# --------------------------------------------------------------------------
# 机位布局解析
# --------------------------------------------------------------------------


@dataclass
class TurbineRecord:
    """单台机位的编号、源坐标与原始属性。"""

    turbine_id: str
    source_coords: np.ndarray  # 源 CRS 下坐标 (2,)
    attributes: dict = field(default_factory=dict)
    geographic: Optional[np.ndarray] = None  # (lon, lat)
    local: Optional[np.ndarray] = None  # 本地米制坐标 (2,)

    def to_dict(self) -> dict:
        return {
            "turbine_id": self.turbine_id,
            "source_coords": np.asarray(self.source_coords, dtype=float).tolist(),
            "attributes": self.attributes,
        }


def _point_features(doc: dict) -> list[dict]:
    doc_type = doc.get("type")
    if doc_type == "FeatureCollection":
        features = doc.get("features")
        if not isinstance(features, list) or not features:
            raise GeoJSONError("机位布局 FeatureCollection 不包含任何 Feature")
        return [f for f in features if isinstance(f, dict) and f.get("geometry") is not None]
    if doc_type == "Feature":
        return [doc] if doc.get("geometry") is not None else []
    if doc_type == "GeometryCollection":
        raise GeoJSONError(
            "机位布局不接受 GeometryCollection，请使用包含 Point Feature 的 "
            "FeatureCollection 以便携带机位编号与属性"
        )
    if doc_type == "Point":
        # 裸 Point 无法携带编号，明确拒绝
        raise GeoJSONError("裸 Point 几何没有机位编号，请使用带 properties 的 Feature")
    raise GeoJSONError(f"不支持的机位布局 GeoJSON 类型: {doc_type!r}")


def _extract_turbine_id(feature: dict, index: int) -> tuple[str, Any]:
    """从 Feature.id 或 properties 中提取机位编号。"""
    props = feature.get("properties") or {}
    for key in _ID_KEYS:
        if key in props and props[key] is not None and str(props[key]).strip() != "":
            return str(props[key]).strip(), props[key]
    fid = feature.get("id")
    if fid is not None and str(fid).strip() != "":
        return str(fid).strip(), fid
    raise GeoJSONError(
        f"第 {index + 1} 个机位 Feature 缺少编号（id），请在 properties.id / "
        "turbine_id / name 或 Feature.id 中提供"
    )


def parse_layout(doc: dict) -> list[TurbineRecord]:
    """解析机位布局 GeoJSON 为机位记录列表（源 CRS 坐标）。"""
    features = _point_features(doc)
    records: list[TurbineRecord] = []
    seen_ids: set[str] = set()
    dims_seen: set[int] = set()

    for idx, feature in enumerate(features):
        geom = feature["geometry"]
        _ensure_geometry_type(geom, ("Point",))
        coords = geom.get("coordinates")
        if not isinstance(coords, (list, tuple)):
            raise GeoJSONError(f"第 {idx + 1} 个机位坐标必须是数组")
        dims_seen.add(len(coords))
        if len(dims_seen) > 1:
            raise GeoJSONError(
                f"机位坐标维度不一致（2D/3D 混用），已出现维度: {sorted(dims_seen)}"
            )
        point = _validate_position(coords, context=f"机位 #{idx + 1}")

        turbine_id, _ = _extract_turbine_id(feature, idx)
        if turbine_id in seen_ids:
            raise GeoJSONError(f"机位编号重复: {turbine_id!r}（第 {idx + 1} 个 Feature）")
        seen_ids.add(turbine_id)

        props = dict(feature.get("properties") or {})
        records.append(
            TurbineRecord(
                turbine_id=turbine_id,
                source_coords=np.asarray(point, dtype=np.float64),
                attributes=props,
            )
        )

    if not records:
        raise GeoJSONError("机位布局中没有任何有效 Point 要素")
    return records


# --------------------------------------------------------------------------
# 导入汇总与约束校验
# --------------------------------------------------------------------------


@dataclass
class LayoutValidationReport:
    """导入布局的边界与间距检查结果。"""

    passed: bool
    n_turbines: int
    min_spacing_m: float
    boundary_tolerance_m: float
    spacing_violations: list[dict] = field(default_factory=list)
    boundary_violations: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_turbines": self.n_turbines,
            "min_spacing_m": self.min_spacing_m,
            "boundary_tolerance_m": self.boundary_tolerance_m,
            "n_spacing_violations": len(self.spacing_violations),
            "n_boundary_violations": len(self.boundary_violations),
            "spacing_violations": self.spacing_violations[:50],
            "boundary_violations": self.boundary_violations[:50],
        }


@dataclass
class ImportedSite:
    """一次 GeoJSON 导入的完整结果。"""

    chain: ProjectionChain
    boundary: SiteBoundary                 # 本地米制场界
    boundary_source: np.ndarray            # 源 CRS 场界顶点
    boundary_properties: dict
    turbines: list[TurbineRecord]
    boundary_path: Optional[str]
    layout_path: Optional[str]
    source_crs_declared_from: str
    validation: Optional[LayoutValidationReport]
    roundtrip_summary: dict

    @property
    def turbine_ids(self) -> list[str]:
        return [t.turbine_id for t in self.turbines]

    @property
    def local_positions(self) -> Optional[np.ndarray]:
        if not self.turbines or self.turbines[0].local is None:
            return None
        return np.asarray([t.local for t in self.turbines], dtype=np.float64)

    def provenance_dict(self) -> dict:
        """来源与坐标转换摘要，写入结果 JSON。"""
        data = {
            "boundary_file": os.path.abspath(self.boundary_path)
            if self.boundary_path
            else None,
            "layout_file": os.path.abspath(self.layout_path)
            if self.layout_path
            else None,
            "source_crs_declared_from": self.source_crs_declared_from,
            "projection": self.chain.as_dict(),
            "zone_check": self.chain.zone_check.as_dict()
            if self.chain.zone_check is not None
            else None,
            "roundtrip_verification": self.roundtrip_summary,
        }
        if self.validation is not None:
            data["layout_import_validation"] = self.validation.as_dict()
        if self.turbines:
            data["turbine_ids"] = self.turbine_ids
        return data


def validate_layout_local(
    records: list[TurbineRecord],
    boundary: SiteBoundary,
    min_spacing_m: float,
    boundary_tolerance_m: float = 1.0,
    spacing_tolerance_m: float = 1e-3,
) -> LayoutValidationReport:
    """在本地米制坐标下检查机位是否全部在场界内且满足最小间距。

    ``spacing_tolerance_m`` 用于吸收优化修复/坐标往返带来的浮点误差，
    仅当间距比要求值小超过该容差时才记为违规。
    """
    positions = np.asarray([r.local for r in records], dtype=np.float64)

    spacing_violations = []
    n = positions.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist < float(min_spacing_m) - spacing_tolerance_m:
                spacing_violations.append(
                    {
                        "turbine_a": records[int(i)].turbine_id,
                        "turbine_b": records[int(j)].turbine_id,
                        "distance_m": dist,
                        "required_m": float(min_spacing_m),
                        "shortfall_m": float(min_spacing_m - dist),
                    }
                )

    boundary_violations = []
    for rec, pos in zip(records, positions):
        if not boundary.contains_point(pos, tolerance=boundary_tolerance_m):
            boundary_violations.append(
                {
                    "turbine_id": rec.turbine_id,
                    "local_x_m": float(pos[0]),
                    "local_y_m": float(pos[1]),
                }
            )

    report = LayoutValidationReport(
        passed=not spacing_violations and not boundary_violations,
        n_turbines=len(records),
        min_spacing_m=float(min_spacing_m),
        boundary_tolerance_m=float(boundary_tolerance_m),
        spacing_violations=spacing_violations,
        boundary_violations=boundary_violations,
    )
    if not report.passed:
        parts = []
        if boundary_violations:
            ids = [v["turbine_id"] for v in boundary_violations[:10]]
            parts.append(f"{len(boundary_violations)} 台机位位于场界之外（{ids}）")
        if spacing_violations:
            sample = [
                f"{v['turbine_a']}-{v['turbine_b']}({v['distance_m']:.1f}m)"
                for v in spacing_violations[:10]
            ]
            parts.append(
                f"{len(spacing_violations)} 对机位间距不足（要求 "
                f"{min_spacing_m:.1f} m）：{sample}"
            )
        raise GeoJSONError("导入布局未通过约束检查：" + "；".join(parts))
    return report


def import_site(
    boundary_path: str,
    *,
    layout_path: Optional[str] = None,
    source_crs: Optional[str] = None,
    target_crs: Optional[str] = None,
    min_spacing_m: Optional[float] = None,
    boundary_tolerance_m: float = 1.0,
    roundtrip_tolerance_m: float = 0.05,
    use_local_frame: bool = True,
    origin_mode: str = "bbox_center",
    allow_cross_zone: bool = False,
) -> ImportedSite:
    """导入场界（及可选机位布局）GeoJSON 并完成全部校验与投影。

    Parameters
    ----------
    boundary_path : str
        场界 GeoJSON 路径。
    layout_path : Optional[str]
        机位布局 GeoJSON 路径；缺省时只导入场界。
    source_crs : Optional[str]
        显式源坐标系；文件内嵌 CRS 也可，二者必须一致。
    target_crs : Optional[str]
        显式米制目标坐标系；缺省时按场址自动选择 UTM 带。
    min_spacing_m : Optional[float]
        导入布局的最小允许间距（米）；布局存在时必传。
    boundary_tolerance_m : float
        机位落在场界内的判定容差（米）。
    roundtrip_tolerance_m : float
        坐标往返误差约定上限（米）。
    use_local_frame / origin_mode / allow_cross_zone
        透传给 :func:`build_projection_chain`。
    """
    boundary_doc = read_geojson(boundary_path)
    src_crs, declared_from = resolve_source_crs(boundary_doc, source_crs)
    boundary_source, boundary_props = parse_boundary(boundary_doc)

    layout_doc = None
    if layout_path is not None:
        layout_doc = read_geojson(layout_path)
        layout_crs, layout_declared = resolve_source_crs(layout_doc, source_crs)
        if not layout_crs.equals(src_crs):
            raise CoordinateSystemError(
                f"场界源坐标系与布局源坐标系不一致：{crs_code(src_crs) or src_crs.name}"
                f" vs {crs_code(layout_crs) or layout_crs.name}"
            )
        # 显式参数优先；布局文件内嵌声明同样记录来源
        if source_crs is None and declared_from.startswith("geojson"):
            declared_from = f"geojson:crs (boundary); {layout_declared} (layout)"

    if src_crs.is_geographic:
        _check_geographic_ranges(boundary_source, context="场界坐标")

    # 场界统一转到地理坐标 (lon, lat)，用于选带与跨带检查
    boundary_geographic = _source_to_geographic(src_crs, boundary_source)
    chain = build_projection_chain(
        src_crs,
        boundary_geographic,
        target_crs=target_crs,
        use_local_frame=use_local_frame,
        origin_mode=origin_mode,
        allow_cross_zone=allow_cross_zone,
    )

    # 场界 -> 本地米制
    boundary_local = chain.to_local(boundary_source)
    boundary = SiteBoundary(boundary_local)

    # 机位记录 -> 地理坐标 + 本地米制
    records: list[TurbineRecord] = []
    if layout_doc is not None:
        records = parse_layout(layout_doc)
        if src_crs.is_geographic:
            layout_src = np.asarray([r.source_coords for r in records])
            _check_geographic_ranges(layout_src, context="机位坐标")
        for rec in records:
            rec.geographic = chain.source_to_geographic(rec.source_coords.reshape(1, 2))[0]
            rec.local = chain.to_local(rec.source_coords.reshape(1, 2))[0]

    # 往返校验：场界顶点 + 所有机位
    check_points = boundary_source
    if records:
        check_points = np.vstack(
            [check_points, np.asarray([r.source_coords for r in records])]
        )
    roundtrip_summary = chain.verify_roundtrip(check_points, roundtrip_tolerance_m)

    validation = None
    if records:
        if min_spacing_m is None:
            raise ValueError("导入机位布局时必须提供 min_spacing_m")
        validation = validate_layout_local(
            records, boundary, float(min_spacing_m), boundary_tolerance_m
        )

    return ImportedSite(
        chain=chain,
        boundary=boundary,
        boundary_source=boundary_source,
        boundary_properties=boundary_props,
        turbines=records,
        boundary_path=boundary_path,
        layout_path=layout_path,
        source_crs_declared_from=declared_from,
        validation=validation,
        roundtrip_summary=roundtrip_summary,
    )


def _source_to_geographic(src_crs, coords: np.ndarray) -> np.ndarray:
    """投影源 CRS 坐标转地理坐标的辅助函数。"""
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    out = np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))
    return out


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def _crs_member(chain: ProjectionChain) -> Optional[dict]:
    """构造 GeoJSON 2008 风格的 crs 成员。"""
    code = crs_code(chain.source_crs)
    if code and code.upper().startswith("EPSG:"):
        epsg_num = code.split(":", 1)[1]
        return {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg_num}"},
        }
    # 没有 EPSG 代码时尽量写入名称，转换摘要 JSON 中保留完整定义
    return {
        "type": "name",
        "properties": {"name": chain.source_crs.name},
    }


def _rounded_source_point(chain: ProjectionChain, local_xy: np.ndarray) -> list[float]:
    """本地米制坐标反变换为源坐标并按坐标系类型取整。"""
    source = chain.to_source(np.asarray(local_xy, dtype=np.float64).reshape(1, 2))[0]
    decimals = _GEO_DECIMALS if chain.source_is_geographic else _PROJECTED_DECIMALS
    return [round(float(source[0]), decimals), round(float(source[1]), decimals)]


def export_boundary_geojson(
    chain: ProjectionChain,
    boundary_local: np.ndarray,
    path: str,
    *,
    properties: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> str:
    """把本地米制场界反变换导出为源 CRS 的 Polygon GeoJSON。

    Returns
    -------
    str
        写入文件的绝对路径。
    """
    vertices = np.asarray(boundary_local, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise GeoJSONError("场界顶点必须是 (N, 2) 数组")

    ring = [
        _rounded_source_point(chain, vertices[i]) for i in range(len(vertices))
    ]
    ring.append(ring[0])  # 闭合

    props = dict(properties or {})
    props.setdefault("boundary_type", "site_boundary")

    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": props,
    }
    doc: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": _crs_member(chain),
        "features": [feature],
    }
    if metadata:
        doc["x_wind_farm_opt_coordinate_summary"] = metadata

    _write_geojson(doc, path)
    return os.path.abspath(path)


def export_layout_geojson(
    chain: ProjectionChain,
    positions_local: np.ndarray,
    turbine_ids: list[str],
    path: str,
    *,
    source_attributes: Optional[list[dict]] = None,
    computed_attributes: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """把优化后的本地米制机位反变换导出为源 CRS 的 Point GeoJSON。

    机位编号顺序与 ``positions_local`` 行顺序一一对应；导入时携带的原始
    属性通过 ``source_attributes`` 原样带回，新增计算结果通过
    ``computed_attributes`` 合并（键名加 ``wfo_`` 前缀，避免覆盖来源数据）。
    """
    positions_local = np.asarray(positions_local, dtype=np.float64)
    if positions_local.ndim != 2 or positions_local.shape[1] != 2:
        raise GeoJSONError("机位坐标必须是 (N, 2) 数组")
    if len(turbine_ids) != positions_local.shape[0]:
        raise GeoJSONError("机位编号数量与坐标行数不一致")

    features = []
    for i, (turbine_id, pos) in enumerate(zip(turbine_ids, positions_local)):
        props: dict[str, Any] = {}
        if source_attributes:
            props.update(source_attributes[i])
        # 统一编号字段，保证再次导入时编号可识别
        props[_EXPORT_ID_KEY] = turbine_id
        if computed_attributes and computed_attributes[i]:
            for key, value in computed_attributes[i].items():
                props[f"wfo_{key}"] = value

        features.append(
            {
                "type": "Feature",
                "id": turbine_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": _rounded_source_point(chain, pos),
                },
                "properties": props,
            }
        )

    doc: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": _crs_member(chain),
        "features": features,
    }
    if metadata:
        doc["x_wind_farm_opt_coordinate_summary"] = metadata

    _write_geojson(doc, path)
    return os.path.abspath(path)


def _write_geojson(doc: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
