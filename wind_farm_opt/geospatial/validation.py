"""GeoJSON 结构校验与场界/机位要素提取。

校验规则（违反即拒绝导入）：
- 缺失坐标参考系且无法按 RFC 7946 安全默认（在 importer 中结合坐标量级判定）；
- 坐标维度混合（2D 与 3D 混用）或维度不合法；
- 非有限坐标（NaN/Infinity，或非数值类型）；
- 机位编号缺失或重复；
- 不支持的几何类型（只接受 Point / Polygon / MultiPolygon 及其 Feature 容器）；
- 多个场界多边形。

本模块只负责结构与数值合法性，坐标参考系解析、跨带、边界、间距检查
分别在 ``crs`` / ``importer`` 中完成。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class GeoJSONValidationError(ValueError):
    """GeoJSON 结构或数值校验失败。"""


# 机位编号在 properties 中的候选键（feature.id 优先级最高）。
_ID_KEYS = ("turbine_id", "turbineId", "id", "name")
# 场界角色标记的候选键。
_ROLE_KEYS = ("sfp_role", "role", "layer")
_BOUNDARY_ROLE_VALUES = {"site_boundary", "boundary", "site", "场界", "边界"}
_TURBINE_ROLE_VALUES = {"turbine", "wind_turbine", "turbine_location", "机位", "风机"}

_POINT_TYPES = {"Point"}
_POLYGON_TYPES = {"Polygon", "MultiPolygon"}
_SUPPORTED_TYPES = _POINT_TYPES | _POLYGON_TYPES


@dataclass
class BoundaryGeometry:
    """场界几何（源坐标系）。

    Attributes
    ----------
    rings : list[np.ndarray]
        多边形环，每个为 (K, 2) 数组（已剥离 Z 值），第一个为外环，
        其余为内环（洞）。MultiPolygon 只允许包含单个多边形。
    properties : dict
        场界要素属性。
    """

    rings: list[np.ndarray]
    properties: dict = field(default_factory=dict)


@dataclass
class TurbineFeature:
    """单个机位要素。"""

    turbine_id: str
    source_xy: np.ndarray  # (2,)
    properties: dict = field(default_factory=dict)


@dataclass
class ParsedGeoJSON:
    """GeoJSON 解析结果。"""

    raw: dict
    crs_member: Any
    boundary: Optional[BoundaryGeometry]
    turbines: list[TurbineFeature]
    dimension: int
    all_source_points: np.ndarray  # 场界顶点与机位合并，(M, 2)
    roles_explicit: bool = False


# ---- JSON 读取（拒绝 NaN / Infinity） -------------------------------------

def _reject_constant(value: str) -> Any:
    raise GeoJSONValidationError(f"JSON 中不允许非有限常量: {value}")


def load_geojson_file(path: str) -> dict:
    """读取并解析 GeoJSON 文件，拒绝 NaN/Infinity 常量。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f, parse_constant=_reject_constant)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise GeoJSONValidationError(f"GeoJSON 解析失败: {e}") from e
    if not isinstance(data, dict):
        raise GeoJSONValidationError("GeoJSON 根节点必须是对象")
    return data


def load_geojson_text(text: str) -> dict:
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise GeoJSONValidationError(f"GeoJSON 解析失败: {e}") from e
    if not isinstance(data, dict):
        raise GeoJSONValidationError("GeoJSON 根节点必须是对象")
    return data


# ---- 坐标基础校验 ---------------------------------------------------------

def _check_position(pos: Any, path: str, dimension: int) -> tuple[float, float, int]:
    """校验单个坐标位置，返回 (x/easting, y/northing, 实际维度)。"""
    if not isinstance(pos, (list, tuple)):
        raise GeoJSONValidationError(f"{path}: 坐标必须是数组")
    dim = len(pos)
    if dim not in (2, 3):
        raise GeoJSONValidationError(
            f"{path}: 坐标维度必须为 2 或 3，实际为 {dim}"
        )
    for axis in range(dim):
        v = pos[axis]
        # bool 是 int 的子类，需要显式排除
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise GeoJSONValidationError(
                f"{path}: 坐标分量必须是数值，发现 {type(v).__name__}"
            )
        if not math.isfinite(float(v)):
            raise GeoJSONValidationError(
                f"{path}: 坐标必须有限，发现 {v}"
            )
    return float(pos[0]), float(pos[1]), dim


def _check_ring(coords: Any, path: str, dimension: int) -> tuple[np.ndarray, int]:
    if not isinstance(coords, list) or len(coords) < 4:
        raise GeoJSONValidationError(
            f"{path}: 多边形环至少需要 4 个位置（首尾闭合）"
        )
    pts = np.empty((len(coords), 2), dtype=np.float64)
    dim_used = dimension
    for i, pos in enumerate(coords):
        x, y, dim = _check_position(pos, f"{path}[{i}]", dimension)
        dim_used = dim
        pts[i] = (x, y)
    if not np.allclose(pts[0], pts[-1], atol=0.0, rtol=0.0):
        raise GeoJSONValidationError(f"{path}: 多边形环必须首尾闭合")
    # 去掉重复的闭合点（SiteBoundary 不要求重复起点）。
    return pts[:-1], dim_used


def _check_dimension_consistent(current: Optional[int], dim: int, path: str) -> int:
    if current is not None and dim != current:
        raise GeoJSONValidationError(
            f"{path}: 坐标维度混合，文件中同时存在 {current}D 与 {dim}D 坐标"
        )
    return dim


# ---- 几何提取 -------------------------------------------------------------

def _point_xy(coords: Any, path: str, dimension: int) -> tuple[np.ndarray, int]:
    x, y, dim = _check_position(coords, path, dimension)
    return np.array([x, y], dtype=np.float64), dim


def _polygon_rings(
    coords: Any, path: str, dimension: int
) -> tuple[list[np.ndarray], int]:
    if not isinstance(coords, list) or not coords:
        raise GeoJSONValidationError(f"{path}: Polygon 坐标不能为空")
    rings = []
    dim_used = dimension
    for r, ring in enumerate(coords):
        arr, dim = _check_ring(ring, f"{path}.ring[{r}]", dimension)
        dim_used = _check_dimension_consistent(dim_used, dim, path)
        rings.append(arr)
    return rings, dim_used


def _feature_role(props: dict) -> Optional[str]:
    for key in _ROLE_KEYS:
        if key in props:
            val = str(props[key]).strip().lower()
            if val in _BOUNDARY_ROLE_VALUES:
                return "boundary"
            if val in _TURBINE_ROLE_VALUES:
                return "turbine"
    return None


def _extract_turbine_id(feature: dict, index_hint: int) -> str:
    fid = feature.get("id")
    if isinstance(fid, (str, int)) and str(fid).strip():
        return str(fid).strip()

    props = feature.get("properties") or {}
    for key in _ID_KEYS:
        if key in props:
            val = props[key]
            if isinstance(val, (str, int)) and str(val).strip():
                return str(val).strip()

    raise GeoJSONValidationError(
        f"第 {index_hint} 个机位要素缺少编号：请设置 Feature.id 或 "
        f"properties.{_ID_KEYS[0]}"
    )


def _process_feature(
    feature: dict,
    index: int,
    state: dict,
) -> None:
    if not isinstance(feature, dict):
        raise GeoJSONValidationError(f"Feature[{index}] 必须是对象")
    geom = feature.get("geometry")
    if geom is None:
        raise GeoJSONValidationError(f"Feature[{index}] 缺少 geometry")
    if not isinstance(geom, dict):
        raise GeoJSONValidationError(f"Feature[{index}].geometry 必须是对象")

    props = feature.get("properties") or {}
    if not isinstance(props, dict):
        raise GeoJSONValidationError(f"Feature[{index}].properties 必须是对象")

    gtype = geom.get("type")
    role = _feature_role(props)
    if role is not None:
        state["roles_explicit"] = True
    path = f"Feature[{index}:{gtype}]"
    _consume_geometry(gtype, geom.get("coordinates"), path, state,
                      props=props, feature=feature, index=index, role=role)


def _consume_geometry(
    gtype: Any,
    coords: Any,
    path: str,
    state: dict,
    props: Optional[dict] = None,
    feature: Optional[dict] = None,
    index: int = 0,
    role: Optional[str] = None,
) -> None:
    props = props or {}

    if gtype not in _SUPPORTED_TYPES:
        raise GeoJSONValidationError(
            f"{path}: 不支持的几何类型 {gtype!r}，仅接受 "
            "Point / Polygon / MultiPolygon"
        )

    dim = state["dimension"]

    if gtype == "Point":
        xy, d = _point_xy(coords, path, dim)
        dim = _check_dimension_consistent(dim, d, path)
        if role == "boundary":
            raise GeoJSONValidationError(f"{path}: 场界必须是多边形几何")
        turbine_id = _extract_turbine_id(feature, index) if feature else f"P{index}"
        state["turbines"].append(TurbineFeature(
            turbine_id=turbine_id, source_xy=xy, properties=dict(props),
        ))
        state["points"].append(xy)

    elif gtype == "Polygon":
        if role == "turbine":
            raise GeoJSONValidationError(f"{path}: 机位必须是 Point 几何")
        rings, d = _polygon_rings(coords, path, dim)
        dim = _check_dimension_consistent(dim, d, path)
        _add_boundary(rings, props, state, path)

    elif gtype == "MultiPolygon":
        if role == "turbine":
            raise GeoJSONValidationError(f"{path}: 机位必须是 Point 几何")
        if not isinstance(coords, list) or not coords:
            raise GeoJSONValidationError(f"{path}: MultiPolygon 坐标不能为空")
        if len(coords) != 1:
            raise GeoJSONValidationError(
                f"{path}: 场界 MultiPolygon 只能包含 1 个多边形，"
                f"实际 {len(coords)} 个"
            )
        rings, d = _polygon_rings(coords[0], path, dim)
        dim = _check_dimension_consistent(dim, d, path)
        _add_boundary(rings, props, state, path)

    state["dimension"] = dim


def _add_boundary(
    rings: list[np.ndarray],
    props: dict,
    state: dict,
    path: str,
) -> None:
    if state["boundary"] is not None:
        raise GeoJSONValidationError(
            f"{path}: 检测到多个场界多边形，一个场址只允许一个场界"
        )
    outer = rings[0]
    if len(outer) < 3:
        raise GeoJSONValidationError(f"{path}: 场界外环至少需要 3 个不同顶点")
    if not np.all(np.isfinite(outer)):
        raise GeoJSONValidationError(f"{path}: 场界坐标必须全部有限")
    state["boundary"] = BoundaryGeometry(rings=rings, properties=dict(props))
    state["points"].extend(rings)


# ---- 顶层解析 -------------------------------------------------------------

def parse_geojson(data: dict) -> ParsedGeoJSON:
    """解析并校验 GeoJSON 对象，提取场界与机位。"""
    t = data.get("type")
    crs_member = data.get("crs")
    if crs_member is not None and not isinstance(crs_member, dict):
        raise GeoJSONValidationError("GeoJSON crs 成员必须是对象")

    state: dict = {
        "boundary": None,
        "turbines": [],
        "points": [],
        "dimension": None,
        "roles_explicit": False,
    }

    if t == "FeatureCollection":
        features = data.get("features")
        if not isinstance(features, list):
            raise GeoJSONValidationError("FeatureCollection 必须包含 features 数组")
        for i, feature in enumerate(features):
            _process_feature(feature, i, state)

    elif t == "Feature":
        _process_feature(data, 0, state)

    elif t in _SUPPORTED_TYPES:
        _consume_geometry(t, data.get("coordinates"), t, state, index=0)

    elif t is None:
        raise GeoJSONValidationError("GeoJSON 缺少 type 字段")
    else:
        raise GeoJSONValidationError(
            f"不支持的 GeoJSON 根类型 {t!r}，仅接受 FeatureCollection / "
            "Feature / Point / Polygon / MultiPolygon"
        )

    if state["dimension"] is None:
        raise GeoJSONValidationError("GeoJSON 中没有任何有效坐标")

    _check_duplicate_ids(state["turbines"])

    all_points = (
        np.vstack(state["points"]) if state["points"]
        else np.zeros((0, 2), dtype=np.float64)
    )

    return ParsedGeoJSON(
        raw=data,
        crs_member=crs_member,
        boundary=state["boundary"],
        turbines=state["turbines"],
        dimension=state["dimension"],
        all_source_points=all_points,
        roles_explicit=state["roles_explicit"],
    )


def _check_duplicate_ids(turbines: list[TurbineFeature]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for tb in turbines:
        if tb.turbine_id in seen and tb.turbine_id not in duplicates:
            duplicates.append(tb.turbine_id)
        seen.add(tb.turbine_id)
    if duplicates:
        raise GeoJSONValidationError(
            f"机位编号重复: {', '.join(duplicates)}；编号必须在布局内唯一"
        )
