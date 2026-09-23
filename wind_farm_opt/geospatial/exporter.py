"""优化后布局的 GeoJSON 导出。

- 使用导入时建立的 :class:`CoordinateTransformer` 做逆变换，保证
  机位编号、地理位置与源系统一致；
- 多边形、机位编号和属性完整回写；
- 导出前再次执行正变换往返复核，残差超差即拒绝写出；
- 输出 FeatureCollection（GeoJSON 2008 风格 crs 成员），默认拒绝 NaN/Infinity；
- 同时返回转换摘要字典，供结果 JSON 保存来源、目标坐标系与转换信息。
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np

from .importer import ImportedSite
from .transform import CoordinateTransformer


class ExportError(ValueError):
    """导出失败（往返残差超差、坐标非有限或编号不匹配）。"""


def _crs_member(transformer: CoordinateTransformer) -> dict:
    """构造 GeoJSON 2008 风格 crs 成员。"""
    return {"type": "name", "properties": {"name": transformer.source.code}}


def _clean_value(value: Any) -> Any:
    """把 numpy 标量/数组转为可 JSON 序列化的 Python 值，并拒绝非有限数。"""
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_value(value.item())
    if isinstance(value, np.ndarray):
        return [_clean_value(v) for v in value.tolist()]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExportError(f"属性包含非有限数值: {value}")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    # 其他类型转字符串，避免 json.dump 失败
    return str(value)


def _position(source_xy: np.ndarray, dimension: int) -> list:
    x, y = float(source_xy[0]), float(source_xy[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ExportError("逆变换产生非有限坐标")
    return [x, y] if dimension == 2 else [x, y, 0.0]


def build_export_geojson(
    site: ImportedSite,
    metric_positions: np.ndarray,
    *,
    turbine_ids: Optional[list[str]] = None,
    turbine_attributes: Optional[dict[str, dict]] = None,
    layout_stage: str = "optimized",
    extra_boundary_properties: Optional[dict] = None,
    include_transform_summary: bool = True,
) -> dict:
    """构造优化后布局的 GeoJSON 对象（源坐标系）。

    Parameters
    ----------
    site
        导入结果，携带变压器、原编号、属性与场界。
    metric_positions
        优化后的机位本地米制坐标 (N, 2)，顺序须与编号一致。
    turbine_ids
        机位编号序列；默认使用导入时的编号。若提供，必须与导入编号集合一致
        （允许重排），防止编号-地理对应关系错乱。
    turbine_attributes
        编号 -> 附加属性（如 wake_loss_pct、net_aep_mwh），合并进各机位属性。
    layout_stage
        布局阶段标记（optimized / baseline / imported）。
    extra_boundary_properties
        合并进场界要素的附加属性。
    include_transform_summary
        是否在 FeatureCollection 的自定义成员中写入转换摘要。
    """
    transformer = site.transformer
    positions = np.asarray(metric_positions, dtype=np.float64)

    orig_ids = site.turbine_ids
    if turbine_ids is None:
        ids = list(orig_ids)
    else:
        ids = [str(x) for x in turbine_ids]
        if sorted(ids) != sorted(orig_ids):
            raise ExportError(
                "导出编号集合与导入编号不一致："
                f"新增/缺失 {sorted(set(ids) ^ set(orig_ids))}"
            )

    if positions.shape != (len(ids), 2):
        raise ExportError(
            f"机位坐标形状 {positions.shape} 与编号数 {len(ids)} 不匹配"
        )
    if not np.all(np.isfinite(positions)):
        raise ExportError("机位米制坐标包含非有限值")

    # 建立编号 -> 米制坐标 映射，按导入顺序输出，保证属性对齐。
    pos_by_id = {tid: positions[i] for i, tid in enumerate(ids)}
    ordered_metric = np.vstack([pos_by_id[tid] for tid in orig_ids])

    # 往返复核：米制 -> 源 -> 米制，残差必须在约定容差内。
    source_xy = transformer.inverse(ordered_metric)
    back_metric = transformer.forward(source_xy)
    residual = np.linalg.norm(ordered_metric - back_metric, axis=1)
    max_err = float(np.max(residual)) if len(residual) else 0.0
    if not math.isfinite(max_err) or max_err > transformer.roundtrip_tolerance_m:
        raise ExportError(
            f"导出往返残差 {max_err:.6g} m 超过容差 "
            f"{transformer.roundtrip_tolerance_m:.6g} m"
        )

    attrs = turbine_attributes or {}

    # ---- 场界要素（逆变换外环与内环） ----
    boundary_props = dict(site.boundary_properties)
    boundary_props.setdefault("sfp_role", "site_boundary")
    if extra_boundary_properties:
        boundary_props.update(_clean_value(dict(extra_boundary_properties)))

    rings_metric = [site.boundary.vertices, *site.boundary.holes]
    rings_source = [transformer.inverse(r) for r in rings_metric]
    polygon_coords = []
    for ring in rings_source:
        closed = np.vstack([ring, ring[0:1]])
        polygon_coords.append(
            [_position(p, site.source_dimension) for p in closed]
        )

    boundary_feature = {
        "type": "Feature",
        "properties": _clean_value(boundary_props),
        "geometry": {"type": "Polygon", "coordinates": polygon_coords},
    }

    # ---- 机位要素 ----
    turbine_features = []
    for i, tid in enumerate(orig_ids):
        props = dict(site.turbine_properties[i])
        props.setdefault("turbine_id", tid)
        props["sfp_role"] = "turbine"
        props["layout_stage"] = layout_stage
        if tid in attrs:
            props.update(_clean_value(dict(attrs[tid])))

        feature = {
            "type": "Feature",
            "id": _json_id(tid),
            "properties": _clean_value(props),
            "geometry": {
                "type": "Point",
                "coordinates": _position(source_xy[i], site.source_dimension),
            },
        }
        turbine_features.append(feature)

    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": _crs_member(transformer),
        "features": [boundary_feature, *turbine_features],
    }

    if include_transform_summary and transformer.summary is not None:
        summary = transformer.summary.to_dict()
        summary["export_roundtrip_max_error_m"] = max_err
        fc["sfp_coordinate_transform"] = _clean_value(summary)

    return fc


def _json_id(tid: str) -> Any:
    """GeoJSON Feature.id 允许字符串或数字；纯数字编号输出为数字。"""
    try:
        return int(tid)
    except (TypeError, ValueError):
        return tid


def export_geojson(
    site: ImportedSite,
    metric_positions: np.ndarray,
    output_path: str,
    *,
    turbine_ids: Optional[list[str]] = None,
    turbine_attributes: Optional[dict[str, dict]] = None,
    layout_stage: str = "optimized",
    extra_boundary_properties: Optional[dict] = None,
    indent: int = 2,
) -> dict:
    """构造 GeoJSON 并写入文件，返回写出的对象。"""
    geojson = build_export_geojson(
        site,
        metric_positions,
        turbine_ids=turbine_ids,
        turbine_attributes=turbine_attributes,
        layout_stage=layout_stage,
        extra_boundary_properties=extra_boundary_properties,
    )
    # allow_nan=False 是最后一道防线：任何 NaN/Infinity 都会抛错而不是写出。
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=indent, allow_nan=False)
    return geojson
