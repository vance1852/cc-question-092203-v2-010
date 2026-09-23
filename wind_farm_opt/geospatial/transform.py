"""源坐标系 ↔ 场址米制计算坐标系的正逆变换。

- ``forward``：源坐标（经纬度或源投影坐标）→ 本地米制坐标；
- ``inverse``：本地米制坐标 → 源坐标（导出回制图系统）；
- 导入时执行往返残差校验并记录转换摘要；
- 优化、约束、绘图全部在本地米制坐标系中进行，原配置的本地数组
  （如矩形边界 width/height/center）不经本模块处理。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection

from .crs import (
    GeoExtent,
    SourceCRSInfo,
    MetricCRSRecommendation,
    build_extent,
    check_cross_zone,
    crs_to_text,
)


class RoundTripError(ValueError):
    """正反变换往返残差超过约定容差。"""


@dataclass
class TransformSummary:
    """坐标转换摘要，随结果一并保存。

    Attributes
    ----------
    source_crs : str
        源坐标系编码。
    source_crs_kind : str
        ``geographic`` 或 ``projected``。
    source_crs_declared_in : str
        源坐标系来源（crs_member / cli_argument / rfc7946_default）。
    source_crs_assumed : bool
        源坐标系是否为 RFC 7946 默认假设。
    metric_crs : str
        米制计算坐标系编码。
    metric_crs_family : str
        投影族（cgcs2000_gk3 / utm / custom）。
    zone_label : str
        投影带说明。
    source_bounds_lonlat : dict
        场址在 WGS84/CGCS2000 地理坐标下的经纬度范围。
    forward_chain / inverse_chain : str
        pyproj 报告的实际转换链路。
    roundtrip_max_error_m : float
        导入校验点的最大往返残差（米）。
    roundtrip_tolerance_m : float
        约定容差（米）。
    n_check_points : int
        校验点数（场界顶点 + 全部机位）。
    """

    source_crs: str
    source_crs_kind: str
    source_crs_declared_in: str
    source_crs_assumed: bool
    metric_crs: str
    metric_crs_family: str
    zone_label: str
    source_bounds_lonlat: dict
    forward_chain: str
    inverse_chain: str
    roundtrip_max_error_m: float
    roundtrip_tolerance_m: float
    n_check_points: int
    datum_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoordinateTransformer:
    """封装源 CRS 与米制 CRS 之间的双向转换。

    Parameters
    ----------
    source
        源坐标系信息。
    metric
        选定的米制计算坐标系。
    roundtrip_tolerance_m
        允许的最大往返残差（米），默认 1 mm，兼顾浮点误差与精度留量。
    """

    def __init__(
        self,
        source: SourceCRSInfo,
        metric: MetricCRSRecommendation | CRS,
        roundtrip_tolerance_m: float = 1e-3,
    ) -> None:
        self.source = source
        self.source_crs = source.crs
        if isinstance(metric, MetricCRSRecommendation):
            self.metric_crs = metric.crs
            self.metric_code = metric.code
            self.family = metric.family
            self.zone_label = metric.zone_label
        else:
            self.metric_crs = metric
            self.metric_code = crs_to_text(metric)
            self.family = "custom"
            self.zone_label = self.metric_code

        self.roundtrip_tolerance_m = float(roundtrip_tolerance_m)

        # always_xy=True 统一按“经度,x 在前”处理，避免 EPSG:4326
        # 传统轴序（lat, lon）带来的手工平移错误。
        self._to_metric = Transformer.from_crs(
            self.source_crs, self.metric_crs, always_xy=True
        )
        self._to_source = Transformer.from_crs(
            self.metric_crs, self.source_crs, always_xy=True
        )

        self._extent: Optional[GeoExtent] = None
        self._summary: Optional[TransformSummary] = None

    # ---- 基础变换 ---------------------------------------------------------

    def forward(self, xy: np.ndarray) -> np.ndarray:
        """源坐标 → 本地米制坐标，输入输出均为 (N, 2)。"""
        arr = np.asarray(xy, dtype=np.float64)
        x, y = self._to_metric.transform(arr[:, 0], arr[:, 1])
        out = np.column_stack([np.asarray(x), np.asarray(y)])
        if not np.all(np.isfinite(out)):
            raise ValueError("正变换产生非有限坐标，请检查源坐标是否越界")
        return out

    def inverse(self, metric_xy: np.ndarray) -> np.ndarray:
        """本地米制坐标 → 源坐标，输入输出均为 (N, 2)。"""
        arr = np.asarray(metric_xy, dtype=np.float64)
        x, y = self._to_source.transform(arr[:, 0], arr[:, 1])
        out = np.column_stack([np.asarray(x), np.asarray(y)])
        if not np.all(np.isfinite(out)):
            raise ValueError("逆变换产生非有限坐标")
        return out

    def forward_point(self, x: float, y: float) -> tuple[float, float]:
        return self.forward(np.array([[x, y]]))[0].tolist()

    def inverse_point(self, x: float, y: float) -> tuple[float, float]:
        return self.inverse(np.array([[x, y]]))[0].tolist()

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        """任意源坐标转到 WGS84/CGCS2000 地理坐标（用于范围与跨带检查）。"""
        arr = np.asarray(xy, dtype=np.float64)
        if self.source_crs.is_geographic:
            return arr.copy()
        t = Transformer.from_crs(self.source_crs, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(arr[:, 0], arr[:, 1])
        return np.column_stack([np.asarray(lon), np.asarray(lat)])

    # ---- 导入校验 ---------------------------------------------------------

    def verify_import(
        self,
        source_points: np.ndarray,
        cross_zone_points: Optional[np.ndarray] = None,
        cross_zone_check: bool = True,
    ) -> TransformSummary:
        """对全部导入点执行跨带检查与往返残差校验，生成转换摘要。

        Parameters
        ----------
        source_points
            源坐标点（场界顶点与机位坐标合并），形状 (M, 2)，用于往返残差校验。
        cross_zone_points
            用于跨带判定与选带范围的点（通常只取场界顶点——场址由边界定义，
            个别界外/错误机位不应改变场址所属投影带）。默认与 source_points 相同。
        cross_zone_check
            是否执行跨带检查（源为投影坐标时按其地理范围检查）。
        """
        pts = np.asarray(source_points, dtype=np.float64)
        zone_pts = (
            np.asarray(cross_zone_points, dtype=np.float64)
            if cross_zone_points is not None else pts
        )
        lonlat = self.to_lonlat(pts)
        if not np.all(np.isfinite(lonlat)):
            raise ValueError("源坐标无法转换为地理坐标（存在非有限结果）")
        zone_lonlat = self.to_lonlat(zone_pts)
        self._extent = build_extent(zone_lonlat)

        if cross_zone_check:
            family = (
                self.family
                if self.family in {"cgcs2000", "cgcs2000_gk3", "utm"}
                else None
            )
            if family is not None:
                info = check_cross_zone(self._extent, family, zone_lonlat)
                if info is not None:
                    from .crs import CrossZoneError
                    raise CrossZoneError(
                        f"场址跨越{info['zone_label']}边界: 顶点分布于 {info['zones']}，"
                        f"经度范围 {info['lon_min']:.4f}..{info['lon_max']:.4f}"
                        f"（跨度 {info['lon_span_deg']:.3f}°，单带宽 {info['max_span_deg']:.0f}°）。"
                        "单投影带无法保证全场距离精度，请缩小场址或分带处理。"
                    )

        metric_pts = self.forward(pts)
        back = self.inverse(metric_pts)
        back_metric = self.forward(back)
        residual = np.linalg.norm(metric_pts - back_metric, axis=1)
        max_err = float(np.max(residual)) if len(residual) else 0.0

        if not np.isfinite(max_err) or max_err > self.roundtrip_tolerance_m:
            raise RoundTripError(
                f"正反变换往返残差 {max_err:.6g} m 超过容差 "
                f"{self.roundtrip_tolerance_m:.6g} m"
            )

        fwd_desc = self._describe_pipeline(TransformDirection.FORWARD)
        inv_desc = self._describe_pipeline(TransformDirection.INVERSE)

        datum_note = ""
        if self.source.is_geographic and "Ballpark" in (fwd_desc + inv_desc):
            datum_note = (
                "源地理坐标系与 CGCS2000 之间无官方基准转换参数，PROJ 使用"
                " Ballpark（近似）变换；WGS84 与 CGCS2000 在风电场尺度下差异通常"
                " 亚米级，若制图系统要求严格基准，请显式提供 CGCS2000(EPSG:4490)"
                " 源数据或带官方参数的自定义 CRS。"
            )

        lon_min = self._extent.lon_min
        lon_max = self._extent.lon_max
        lat_min = self._extent.lat_min
        lat_max = self._extent.lat_max

        self._summary = TransformSummary(
            source_crs=self.source.code,
            source_crs_kind="geographic" if self.source.is_geographic else "projected",
            source_crs_declared_in=self.source.declared_in,
            source_crs_assumed=self.source.assumed,
            metric_crs=self.metric_code,
            metric_crs_family=self.family,
            zone_label=self.zone_label,
            source_bounds_lonlat={
                "lon_min": lon_min, "lon_max": lon_max,
                "lat_min": lat_min, "lat_max": lat_max,
            },
            forward_chain=fwd_desc,
            inverse_chain=inv_desc,
            roundtrip_max_error_m=max_err,
            roundtrip_tolerance_m=self.roundtrip_tolerance_m,
            n_check_points=int(len(pts)),
            datum_note=datum_note,
        )
        return self._summary

    def _describe_pipeline(self, direction: TransformDirection) -> str:
        """返回 pyproj 实际使用的转换链路描述。"""
        try:
            desc = self._to_metric.description if direction == TransformDirection.FORWARD \
                else self._to_source.description
            return str(desc).strip()
        except Exception:
            return "unknown"

    @property
    def summary(self) -> Optional[TransformSummary]:
        return self._summary

    @property
    def extent(self) -> Optional[GeoExtent]:
        return self._extent
