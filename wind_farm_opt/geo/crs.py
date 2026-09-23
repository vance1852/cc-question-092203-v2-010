"""地理坐标参考系（CRS）与米制投影链。

勘测交付的 GeoJSON 通常是 WGS84/CGCS2000 经纬度坐标，而布局优化、
间距与尾流计算必须在米制平面坐标系中进行。本模块负责：

* 显式解析并要求数据源坐标参考系；
* 按场址地理位置自动选择合适的 UTM 米制带，或接受人工指定；
* 拒绝明显跨投影带的场址；
* 记录 ``源地理坐标 -> 米制投影 -> 本地平移显示坐标`` 的正、反变换；
* 验证正反变换的往返误差在约定容差以内。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pyproj import CRS, Geod, Transformer
from pyproj.exceptions import CRSError as _PyprojCRSError


class CoordinateSystemError(ValueError):
    """坐标参考系缺失、无法识别或场址跨带等错误。"""


# 常见地理坐标系别名（值为标准 EPSG 代码）
_GEOGRAPHIC_ALIASES = {
    "WGS84": "EPSG:4326",
    "WGS 84": "EPSG:4326",
    "WGS-84": "EPSG:4326",
    "CGCS2000": "EPSG:4490",
    "CGCS 2000": "EPSG:4490",
    "CGCS-2000": "EPSG:4490",
    "BEIJING1954": "EPSG:4214",
    "XIAN1980": "EPSG:4610",
}

# urn:ogc:def:crs:EPSG::4326 / urn:ogc:def:crs:epsg::4326
_URN_EPSG_RE = re.compile(r"^urn:ogc:def:crs:epsg::(\d+)$", re.IGNORECASE)
_PLAIN_EPSG_RE = re.compile(r"^epsg:(\d+)$", re.IGNORECASE)
_OGC_CRS84_RE = re.compile(r"^ogc:crs84$", re.IGNORECASE)


def _normalize_spec(spec: str) -> str:
    """把各种 CRS 写法归一化为 pyproj 可识别的字符串。"""
    text = str(spec).strip()
    upper = text.upper().replace(" ", "")
    if upper in _GEOGRAPHIC_ALIASES:
        return _GEOGRAPHIC_ALIASES[upper]
    m = _URN_EPSG_RE.match(text)
    if m:
        return f"EPSG:{m.group(1)}"
    m = _PLAIN_EPSG_RE.match(text)
    if m:
        return f"EPSG:{m.group(1)}"
    if _OGC_CRS84_RE.match(text.replace(" ", "")):
        # OGC:CRS84 即 WGS84 且明确经、纬轴序
        return "EPSG:4326"
    return text


def parse_crs(spec: str, *, role: str = "坐标参考系") -> CRS:
    """把用户给定或文件中声明的 CRS 字符串解析为 :class:`pyproj.CRS`。

    Parameters
    ----------
    spec : str
        EPSG 代码（``EPSG:4326``）、OGC URN（``urn:ogc:def:crs:EPSG::4326``）
        或常见别名（``WGS84``、``CGCS2000``）等。
    role : str
        错误信息中使用的角色描述。
    """
    if spec is None or str(spec).strip() == "":
        raise CoordinateSystemError(f"未声明{role}，拒绝按隐式坐标处理")
    normalized = _normalize_spec(spec)
    try:
        return CRS.from_user_input(normalized)
    except _PyprojCRSError as exc:
        raise CoordinateSystemError(f"无法识别{role}: {spec!r}（{exc}）") from exc


def crs_code(crs: CRS) -> Optional[str]:
    """返回 CRS 的 EPSG 代码字符串，无法确定时返回 None。"""
    code = crs.to_epsg()
    return f"EPSG:{code}" if code else None


def crs_label(crs: CRS) -> str:
    """用于报告的 CRS 标签：优先 EPSG 代码，否则用名称。"""
    return crs_code(crs) or crs.name


def extract_crs_from_geojson(doc: dict) -> tuple[Optional[CRS], Optional[str]]:
    """从 GeoJSON 文档的旧式 ``crs`` 成员读取坐标参考系。

    支持 GeoJSON 2008 的 named CRS：

    .. code-block:: json

        {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}}

    Returns
    -------
    tuple[Optional[CRS], Optional[str]]
        (CRS 对象, 原始声明字符串)；文档未携带 crs 成员时为 (None, None)。
    """
    crs_member = doc.get("crs")
    if crs_member is None:
        return None, None
    if not isinstance(crs_member, dict):
        raise CoordinateSystemError("GeoJSON 的 crs 成员必须是对象")

    crs_type = crs_member.get("type")
    if crs_type == "name":
        props = crs_member.get("properties") or {}
        name = props.get("name")
        if not name:
            raise CoordinateSystemError("GeoJSON named CRS 缺少 properties.name")
        return parse_crs(name, role="GeoJSON 声明的坐标参考系"), str(name)

    # linked CRS 需要外部 URI 引用，无法离线可靠解析，明确拒绝
    raise CoordinateSystemError(
        f"暂不支持类型为 {crs_type!r} 的 GeoJSON CRS，"
        "请改用 EPSG 代码命名的 CRS 或通过 --source-crs 显式指定"
    )


def resolve_source_crs(
    doc: dict,
    explicit: Optional[str] = None,
) -> tuple[CRS, str]:
    """确定数据源坐标参考系。

    优先使用显式参数；文件内嵌 CRS 同样有效。二者同时给出且不一致时
    拒绝，二者都缺失时拒绝——系统不接受“不知道坐标系”的输入。

    Returns
    -------
    tuple[CRS, str]
        (CRS 对象, 来源说明，如 ``"argument:EPSG:4326"`` 或 ``"geojson:crs"``)
    """
    file_crs, file_decl = extract_crs_from_geojson(doc)

    if explicit is not None and str(explicit).strip() != "":
        arg_crs = parse_crs(explicit, role="源坐标参考系")
        if file_crs is not None and not arg_crs.equals(file_crs):
            raise CoordinateSystemError(
                f"命令行/配置声明的源坐标系 {crs_label(arg_crs)} 与 GeoJSON 内嵌的 "
                f"{crs_label(file_crs)} 不一致，请删除其中之一"
            )
        return arg_crs, f"argument:{_normalize_spec(explicit)}"

    if file_crs is not None:
        return file_crs, f"geojson:crs:{file_decl}"

    raise CoordinateSystemError(
        "输入数据缺少坐标参考系：GeoJSON 未携带 crs 成员，且未通过 "
        "--source-crs / source_crs 显式指定，拒绝按未知坐标处理"
    )


def utm_zone_for_longitude(longitude: float) -> int:
    """根据经度计算 UTM 带号（1..60）。"""
    zone = int(math.floor((float(longitude) + 180.0) / 6.0)) + 1
    if zone < 1:
        zone = 1
    elif zone > 60:
        zone = 60
    return zone


def suggest_utm_crs(lons: np.ndarray, lats: np.ndarray) -> tuple[CRS, int, bool]:
    """根据场址经纬度选择 UTM 米制坐标系。

    Returns
    -------
    tuple[CRS, int, bool]
        (UTM CRS, 带号, 是否北半球)
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    center_lon = float((np.nanmin(lons) + np.nanmax(lons)) / 2.0)
    center_lat = float(np.nanmean(lats))
    zone = utm_zone_for_longitude(center_lon)
    northern = center_lat >= 0.0
    epsg = 32600 + zone if northern else 32700 + zone
    return CRS.from_epsg(epsg), zone, northern


def _epsg_utm_zone(crs: CRS) -> Optional[tuple[int, bool]]:
    """若 CRS 是 UTM 带，返回 (带号, 是否北半球)，否则 None。"""
    code = crs.to_epsg()
    if code is None:
        return None
    if 32601 <= code <= 32660:
        return code - 32600, True
    if 32701 <= code <= 32760:
        return code - 32700, False
    return None


def _is_metric_projected(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    unit = (crs.axis_info[0].unit_name or "").lower()
    return unit in ("metre", "meter", "m", "meters", "metres")


@dataclass
class ZoneCheck:
    """场址投影带检查结果。"""

    utm_zones: list[int]
    lon_min: float
    lon_max: float
    lon_span_deg: float
    cross_zone: bool

    def as_dict(self) -> dict:
        return {
            "utm_zones": self.utm_zones,
            "longitude_min_deg": self.lon_min,
            "longitude_max_deg": self.lon_max,
            "longitude_span_deg": self.lon_span_deg,
            "cross_zone": self.cross_zone,
        }


def check_zone(lons: np.ndarray, *, allow_cross_zone: bool = False) -> ZoneCheck:
    """检查场址是否明显跨越多个 UTM 带。

    跨带时无法用单一米制投影带保证全场地形精度，默认直接拒绝；
    ``allow_cross_zone=True`` 时仅返回检查结果（调用方负责提示风险）。
    """
    lons = np.asarray(lons, dtype=np.float64)
    zones = sorted({utm_zone_for_longitude(lon) for lon in lons.tolist()})
    lon_min = float(np.min(lons))
    lon_max = float(np.max(lons))
    span = lon_max - lon_min
    cross_zone = len(zones) > 1 or span > 6.0

    if cross_zone and not allow_cross_zone:
        raise CoordinateSystemError(
            f"场址经度范围 {lon_min:.4f}°~{lon_max:.4f}°（跨度 {span:.3f}°），"
            f"落入 UTM 第 {zones} 带，属于明显跨带场址，拒绝使用单一米制"
            "投影带；请拆分场址或改用覆盖全场的自定义投影坐标系"
        )
    return ZoneCheck(zones, lon_min, lon_max, span, cross_zone)


@dataclass
class ProjectionChain:
    """源 CRS 与米制计算坐标系之间的正、反变换链。

    变换链为::

        源坐标(lon,lat 或投影坐标)
          -> WGS 地理坐标(lon,lat)
          -> 目标米制投影坐标(easting,northing)
          -> 本地显示坐标(减去 local_origin)

    优化与绘图全部在“本地米制坐标”中进行，导出时逐级反变换回源 CRS。
    """

    source_crs: CRS
    target_crs: CRS
    source_is_geographic: bool
    local_origin: np.ndarray  # 目标投影坐标下的平移原点 (easting, northing)
    use_local_frame: bool
    zone: Optional[int]
    northern: Optional[bool]
    target_selection: str  # "auto_utm" | "manual" | "source_projected"
    zone_check: Optional[ZoneCheck]
    _src_to_geo: Optional[Transformer] = field(default=None, repr=False)
    _geo_to_src: Optional[Transformer] = field(default=None, repr=False)
    _geo_to_proj: Optional[Transformer] = field(default=None, repr=False)
    _proj_to_geo: Optional[Transformer] = field(default=None, repr=False)
    _geod: Optional[Geod] = field(default=None, repr=False)

    # ---- 单级变换 -----------------------------------------------------

    def source_to_geographic(self, coords: np.ndarray) -> np.ndarray:
        """源坐标 -> 地理坐标 (lon, lat)。源本身是地理坐标时原样返回。"""
        coords = np.asarray(coords, dtype=np.float64)
        if self.source_is_geographic:
            return coords.copy()
        return np.column_stack(
            self._src_to_geo.transform(coords[:, 0], coords[:, 1])
        )

    def geographic_to_source(self, lonlat: np.ndarray) -> np.ndarray:
        """地理坐标 (lon, lat) -> 源坐标。"""
        lonlat = np.asarray(lonlat, dtype=np.float64)
        if self.source_is_geographic:
            return lonlat.copy()
        return np.column_stack(
            self._geo_to_src.transform(lonlat[:, 0], lonlat[:, 1])
        )

    def geographic_to_projected(self, lonlat: np.ndarray) -> np.ndarray:
        """地理坐标 -> 目标米制投影坐标 (easting, northing)。"""
        lonlat = np.asarray(lonlat, dtype=np.float64)
        return np.column_stack(
            self._geo_to_proj.transform(lonlat[:, 0], lonlat[:, 1])
        )

    def projected_to_geographic(self, xy: np.ndarray) -> np.ndarray:
        """目标米制投影坐标 -> 地理坐标。"""
        xy = np.asarray(xy, dtype=np.float64)
        return np.column_stack(
            self._proj_to_geo.transform(xy[:, 0], xy[:, 1])
        )

    # ---- 端到端变换（管线使用） ---------------------------------------

    def to_projected(self, source_coords: np.ndarray) -> np.ndarray:
        """源坐标 -> 米制投影坐标。"""
        return self.geographic_to_projected(self.source_to_geographic(source_coords))

    def to_local(self, source_coords: np.ndarray) -> np.ndarray:
        """源坐标 -> 本地米制计算/显示坐标。"""
        projected = self.to_projected(source_coords)
        if self.use_local_frame:
            return projected - self.local_origin
        return projected

    def local_to_projected(self, local_xy: np.ndarray) -> np.ndarray:
        """本地米制坐标 -> 目标投影坐标。"""
        local_xy = np.asarray(local_xy, dtype=np.float64)
        if self.use_local_frame:
            return local_xy + self.local_origin
        return local_xy

    def to_source(self, local_xy: np.ndarray) -> np.ndarray:
        """本地米制坐标 -> 源坐标（反变换）。"""
        projected = self.local_to_projected(local_xy)
        return self.geographic_to_source(self.projected_to_geographic(projected))

    # ---- 往返误差 -----------------------------------------------------

    def roundtrip_errors(self, source_coords: np.ndarray) -> np.ndarray:
        """计算每个点 ``源 -> 本地 -> 源`` 的大地线往返误差（米）。"""
        source_coords = np.asarray(source_coords, dtype=np.float64)
        back = self.to_source(self.to_local(source_coords))
        g0 = self.source_to_geographic(source_coords)
        g1 = self.source_to_geographic(back)
        try:
            geod = self.source_crs.get_geod()
        except Exception:
            geod = Geod(ellps="WGS84")
        _, _, distances = geod.inv(g0[:, 0], g0[:, 1], g1[:, 0], g1[:, 1])
        return np.abs(np.asarray(distances, dtype=np.float64))

    def verify_roundtrip(
        self,
        source_coords: np.ndarray,
        tolerance_m: float,
    ) -> dict:
        """验证往返误差并返回摘要；超过容差时抛出异常。"""
        errors = self.roundtrip_errors(source_coords)
        summary = {
            "n_points_checked": int(errors.shape[0]),
            "max_error_m": float(np.max(errors)),
            "rms_error_m": float(np.sqrt(np.mean(errors**2))),
            "tolerance_m": float(tolerance_m),
            "passed": bool(np.max(errors) <= tolerance_m),
        }
        if not summary["passed"]:
            raise CoordinateSystemError(
                f"坐标往返误差 {summary['max_error_m']:.6f} m 超过约定容差 "
                f"{tolerance_m:.6f} m，坐标系或投影选择存在问题"
            )
        return summary

    # ---- 摘要 ---------------------------------------------------------

    def as_dict(self) -> dict:
        """导出可写入结果文件的变换链摘要。"""
        src_code = crs_code(self.source_crs)
        tgt_code = crs_code(self.target_crs)
        zone_text = None
        if self.zone is not None:
            hemi = "N" if self.northern else "S"
            zone_text = f"UTM {self.zone:02d}{hemi}"
        return {
            "source_crs": {
                "code": src_code,
                "name": self.source_crs.name,
                "type": "geographic" if self.source_is_geographic else "projected",
            },
            "target_crs": {
                "code": tgt_code,
                "name": self.target_crs.name,
                "type": "projected",
                "zone": zone_text,
                "selection": self.target_selection,
            },
            "transform_chain": (
                f"{src_code or self.source_crs.name} -> "
                f"{tgt_code or self.target_crs.name} -> "
                f"local_translation"
            ),
            "local_frame": {
                "enabled": self.use_local_frame,
                "origin_easting_m": float(self.local_origin[0]),
                "origin_northing_m": float(self.local_origin[1]),
                "note": "本地坐标 = 投影坐标 - 原点；仅用于优化计算与图形显示",
            },
            "forward_transform_proj4": self._geo_to_proj.to_proj4(),
            "inverse_transform_proj4": self._proj_to_geo.to_proj4(),
        }


def build_projection_chain(
    source_crs: CRS,
    boundary_geographic: np.ndarray,
    *,
    target_crs: Optional[str] = None,
    use_local_frame: bool = True,
    origin_mode: str = "bbox_center",
    allow_cross_zone: bool = False,
) -> ProjectionChain:
    """根据场址边界构建完整的正反投影变换链。

    Parameters
    ----------
    source_crs : CRS
        数据源坐标参考系。
    boundary_geographic : np.ndarray
        场界顶点的地理坐标 (lon, lat)，形状 (N, 2)。
    target_crs : Optional[str]
        人工指定的米制目标 CRS；缺省时按场址自动选择 UTM 带。
    use_local_frame : bool
        是否再平移到以场址为中心的本地米制坐标。
    origin_mode : str
        本地原点取法：``bbox_center``（默认，图形以场址为中心）或
        ``southwest``（坐标全部为正）。
    allow_cross_zone : bool
        是否允许跨带场址（默认拒绝）。
    """
    boundary_geographic = np.asarray(boundary_geographic, dtype=np.float64)
    if boundary_geographic.ndim != 2 or boundary_geographic.shape[1] != 2:
        raise CoordinateSystemError("场界地理坐标必须是形状 (N, 2) 的数组")

    # 先记录跨带情况（不在此处抛错）：是否拒绝取决于目标带如何确定。
    zone_info = check_zone(boundary_geographic[:, 0], allow_cross_zone=True)

    source_is_geographic = bool(source_crs.is_geographic)

    if target_crs is not None and str(target_crs).strip() != "":
        tgt = parse_crs(target_crs, role="目标米制坐标参考系")
        if not _is_metric_projected(tgt):
            raise CoordinateSystemError(
                f"目标坐标系 {crs_label(tgt)} 不是以米为单位的投影坐标系"
            )
        selection = "manual"
    elif not source_is_geographic:
        if not _is_metric_projected(source_crs):
            raise CoordinateSystemError(
                f"源坐标系 {crs_label(source_crs)} 既不是地理坐标系也不是米制"
                "投影坐标系，无法用于布局计算"
            )
        tgt = source_crs
        selection = "source_projected"
    else:
        # 只有自动选 UTM 带时，跨带才会导致单一投影带无法覆盖全场，默认拒绝
        if zone_info.cross_zone and not allow_cross_zone:
            raise CoordinateSystemError(
                f"场址经度范围 {zone_info.lon_min:.4f}°~{zone_info.lon_max:.4f}°"
                f"（跨度 {zone_info.lon_span_deg:.3f}°），落入 UTM 第 "
                f"{zone_info.utm_zones} 带，属于明显跨带场址，拒绝使用单一米制"
                "投影带；请用 --target-crs 指定覆盖全场的米制投影，拆分场址，"
                "或显式 --allow-cross-zone 放宽"
            )
        tgt, zone, northern = suggest_utm_crs(
            boundary_geographic[:, 0], boundary_geographic[:, 1]
        )
        target_crs = crs_code(tgt)
        selection = "auto_utm"

    utm_info = _epsg_utm_zone(tgt)
    zone = utm_info[0] if utm_info else None
    northern = utm_info[1] if utm_info else None

    if utm_info is not None and utm_info[0] not in zone_info.utm_zones:
        raise CoordinateSystemError(
            f"指定的目标投影带为 UTM 第 {utm_info[0]} 带，但场址位于第 "
            f"{zone_info.utm_zones} 带，目标坐标系与场址不匹配"
        )

    if source_is_geographic:
        src_to_geo = None
        geo_to_src = None
    else:
        src_to_geo = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        geo_to_src = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)

    geo_to_proj = Transformer.from_crs("EPSG:4326", tgt, always_xy=True)
    proj_to_geo = Transformer.from_crs(tgt, "EPSG:4326", always_xy=True)

    projected = np.column_stack(
        geo_to_proj.transform(boundary_geographic[:, 0], boundary_geographic[:, 1])
    )
    if not np.all(np.isfinite(projected)):
        raise CoordinateSystemError(
            f"场界坐标投影到 {crs_label(tgt)} 后出现非有限值，目标坐标系可能"
            "无法覆盖该场址"
        )

    if use_local_frame:
        if origin_mode == "bbox_center":
            origin = np.array([
                (projected[:, 0].min() + projected[:, 0].max()) / 2.0,
                (projected[:, 1].min() + projected[:, 1].max()) / 2.0,
            ])
        elif origin_mode == "southwest":
            origin = np.array([projected[:, 0].min(), projected[:, 1].min()])
        else:
            raise CoordinateSystemError(f"未知的本地原点模式: {origin_mode}")
    else:
        origin = np.zeros(2)

    return ProjectionChain(
        source_crs=source_crs,
        target_crs=tgt,
        source_is_geographic=source_is_geographic,
        local_origin=origin,
        use_local_frame=use_local_frame,
        zone=zone,
        northern=northern,
        target_selection=selection,
        zone_check=zone_info,
        _src_to_geo=src_to_geo,
        _geo_to_src=geo_to_src,
        _geo_to_proj=geo_to_proj,
        _proj_to_geo=proj_to_geo,
    )
