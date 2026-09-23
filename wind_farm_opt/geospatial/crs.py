"""坐标参考系（CRS）解析与场址适配米制坐标系选择。

职责：
- 解析 GeoJSON 的 ``crs`` 成员、命令行覆盖值（EPSG / URN / WKT / PROJJSON）；
- 判定源坐标系是地理坐标系还是投影坐标系；
- 按场址经纬度范围自动选择米制计算坐标系
  （中国境内优先 CGCS2000 三度带高斯-克吕格投影，境外使用 WGS84 UTM）；
- 识别明显跨带场址，避免把跨越投影带边界的场址放在单一投影带中计算。

仅依赖 pyproj/PROJ，不在内存中自行实现椭球变换。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from pyproj import CRS
from pyproj.exceptions import CRSError


# ---- 例外 ----------------------------------------------------------------

class CRSMissingError(ValueError):
    """源数据未声明坐标参考系，且无法依据标准默认值安全推断。"""


class CrossZoneError(ValueError):
    """场址坐标跨越投影带边界，单一米制坐标系无法保证全场距离精度。"""


# ---- 常量 ----------------------------------------------------------------

# RFC 7946 规定无 crs 成员的 GeoJSON 一律视为 WGS84 经纬度。
RFC7946_DEFAULT = "EPSG:4326"

# OGC CRS84 与 WGS84 经纬度等价（轴序均为经度、纬度）。
_CRS84_URNS = {
    "urn:ogc:def:crs:ogc:1.3:crs84",
    "http://www.opengis.net/def/crs/ogc/1.3/crs84",
}

# 中国陆地及近岸大致经纬度窗口，用于在自动模式下选择 CGCS2000 三度带。
_CHINA_LON_MIN, _CHINA_LON_MAX = 73.0, 135.5
_CHINA_LAT_MIN, _CHINA_LAT_MAX = 2.0, 54.5

# CGCS2000 三度带高斯-克吕格投影（不含带号前缀，东伪偏移 500000）：
# 中央经线 75E..135E，步长 3°，EPSG 4534 起连续编号。
_CGCS2000_CM_FIRST_EPSG = 4534
_CGCS2000_CM_FIRST_LON = 75
_CGCS2000_CM_STEP = 3


# ---- 数据结构 -------------------------------------------------------------

@dataclass(frozen=True)
class SourceCRSInfo:
    """源坐标参考系解析结果。

    Attributes
    ----------
    crs : CRS
        pyproj 坐标参考系对象。
    code : str
        规范化后的可读编码，如 ``EPSG:4326``；无法映射 EPSG 时使用 WKT 摘要。
    declared_in : str
        来源：``crs_member``（GeoJSON 成员）、``cli_argument``（命令行覆盖）
        或 ``rfc7946_default``（按 RFC 7946 默认 WGS84）。
    assumed : bool
        是否为未经显式声明的默认假设。
    """

    crs: CRS
    code: str
    declared_in: str
    assumed: bool

    @property
    def is_geographic(self) -> bool:
        return bool(self.crs.is_geographic)

    @property
    def is_metric_projected(self) -> bool:
        return bool(self.crs.is_projected) and self.crs.axis_info[0].unit_name == "metre"


@dataclass(frozen=True)
class MetricCRSRecommendation:
    """自动选择的米制计算坐标系。"""

    crs: CRS
    code: str
    family: str  # "cgcs2000_gk3" 或 "utm"
    zone_label: str


@dataclass(frozen=True)
class GeoExtent:
    """场址经纬度范围。"""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @property
    def lon_span(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.lon_min + self.lon_max) / 2.0,
            (self.lat_min + self.lat_max) / 2.0,
        )


# ---- CRS 解析 -------------------------------------------------------------

def _normalize_crs_text(text: str) -> str:
    """把各种 EPSG/URN 写法归一化为 pyproj 可识别的字符串。"""
    raw = text.strip()
    low = raw.lower()

    if low in _CRS84_URNS or low == "crs84":
        return RFC7946_DEFAULT

    # urn:ogc:def:crs:EPSG::4326 / http://www.opengis.net/def/crs/EPSG/0/4326
    m = re.match(
        r"^(?:urn:ogc:def:crs:|https?://www\.opengis\.net/def/crs/)"
        r"epsg[:/](?:\d+)[:/](\d+)$",
        low,
    )
    if m:
        return f"EPSG:{m.group(1)}"

    m = re.match(r"^epsg\s*:\s*(\d+)$", low)
    if m:
        return f"EPSG:{m.group(1)}"

    return raw


def crs_to_text(crs: CRS) -> str:
    """返回便于写入摘要的编码文本。"""
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.name or crs.to_proj4()


def build_crs(spec: Any) -> CRS:
    """从字符串、PROJJSON 字典或 pyproj CRS 构建 CRS。"""
    if isinstance(spec, CRS):
        return spec
    if isinstance(spec, dict):
        # PROJJSON（pyproj 可直接反序列化）
        try:
            return CRS.from_json_dict(spec)
        except CRSError:
            # GeoJSON 2008 风格 {"type":"name","properties":{"name":...}}
            name = (spec.get("properties") or {}).get("name")
            if isinstance(name, str):
                return build_crs(name)
            raise
    if isinstance(spec, str) and spec.strip():
        return CRS.from_user_input(_normalize_crs_text(spec))
    raise CRSError(f"无法识别的坐标系描述: {spec!r}")


def resolve_source_crs(
    crs_member: Optional[Any],
    cli_override: Optional[str],
    looks_geographic: Optional[bool],
) -> SourceCRSInfo:
    """确定源坐标参考系。

    优先级：命令行显式覆盖 > GeoJSON ``crs`` 成员 > RFC 7946 默认 WGS84。

    Parameters
    ----------
    crs_member
        GeoJSON 顶层的 ``crs`` 成员内容。
    cli_override
        命令行 ``--source-crs`` 提供的坐标系文本。
    looks_geographic
        数据坐标是否明显为经纬度（全部落在合法经纬度范围内）。
        当只能回退到 RFC 默认值、但坐标却是投影坐标量级时，拒绝导入。
    """
    if cli_override is not None:
        crs = build_crs(cli_override)
        return SourceCRSInfo(crs=crs, code=crs_to_text(crs),
                             declared_in="cli_argument", assumed=False)

    if crs_member is not None:
        crs = build_crs(crs_member)
        return SourceCRSInfo(crs=crs, code=crs_to_text(crs),
                             declared_in="crs_member", assumed=False)

    if looks_geographic:
        crs = build_crs(RFC7946_DEFAULT)
        return SourceCRSInfo(crs=crs, code=RFC7946_DEFAULT,
                             declared_in="rfc7946_default", assumed=True)

    raise CRSMissingError(
        "GeoJSON 未声明坐标参考系，且坐标量级不像经纬度。"
        "请在文件中加入 crs 成员（GeoJSON 2008 风格）或通过 --source-crs 显式指定，"
        "例如 --source-crs EPSG:32650。"
    )


# ---- 经纬度范围与跨带判定 --------------------------------------------------

def build_extent(lonlat) -> GeoExtent:
    """从 (N, 2) 经纬度数组构造场址范围。"""
    lons = [p[0] for p in lonlat]
    lats = [p[1] for p in lonlat]
    return GeoExtent(min(lons), max(lons), min(lats), max(lats))


def cgcs2000_zone_central_meridian(lon: float) -> int:
    """经度对应的 CGCS2000 三度带中央经线（75..135，步长 3）。"""
    cm = int(math.floor((lon + _CGCS2000_CM_STEP / 2.0) / _CGCS2000_CM_STEP)) * _CGCS2000_CM_STEP
    return min(max(cm, _CGCS2000_CM_FIRST_LON), 135)


def cgcs2000_epsg_for_central_meridian(cm: int) -> int:
    """中央经线对应的 CGCS2000 三度带 EPSG 编码（CM 系列）。"""
    if (cm - _CGCS2000_CM_FIRST_LON) % _CGCS2000_CM_STEP != 0:
        raise ValueError(f"中央经线必须是 {_CGCS2000_CM_STEP}° 的奇数半倍数: {cm}")
    return _CGCS2000_CM_FIRST_EPSG + (cm - _CGCS2000_CM_FIRST_LON) // _CGCS2000_CM_STEP


def utm_zone_number(lon: float) -> int:
    """经度对应的 UTM 带号（1..60）。"""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return min(max(zone, 1), 60)


def _in_china(extent: GeoExtent) -> bool:
    center_lon, center_lat = extent.center
    return (
        _CHINA_LON_MIN <= center_lon <= _CHINA_LON_MAX
        and _CHINA_LAT_MIN <= center_lat <= _CHINA_LAT_MAX
    )


def recommend_metric_crs(
    extent: GeoExtent,
    preference: str = "auto",
) -> MetricCRSRecommendation:
    """根据场址经纬度范围推荐米制计算坐标系。

    Parameters
    ----------
    extent
        场址经纬度范围。
    preference
        - ``auto``：中国境内选 CGCS2000 三度带，其余选 WGS84 UTM；
        - ``cgcs2000``：强制 CGCS2000 三度带；
        - ``utm``：强制 WGS84 UTM。
    """
    if preference not in {"auto", "cgcs2000", "utm"}:
        raise ValueError(f"未知的米制坐标系偏好: {preference}")

    family = preference
    if family == "auto":
        family = "cgcs2000" if _in_china(extent) else "utm"

    center_lon, center_lat = extent.center

    if family == "cgcs2000":
        cm = cgcs2000_zone_central_meridian(center_lon)
        epsg = cgcs2000_epsg_for_central_meridian(cm)
        crs = CRS.from_epsg(epsg)
        return MetricCRSRecommendation(
            crs=crs, code=f"EPSG:{epsg}", family=family,
            zone_label=f"CGCS2000 三度带 中央经线{cm}E",
        )

    zone = utm_zone_number(center_lon)
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    crs = CRS.from_epsg(epsg)
    hemi = "北" if center_lat >= 0 else "南"
    return MetricCRSRecommendation(
        crs=crs, code=f"EPSG:{epsg}", family=family,
        zone_label=f"WGS84 UTM {zone}{'N' if center_lat >= 0 else 'S'}（{hemi}半球）",
    )


def check_cross_zone(
    extent: GeoExtent,
    family: str,
    points_lonlat=None,
) -> Optional[dict]:
    """检查场址是否明显跨越投影带边界。

    Returns
    -------
    Optional[dict]
        单带时返回 None；跨带时返回跨带详情字典（供例外与摘要使用）。
    """
    if points_lonlat is None or len(points_lonlat) == 0:
        sample_lons = [extent.lon_min, extent.lon_max, *extent.center[:1]]
        points_lonlat = [(lon, extent.center[1]) for lon in sample_lons]

    if family in ("cgcs2000", "cgcs2000_gk3"):
        zones = sorted({
            cgcs2000_zone_central_meridian(float(p[0])) for p in points_lonlat
        })
        width = 3.0
        label = "三度带中央经线"
    elif family == "utm":
        zones = sorted({utm_zone_number(float(p[0])) for p in points_lonlat})
        width = 6.0
        label = "UTM带号"
    else:
        return None

    if len(zones) <= 1:
        return None

    return {
        "family": family,
        "zone_label": label,
        "zones": zones,
        "lon_span_deg": extent.lon_span,
        "max_span_deg": width,
        "lon_min": extent.lon_min,
        "lon_max": extent.lon_max,
    }


def validate_explicit_metric_crs(crs: CRS) -> None:
    """校验用户显式指定的米制坐标系确实是以米为单位的投影坐标系。"""
    if not crs.is_projected:
        raise CRSError(f"目标坐标系必须是投影坐标系: {crs_to_text(crs)}")
    unit = crs.axis_info[0].unit_name
    if unit != "metre":
        raise CRSError(
            f"目标坐标系长度单位必须为米，当前为 {unit}: {crs_to_text(crs)}"
        )
