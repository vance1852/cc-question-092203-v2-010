"""地理坐标与 GeoJSON 导入导出子包。

负责在勘测交付的带坐标参考系的 GeoJSON 与内部本地米制坐标之间建立
可记录、可验证的正反变换，详见 :mod:`wind_farm_opt.geo.crs` 与
:mod:`wind_farm_opt.geo.geojson`。
"""

from .crs import (
    CoordinateSystemError,
    ProjectionChain,
    build_projection_chain,
    crs_code,
    crs_label,
    parse_crs,
    resolve_source_crs,
    suggest_utm_crs,
    utm_zone_for_longitude,
)
from .geojson import (
    GeoJSONError,
    ImportedSite,
    LayoutValidationReport,
    TurbineRecord,
    export_boundary_geojson,
    export_layout_geojson,
    import_site,
    parse_boundary,
    parse_layout,
    read_geojson,
)

__all__ = [
    "CoordinateSystemError",
    "GeoJSONError",
    "ImportedSite",
    "LayoutValidationReport",
    "ProjectionChain",
    "TurbineRecord",
    "build_projection_chain",
    "crs_code",
    "crs_label",
    "export_boundary_geojson",
    "export_layout_geojson",
    "import_site",
    "parse_boundary",
    "parse_crs",
    "parse_layout",
    "read_geojson",
    "resolve_source_crs",
    "suggest_utm_crs",
    "utm_zone_for_longitude",
]
