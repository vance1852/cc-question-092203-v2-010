"""地理空间数据（GeoJSON）导入导出与坐标参考系处理。

公共接口：

- :func:`import_geojson` —— 导入场界与机位，完成 CRS 解析、选带、
  跨带/有限性/维度/编号校验，以及边界与间距检查；
- :func:`export_geojson` / :func:`build_export_geojson` —— 优化后逆变换
  导出，保留编号与属性，附带转换摘要并做往返复核；
- :class:`CoordinateTransformer` —— 源 CRS 与本地米制 CRS 的正反变换；
- 异常类型：:class:`GeoJSONValidationError`、:class:`CRSMissingError`、
  :class:`CrossZoneError`、:class:`LayoutConstraintError`、
  :class:`RoundTripError`、:class:`ExportError`。
"""

from .crs import (
    CRSMissingError,
    CrossZoneError,
    GeoExtent,
    MetricCRSRecommendation,
    SourceCRSInfo,
    build_crs,
    crs_to_text,
)
from .transform import CoordinateTransformer, RoundTripError, TransformSummary
from .validation import (
    BoundaryGeometry,
    GeoJSONValidationError,
    ParsedGeoJSON,
    TurbineFeature,
    load_geojson_file,
    load_geojson_text,
    parse_geojson,
)
from .importer import ImportedSite, LayoutConstraintError, import_geojson
from .exporter import ExportError, build_export_geojson, export_geojson

__all__ = [
    "import_geojson",
    "export_geojson",
    "build_export_geojson",
    "ImportedSite",
    "CoordinateTransformer",
    "TransformSummary",
    "SourceCRSInfo",
    "MetricCRSRecommendation",
    "GeoExtent",
    "GeoJSONValidationError",
    "CRSMissingError",
    "CrossZoneError",
    "LayoutConstraintError",
    "RoundTripError",
    "ExportError",
    "load_geojson_file",
    "load_geojson_text",
    "parse_geojson",
    "ParsedGeoJSON",
    "BoundaryGeometry",
    "TurbineFeature",
    "build_crs",
    "crs_to_text",
]
