# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## GeoJSON 场界与机位导入导出

勘测交付的带坐标系 GeoJSON 场界/机位可直接导入，工具会显式处理坐标参考系，
在适合场址的米制坐标系中完成优化，再把相同机位编号和地理位置逆变换回源系统。

```bash
# 基本用法：WGS84 经纬度 GeoJSON，自动选择米制计算坐标系
python -m wind_farm_opt --import-geojson site.geojson --output-dir output

# 显式指定源坐标系（覆盖文件 crs 成员）或目标米制坐标系
python -m wind_farm_opt --import-geojson site.geojson \
    --source-crs EPSG:32650 --metric-crs EPSG:32650

# 强制选带偏好：auto（默认，中国境内 CGCS2000 三度带，境外 WGS84 UTM）/ cgcs2000 / utm
python -m wind_farm_opt --import-geojson site.geojson --metric-crs-preference utm
```

GeoJSON 约定：

- 一个 `FeatureCollection` 包含一个场界 `Polygon`（或单多边形 `MultiPolygon`，
  可选内环表示禁建区）和若干机位 `Point`；机位编号取 `Feature.id`，
  其次 `properties.turbine_id`/`id`/`name`，全部属性随导出保留。
- 源坐标参考系读取 GeoJSON 的 `crs` 成员（GeoJSON 2008 风格）或 `--source-crs`；
  无 `crs` 成员时按 RFC 7946 视为 WGS84 经纬度。**投影量级坐标缺失坐标系会被拒绝**，
  避免把经纬度当平面距离手工平移。
- 米制计算坐标系按场界自动选带：中国境内选 CGCS2000 三度带高斯-克吕格
  （EPSG 4534 系列，中央经线 75E–135E），境外选 WGS84 UTM；正反变换由 pyproj/PROJ 完成。
- 优化后导出 `optimized_layout.geojson`（`--no-optimization` 时为
  `evaluated_layout.geojson`），坐标系与源一致，并写入 `crs` 成员和
  `sfp_coordinate_transform` 转换摘要（源/目标 CRS、转换链路、往返残差、经纬度范围等）。
- `results.json` 的 `coordinate_transform` 段同样记录来源文件、源/目标坐标系、
  转换链路、往返最大残差、贴边机位等信息。

校验与拒绝（命令行退出码 2）：

- 缺失坐标参考系且无法按 RFC 7946 安全默认；
- 坐标维度混合（2D/3D 混用）、非有限坐标（NaN/Infinity）；
- 机位编号缺失或重复、多个场界多边形、不支持的几何类型；
- 场址明显跨越投影带边界（单投影带无法保证全场距离精度）；
- 导入机位必须先通过边界检查（`--boundary-tolerance`，默认 0.5 m）和
  最小间距检查（按 `--min-spacing` 倍数与机型转子直径计算）。

正反变换在导入时对场界顶点与全部机位做往返校验，默认容差 1 mm
（`--roundtrip-tolerance`）；导出时再次复核，超差拒绝写出。
导入模式下图形仍以便于阅读的场址局部米制范围呈现（坐标轴标注投影带原点
东距/北距），原本地坐标配置（矩形宽高、中心等）与不带 `--import-geojson`
的运行完全不受影响。

GeoJSON 专项测试：

```bash
PYTHONPATH=. python3 quick_test_geojson.py
```
