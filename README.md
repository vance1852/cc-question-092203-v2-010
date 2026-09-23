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

勘测团队交付的带坐标参考系的 GeoJSON 场界与机位，可以直接导入；系统先投影到适合场址的米制计算坐标系，再做约束检查、布局优化，最后把相同机位编号和属性反变换导出回原坐标系，交还制图系统。

```bash
# 场界与机位都在 GeoJSON 中（文件内嵌 crs，或用 --source-crs 显式给出）
python -m wind_farm_opt \
  --boundary-geojson examples/site_boundary.geojson \
  --layout-geojson examples/turbine_layout.geojson \
  --output-dir output

# 文件不带 crs 成员时必须显式声明源坐标系
python -m wind_farm_opt --boundary-geojson site.geojson --source-crs EPSG:4326

# 手工指定米制计算坐标系（缺省时按场址经度自动选择 UTM 带）
python -m wind_farm_opt --boundary-geojson site.geojson \
  --source-crs EPSG:4490 --target-crs EPSG:32650
```

`examples/` 下提供了 WGS84 的示例场界（不规则多边形）与 12 台机位布局。

### 坐标系处理流程

1. **显式要求源坐标系**：GeoJSON 必须内嵌 `crs` 成员，或通过 `--source-crs` / 配置 `geo.source_crs` 给出（支持 `EPSG:4326`、`EPSG:4490`(CGCS2000)、OGC URN、`WGS84` 等写法）；缺失即拒绝，绝不按隐式坐标处理。
2. **选择米制计算坐标系**：地理坐标按场址中心经度自动选择 UTM 带（南北半球自动识别）；源已是米制投影坐标时直接沿用；也可用 `--target-crs` 指定。
3. **本地显示坐标**：在投影坐标基础上减去场址包络中心得到本地米制坐标，优化、间距/尾流计算与绘图都在该坐标系进行，图形保持以场址为中心的千米级范围；导出时逐级反变换回源 CRS。
4. **记录正反变换**：结果 `results.json` 的 `geo` 段保存来源文件、源/目标坐标系（EPSG、名称、选带方式）、本地原点、正/反投影 PROJ 字符串、UTM 跨带检查和往返误差摘要；导出的 GeoJSON 携带原 CRS。

### 输入校验（任一不满足即拒绝）

- 缺失坐标参考系，或命令行声明的 CRS 与文件内嵌 CRS 不一致；
- 坐标包含非有限数（NaN/Inf），或 2D/3D 维度混用（含高程分量）；
- 机位编号缺失或重复；
- 场址明显跨越多个 UTM 投影带（可用 `--allow-cross-zone` 放宽，仅记录风险）；
- 场界多边形未闭合、顶点不足或存在洞；
- 导入机位位于场界之外，或机位对间距小于最小间距（按转子直径倍数换算）；
- 把投影米制大数误标为经纬度等坐标参考系明显错误的情况。

导入布局在通过边界与间距检查后作为基线，并作为优化算法（GA/PSO）的初始种子，机位编号顺序在导入、优化、导出全过程保持不变。往返误差默认要求不超过 0.05 m，可用 `--roundtrip-tolerance` 调整。

### 配置文件方式

`geo` 配置段为可选，不配置时仍使用原有本地米制矩形/六边形边界，行为不变：

```json
{
  "geo": {
    "boundary_geojson": "examples/site_boundary.geojson",
    "layout_geojson": "examples/turbine_layout.geojson",
    "source_crs": "EPSG:4326",
    "target_crs": null,
    "origin_mode": "bbox_center",
    "use_local_frame": true,
    "boundary_tolerance_m": 1.0,
    "roundtrip_tolerance_m": 0.05,
    "allow_cross_zone": false,
    "export_geojson": true
  }
}
```

## 测试

```bash
python -m pytest tests/
```

测试覆盖 CRS 解析与选带、跨带拒绝、正反变换往返精度、GeoJSON 各类非法输入拒绝、边界/间距检查、编号与属性保持、以及 CLI 端到端导入-优化-导出-再导入往返。

