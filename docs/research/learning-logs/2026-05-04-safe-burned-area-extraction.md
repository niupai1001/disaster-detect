# 学习日志：SAFE burned area extraction

日期：2026-05-04

来源：

- Paper: https://www.mdpi.com/2072-4292/17/1/54

## 论文旨在解决什么问题

SAFE 要解决快速、高精度 wildfire burned-area extraction 的问题。传统火灾监测常在时间分辨率和空间分辨率之间取舍：MODIS/VIIRS 时效好但空间分辨率粗，Sentinel-2 空间分辨率好但处理量大且时效有限。论文希望结合热点产品的时效性、Sentinel-2 的高空间细节和 SAM 的泛化能力，实现不依赖本地训练的快速烧毁区提取。

## 使用的数据

- Sentinel-2 optical imagery，用于高空间分辨率 burned-area candidates 与后续分割。
- MODIS MCD64A1 burned area product，以及 MODIS/VIIRS hotspot product，用于快速定位可能火区。
- Dynamic World land-cover data，用于约束和排除明显非燃烧地物。
- 实验覆盖 grassland 和 forest 两类典型 wildfire 场景，包含蒙古南部、哈萨克斯坦西北部、澳大利亚南部、坦桑尼亚东北部等测试区域。

## 使用的方法

- 两步定位策略：先用 MODIS/VIIRS 火点和 Sentinel-2 生成候选区域，再用 SAM/SAM2 对候选区域进行细化分割。
- 时间对齐：选择火灾日期附近 Sentinel-2 影像，并在需要时利用灾前/灾后信息辅助候选区生成。
- 空间聚合：对火点进行拓扑分析和聚类，合并邻近火点，扩展 AOI 边界以减少漏检。
- 后处理：结合 land-cover、spectral indices 和局部规则剔除非 burned area。
- 衍生用途：SAFE 生成的高分辨率 burned-area data 可用于训练轻量化区域模型。

## 创新点

1. 多源融合创新：把 MODIS/VIIRS 的高时效火点与 Sentinel-2 的高空间分辨率结合。
2. 基础模型应用创新：把 SAM 用作候选 burned area 精细化工具，而不是从零训练分割模型。
3. 计算效率创新：先缩小候选区域再分割，减少全图运行 SAM 的成本。
4. 数据闭环创新：用 SAFE 结果进一步训练区域轻量模型，形成快速提取到本地模型的过渡。

## 实验结论与局限

- 论文报告 SAFE 在多个测试区域上的 F1-score 平均高于对比方法。
- SAFE 对 grassland 和 forest wildfire 场景表现较好，适合快速响应。
- 农田收割、翻耕、裸土等地物可能与烧毁痕迹混淆。
- SAFE 依赖热点产品、Sentinel-2 数据可用性和候选区质量；不能无条件迁移到所有灾害类型。

## 对本项目的启发

1. 火灾不一定只能做监督训练；可以先做 MODIS/VIIRS 或本地事件记录约束下的候选区提取。
2. SAM 的合理位置是候选区精细分割，而不是替代遥感物理特征。
3. 本项目应考虑“候选生成 -> 分割细化 -> 光谱/地理 QA -> 轻量模型训练”的闭环。

## 对头脑风暴的约束

- 基础模型路线必须与遥感数据产品和光谱规则结合。
- 技术部门要评估本地是否有热点产品、事件时间或灾前灾后影像可用。
- QA 要专门评估农田、裸土、阴影、云和烟导致的 false positive。

## 下一步要查

- SAFE 具体使用哪些 spectral indices 和 thresholds。
- 是否可在没有 MODIS/VIIRS 火点的本地数据上用 CSV 事件 WKT 替代候选定位。
- SAFE 输出如何转换为可训练标签而不放大伪标签错误。
