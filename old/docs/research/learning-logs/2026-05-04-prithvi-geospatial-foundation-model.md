# 学习日志：Prithvi geospatial foundation model

日期：2026-05-04

来源：

- NASA Earthdata release: https://www.earthdata.nasa.gov/news/blog/ibm-research-nasa-impact-release-open-source-geospatial-foundation-model-hugging-face
- NASA Science update: https://science.nasa.gov/science-research/ai-geospatial-model-earth/
- IBM Research Prithvi-EO note: https://research.ibm.com/publications/prithvi-eo-an-open-access-geospatial-foundation-model-advancing-earth-science-through-global-collaboration

## 论文/来源旨在解决什么问题

Prithvi 要解决通用 AI foundation model 与遥感地球观测数据结构不匹配的问题。普通视觉基础模型多基于自然图像，而地球观测影像具有多光谱、多时相、大范围地理覆盖和物理传感器约束。NASA/IBM 希望构建一个用 Harmonized Landsat and Sentinel-2 数据预训练的 geospatial foundation model，使其可迁移到 flood mapping、burn scar detection、crop classification 等下游任务。

## 使用的数据

- 预训练数据来自 NASA Harmonized Landsat and Sentinel-2 (HLS) imagery。
- Prithvi 初版基于 HLS 数据和自监督学习构建，并在 Hugging Face 开源。
- NASA/IBM 后续版本扩展到更全球化的数据覆盖，用于更多地球科学任务。
- 下游示例包括 post-disaster flood mapping 和 fire burn scar detection。

## 使用的方法

- 模型是 temporal Vision Transformer / geospatial foundation model。
- 预训练方式类似 masked modeling：让模型从遮盖信息中学习地表时空光谱表示。
- 下游使用方式是 fine-tuning：用较少标注数据适配 flood、burn scar、crop 等任务。
- 工程方式强调开放权重、Hugging Face pipeline 和可复用 fine-tuning workflow。

## 创新点

1. 遥感专用基础模型：使用 HLS 多光谱地球观测数据，而不是自然图像数据。
2. 开源工程创新：NASA/IBM 将模型和流程开放，降低遥感任务迁移学习门槛。
3. 数据效率创新：公开材料显示它在 flood 和 burn scar 等任务中可用更少标注数据达到较好表现。
4. 任务覆盖创新：一个 base model 支持多个地球科学任务，适合作为项目的迁移学习候选。

## 实验结论与局限

- Prithvi 在 flood mapping 与 burn scar detection 等任务上已有公开应用案例。
- 它更适合与 HLS/Landsat/Sentinel 类输入兼容的任务。
- 如果本项目本地波段、分辨率、投影或时间结构与 HLS 差异大，需要做适配层。
- Foundation model 不能替代标签审计、地理配准和评价设计。

## 对本项目的启发

1. 对多光谱、小标注数据，Prithvi/SatMAE 类遥感基础模型可能比 SAM 更合适。
2. 如果本地影像能映射到 HLS/Sentinel/Landsat 相关波段，应保留迁移学习路线。
3. 可以把 Prithvi 作为 burn scar / flood-like change detection 的候选 backbone 或 feature extractor。

## 对头脑风暴的约束

- 不要把“基础模型”混成一类：SAM 是 promptable natural-image segmentation，Prithvi 是 geospatial foundation model。
- 技术部门必须先做输入兼容性审计。
- QA 必须比较 Prithvi/SatMAE 路线与简单 baseline 的真实收益。

## 下一步要查

- 当前可用的 Prithvi 权重、输入 band schema、patch size 和 fine-tuning 示例。
- 是否有 burn scar segmentation notebook 可直接参考。
- 本地 TIFF 到 HLS band schema 的映射关系。
