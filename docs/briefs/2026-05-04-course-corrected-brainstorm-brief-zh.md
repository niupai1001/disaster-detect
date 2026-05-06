# 纠偏头脑风暴简报：多波段灾害语义分割的第一实现切片

日期：2026-05-04

## 目标

在方向纠偏之后，重新开展一次对抗式头脑风暴，定义遥感灾害项目的第一实现切片。

校正后的项目目标是：

> 给定 Sentinel-2 等传感器的多波段卫星影像，判断是否存在灾害区域，识别灾害类型，并定位灾害发生的位置。

优先输出形式是多类别语义分割掩码或地理多边形，同时包含灾害类别，并能追溯到源波段。本轮头脑风暴必须明确防止项目再次滑回火灾单类 NBR/指数管线或 YOLO 边界框数据集管线。

## 必须吸收的上下文

主要纠偏输入：

- `docs/2026-05-04-research-memo-course-correction-method-.md`
- `docs/briefs/2026-05-04-course-corrected-multispectral-disaster-segmentation.md`
- 中文归档：`docs/briefs/2026-05-04-course-corrected-multispectral-disaster-segmentation-zh.md`

此前项目上下文只能作为历史证据，不能作为绑定方向：

- `task-7c670f442094`
- `.agent-team/artifacts/task-7c670f442094/brainstorm/brainstorm-BrainstormMemo.md`
- `.agent-team/artifacts/task-7c670f442094/technical/technical-TechnicalPlan.md`
- `.agent-team/artifacts/task-7c670f442094/review/review-review-lead-ReviewNote.md`

需要保留的研究证据层：

- `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- `docs/literature/2026-05-04-pre-project-method-landscape.md`
- `docs/research/learning-logs/`

## 来自纠偏备忘录的本地数据事实

已知类别：

| 类别 | 多边形数 | 事件数 | 灾种 |
| --- | ---: | ---: | --- |
| C2 | 500 | 271 | 泥石流 |
| C3 | 50 | 28 | 待确认 |
| C5 | 837 | 294 | 火灾 |

已知数据资产：

- 使用 `F01` 到 `F17` 等本地波段编号的多波段 `.tif` 影像；
- 本地 CSV 文件中的 WKT 多边形标注；
- 现有 `datasets/c5_fire_yolo/`，它只是火灾单类别实验资产，不能定义项目目标。

## 头脑风暴问题

如果项目目标是多波段卫星灾害定位与识别，第一实现切片应该是什么？

答案必须说明：

- 模型输入表示；
- 标签和掩码构建策略；
- 第一切片的类别范围；
- 预期输出 artifact；
- 训练前 QA gate；
- 明确暂缓的内容。

## 必须讨论的方法家族

工作坊必须讨论以下路线，只有给出理由后才能拒绝或暂缓：

- 从 WKT 派生掩码的多类别语义分割；
- 如果仍服务于项目目标，火灾优先语义分割作为窄启动路线；
- 面向粗多边形的弱监督和 ignore 区域设计；
- 多光谱通道堆叠和光谱指数作为输入特征；
- 如果灾前/灾后影像对通过验证，则考虑变化检测；
- 经典光谱指数基线作为诊断工具；
- 如果需要区分实例，则考虑实例分割；
- 地理基础模型或 SAM 辅助掩码生成和细化，但只能作为工具，不能当作真值；
- YOLO/DETR 目标检测只能作为兜底或提示生成路线。

## 部门分工

### research.lead

定义校正后的问题，并挑战任何不能输出灾害类型和空间掩码/多边形的路线。保留有证据支持的替代方案，但推荐研究上有效的第一切片。

### technical.lead

挑战可行性并定义最小可实现切片。说明数据契约、输入通道、掩码生成、切分策略依赖、架构边界，以及如何在不提前锁死模型的情况下开始实现。

### qa.tester

挑战可测试性。定义验收标准、泄漏风险、几何/掩码对齐检查、标签噪声策略、基线指标、可视化检查 gate，以及哪些条件会阻止训练或成功声明。

## 必须输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包含独立立场、挑战/反驳/协商内容、共识、权衡、决策记录、被拒绝或暂缓的选项，以及下一有效部门。

## 非目标

- 不训练模型。
- 不运行数据集变更或数据管线。
- 不把旧的火灾 YOLO 数据作为项目主目标。
- 不把 NBR 或任何光谱指数作为最终产品。
- 除定义第一实现切片外，不选择最终模型架构。
