# 学习日志：xBD disaster damage dataset

日期：2026-05-04

来源：

- Paper: https://arxiv.org/abs/1911.09296

## 论文旨在解决什么问题

xBD 要解决的是灾后快速评估建筑损毁缺少大规模、标准化、可训练数据的问题。传统灾害响应往往需要人在灾后 24-48 小时内现场评估建筑损害，危险且效率低。论文试图用灾前/灾后卫星影像、建筑 polygon 和损毁等级标签，推动 change detection 与 building damage assessment 自动化。

## 使用的数据

- 数据包含灾前与灾后 satellite imagery。
- 标签包括 building polygons、ordinal damage level，以及卫星元数据。
- 还包含环境因素的 bounding boxes 与标签，例如 fire、water、smoke。
- 数据集包含 850,736 个建筑标注，覆盖 45,362 km2 影像。
- 灾害类型跨多种自然灾害事件，核心单位是事件、时间、建筑对象和损毁等级。

## 使用的方法

- 数据构建方法：为同一地区组织 pre-event/post-event 影像对，并对建筑轮廓和损毁等级进行标注。
- 任务建模方法：把灾害识别拆为 localization 与 damage classification/change assessment，而不是只做单图分类。
- 评价思路：关注建筑定位、灾前灾后变化、损毁等级识别，以及面向灾害响应的可用性。
- 论文主要贡献是数据集与任务框架，不是提出一个单一模型架构。

## 创新点

1. 数据规模创新：当时最大的建筑灾害损毁评估数据集之一，标注量大且覆盖面积广。
2. 任务结构创新：把 pre/post change detection、object localization 和 ordinal damage label 放进同一框架。
3. 灾害响应创新：强调遥感 AI 对 humanitarian assistance 与 disaster recovery 的实际价值。
4. 元数据意识：不仅给图像和标签，也保留 satellite metadata，方便分析影像条件对模型表现的影响。

## 实验结论与局限

- 很多灾害任务不是单图像识别，而是灾前/灾后变化理解。
- xBD 的对象是建筑损毁，不能直接替代火灾或泥石流范围分割。
- polygon 是建筑对象，不是灾害边界；本项目借鉴的是任务框架和时间配对思想。
- 损毁等级是 ordinal label，评价和建模难度高于普通二分类。

## 对本项目的启发

1. 在训练任何单图模型前，先审计本地数据是否有可用的灾前/灾后影像。
2. 火灾烧毁区、泥石流、滑坡都可能更适合变化检测。
3. 需要把 event key 和 image date 作为一级字段，不只是文件名字符串。
4. 分割之外，未来可考虑灾害强度/等级，但当前先不扩展。

## 对头脑风暴的约束

- 必须保留 change detection 路线。
- QA 要防止同一事件不同日期泄漏到 train/val/test。
- 技术部门要提出 pre/post pair builder 的可能性审计。

## 下一步要查

- 本地每个事件是否有灾前影像。
- xBD 常用模型架构：Siamese U-Net、变化检测网络、two-stage building localization + damage classification。
- 是否存在面向火灾/滑坡的 pre/post segmentation benchmark。
