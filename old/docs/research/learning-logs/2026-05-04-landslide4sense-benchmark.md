# 学习日志：Landslide4Sense benchmark

日期：2026-05-04

来源：

- Paper: https://arxiv.org/abs/2206.00515
- Dataset: https://zenodo.org/records/10463239

## 论文旨在解决什么问题

Landslide4Sense 要解决的是遥感滑坡检测缺少统一、公开、可比较 benchmark 的问题。滑坡边界识别在光学影像中很难，因为滑坡的光谱外观会随土壤、植被、阴影、灾后恢复阶段和传感器条件变化。论文的目标是构建一个融合光学与地形信息的数据集，并用多种深度分割模型建立参考性能。

## 使用的数据

- 共 3,799 个遥感图像 patch。
- 输入融合 Sentinel-2 optical layers、ALOS PALSAR 派生的 DEM 和 slope。
- 每个像素标注为 landslide 或 non-landslide。
- 数据覆盖四个不同地区和事件时间：Iburi、Kodagu、Gorkha、Taiwan。
- 数据标签来自多源资料和人工标注，用于支持 landslide inventory 更新。

## 使用的方法

- 输入方法：把 Sentinel-2 光学波段与 DEM、slope 作为多源特征输入。
- 模型方法：从零训练并比较 11 个分割模型，包括 U-Net、ResU-Net、PSPNet、ContextNet、DeepLab-v2、DeepLab-v3+、FCN-8s、LinkNet、FRRN-A、FRRN-B、SQNet。
- 划分方法：在每个研究区域中用一部分 patch 训练，并在独立 patch 上测试，强调空间独立性。
- 评价方法：比较模型对 landslide mask 的像素级分割表现，论文报告 ResU-Net 在该任务中表现最好。

## 创新点

1. 数据创新：把光学 Sentinel-2 与 DEM/slope 结合，直接面向滑坡边界检测，而不是只做滑坡点或滑坡区域粗定位。
2. Benchmark 创新：统一比较多个经典分割模型，为后续滑坡遥感研究提供参考基线。
3. 泛化设置：数据跨多个灾害事件和地理区域，使模型必须面对跨区域变化，而不是只记住一个地区纹理。
4. 任务提醒：论文明确说明地形信息能帮助解决仅靠光学数据难以检测滑坡边界的问题。

## 实验结论与局限

- DEM 和 slope 对滑坡边界识别很关键。
- ResU-Net 在该 benchmark 上表现突出，但这不意味着它一定适合泥石流或火灾任务。
- 数据是 landslide benchmark，不是严格的 debris-flow benchmark；泥石流可能还需要河道、汇流、坡向、降雨和沟谷形态信息。
- 如果本项目没有 DEM/slope，直接套用滑坡分割路线会缺少关键输入。

## 对本项目的启发

1. 泥石流不能简单当作“另一类火灾”。它更接近 landslide/mass movement，需要地形与纹理特征。
2. 本项目应该尽早确认是否可以获取 DEM、slope、aspect、curvature、flow accumulation 等辅助数据。
3. 如果只有现有多光谱 TIFF，也要明确这会限制泥石流识别能力。
4. 评价划分必须考虑地点和事件，不能随机切 patch。

## 对头脑风暴的约束

- 泥石流路线必须单独讨论，不能附着在火灾路线后面。
- 技术部门应提出 DEM/slope 数据补充或缺失时的替代方案。
- QA 必须检查跨事件、跨区域和相邻 patch 泄漏。

## 下一步要查

- Landslide4Sense 的具体 band 顺序和数据格式。
- 是否有与泥石流更直接相关的公开 benchmark。
- 传统 OBIA + DEM/RF 方法与深度模型的差别。
