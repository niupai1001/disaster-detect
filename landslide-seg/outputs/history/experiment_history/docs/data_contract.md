# 数据契约

本文档定义"什么样的数据才允许进入本模型"，以及现有数据的实测状态。
**契约优先于便利**：不满足契约的样本一律拒绝，不做静默修补。

> **当前满足契约的数据源**：**Sen12Landslides**（`B02–B12` 含 B8A + SCL + 日期 + CRS），
> 其缺失的 B01/B09 由**同一 L2A 产品、同一天**回补。
> 评审过程与替代数据集（CAS 滑坡数据集）的否决理由见 `docs/data_source_review.md`。
> Landslide4Sense 不满足契约，仅保留作旁支实验数据。

---

## 1. 模型侧输入要求（来自官方实现）

| 用途 | 波段 | 通道数 | 空间 | 其他 |
|---|---|---|---|---|
| TerraMind `S2L2A`（含 TiM） | B01,B02,B03,B04,B05,B06,B07,B08,**B8A**,B09,B11,B12 | 12 | 224×224（预训练 tokenizer 固定 14×14 token） | DN 尺度，官方 mean/std 标准化 |
| AnySat `s2` | B02,B03,B04,B05,B06,B07,B08,**B8A**,B11,B12 | 10 | 任意方形，`patch_size` 为米且是 10 的倍数 | 需 `s2_dates`（一年内日序，0=1/1）；DN 尺度，官方 mean/std |
| 标签 | — | 1 | 与影像同网格 | `{0,1}`，未知区域用 `ignore_index` |

要点：
- **TerraMind 不支持波段子集**（`build_terrammind_tim` 中 `bands != None` 直接
  `NotImplementedError`），缺 B8A 即不可用；
- B01/B09 原生 60m，重采样到 10m **不增加真实信息**；
- AnySat 的 `patch_size` 单位是米（`forward_release(patch_size//10)`），
  1280m 图块上 40m → 32×32=1024 patch，符合官方"不超过约 1024 patch"的建议；
- AnySat dense 输出对 s2 固定在原生 10m 网格，通道为 `2×D`（base: 1536）。

---

## 2. 现有数据实测（Landslide4Sense）

来源：`https://hf-mirror.com/datasets/ibm-nasa-geospatial/Landslide4sense`（IBM-NASA 镜像，
与原版一致）。本地路径 `N:/Users/yzh/Disater/{images,annotations}`。

```
images/{split}/image_{i}.h5   ->  key "img",  (128,128,14) float64
annotations/{split}/mask_{i}.h5 -> key "mask", (128,128) uint8 {0,1}
split: train 3799 / validation 245 / test 800
```

**通道语义（官方数据卡 + 实测确认）**

| 索引 | 0–11 | 12 | 13 |
|---|---|---|---|
| 内容 | Sentinel-2 **B01…B12**（含 B10，无 B8A） | ALOS PALSAR 坡度 | ALOS PALSAR DEM |

实测佐证：
- 相关系数矩阵中 5/6/7 通道（红边—NIR）互相 0.95–0.98，10/11（SWIR1/2）为 0.91；
- 索引 12 与所有光学波段近乎无关（−0.15…0.02），而索引 13 与 B09 相关 0.47 —— 符合
  "坡度独立、DEM 与地形相关"的物理预期；
- 首通道与次通道相关仅 0.86（其余相邻对 0.96+），符合 B01 噪声较大的特征。

**与契约的差距**

| 项 | 状态 |
|---|---|
| 12 波段（含 B8A） | ❌ 缺 B8A |
| 10 波段 AnySat | ❌ 缺 B8A |
| 日期 | ❌ 无 |
| 地理坐标 / CRS / 产品 ID | ❌ 无 |
| 云/阴影掩膜 | ❌ 无 |
| 事件/区域标识 | ❌ 无 |
| 掩膜 {0,1} | ✅ |
| 数值为 DN/反射率 | ❌ 已被归一化（见 §4） |

**因此：Landslide4Sense 只能用于**
1. 波形/掩膜统计与数据管线单元测试；
2. SAM3 / RemoteSAM 这类"只用 RGB 或 3–14 通道自定义输入"的旁支实验
   （已有 `exp/` 下的结果），**不能**用于本项目的双骨干正式训练。

---

## 3. 补齐 12 波段的正确流程（待执行）

1. 以带地理坐标与日期的滑坡标注（Sen12Landslides 或自建）为索引；
2. 在 Copernicus Data Space 按"同一时相、同一区域"检索 L2A 产品，记录产品 ID 与基线；
3. 下载 12 个波段，统一投影重采样到 10m 参考网格（B01/B09 从 60m 重采样，标注为插值）；
4. 与标注逐波段核验值域、日期、空间对齐（harmonized 数据的辐射 offset 需校正）；
5. 由同一份 `X12` 派生 `X10` 给 AnySat，两支路各自用**官方统计**独立标准化；
6. 不完整/无法匹配的样本进入 `data/quarantine/` 并记录原因；
7. 产出清单：`scene_id, event_id, date, bbox, crs, bands_present, cloud_fraction,
   mask_coverage, split, source_uri, label_status`。

---

## 4. 辐射量级实测（为什么不能直接喂预训练模型）

| 指标 | 实测值 | 说明 |
|---|---|---|
| 各波段中位数 | 1.14 – 1.62 | 含 SLOPE=1.43、DEM=1.20，量级完全一致 |
| 单像素取值范围 | 0 – 6.40 | 非 0–1，也非 0–10000 |
| NDVI 均值 | ≈ −0.016 | 真实陆地应显著为正（森林 0.5+） |

结论：存储值已被数据集提供方归一化/混合，**不可反解为反射率或 DN**。
任何"乘一个系数凑成 DN"的做法都是伪造辐射定标，禁止。

---

## 5. 掩膜与划分

- 掩膜值域 `{0,1}`，实测正样本像素占比中位数在 train 子集上约 0.9%（1–937 px），
  存在**空图块**（前 16 个 train 样本中 9 个全零），训练时必须保留负样本；
- 官方 train/val/test 为随机划分，**无事件/地理互斥信息**，
  本项目要求的 `split_by: [event_id, geography]` 当前无法满足，已在 `blockers.md` B3 记录；
- 云遮挡像素当前无法识别（无云掩膜），`valid` 只做了有限值/非零检查，
  **不代表已排除云影**。这一点会直接导致负样本被污染，属于 B3 的下游影响。

---

## 6. 契约的实施位置

| 约束 | 代码位置 |
|---|---|
| 波段顺序与缺失检测 | `src/landslide_seg/data/band_mapping.py` |
| 严格/合成两种策略 | `band_mapping.resolve(policy=...)` |
| 样本元数据与不可用原因 | `src/landslide_seg/data/loader.py::SampleMeta` |
| 批次构造与 provenance | `src/landslide_seg/data/pipeline.py::batches_from_samples` |
| 合成张量（仅 smoke） | `src/landslide_seg/data/pipeline.py::make_synthetic_batch` |
