# 数据源评审：CAS 滑坡数据集 vs Sen12Landslides

> 结论先行：
> - **CAS Landslide Dataset（中科院成都山地所）不可用于本项目**：全部为 RGB 三通道
>   uint8 TIF，且无坐标系/日期，无法满足 TerraMind(12 波段) 与 AnySat(10 波段) 的输入协议。
> - **Sen12Landslides 可用**：其 S2 文件实测含 `B02–B08 / B8A / B11 / B12`（**含 B8A**，正好是 AnySat 的
>   10 波段）+ SCL 云掩膜 + 掩膜 + DEM + 坐标系 + 日期；TiM 缺的 B01/B09 可从
>   同一 L2A 产品按**同一天**回补，已实测跑通。
>
> 证据脚本：`scripts/audit_sen12.py`、`scripts/real_inference.py`、`scripts/train_full.py --source sen12`

---

## 1. CAS Landslide Dataset

**出处**：Xu et al., *Sci Data* 11, 12 (2024)，中科院·水利部成都山地灾害与环境研究所 AI 团队。
Zenodo `10.5281/zenodo.10294997`（CC BY-NC 4.0），另在国家对地观测科学数据中心（noda.ac.cn）有镜像。

**规模**：20,865–22,049 张 512×512 影像，9 个区域，卫星 + 无人机。

**实测方式**：不下载整包（单包 24 MB–2 GB），用 HTTP Range 远程读取 ZIP 中央目录，
只解出每个子集的第一张影像，再用 rasterio 检查。

| 子集 | 传感器 | 影像通道 | dtype | 尺寸 | CRS | 掩膜 |
|---|---|---|---|---|---|---|
| Tiburon Peninsula（Sentinel） | Sentinel-2 | **3 (RGB)** | uint8 0–255 | 512×512 | **None** | 1 通道 {0,1} |
| Moxitaidi (SAT) | Sentinel-2 | **3 (RGB)** | uint8 | 512×512 | **None** | — |
| Longxi River (SAT) | GF-1 | **3 (RGB)** | uint8 | 512×512 | **None** | — |
| Wenchuan | Landsat 5 | **3 (RGB)** | uint8 | 512×512 | **None** | — |

README 亦明确写明：*"Image Format: RGB images in TIF format"*。

**为什么不可用**

| 需求 | CAS 提供 | 结论 |
|---|---|---|
| TiM 的 12 波段（含 B8A） | RGB 3 通道 | ✗ 差 9 个波段 |
| AnySat 的 10 波段（含 B8A） | RGB 3 通道 | ✗ 差 8 个波段 |
| 反射率/DN 供标准化 | 8-bit 拉伸后的显示用 RGB | ✗ 无法还原辐射量级 |
| 地理配准（回溯原始 L2A） | CRS=None，恒等仿射 | ✗ 无法定位到具体景 |
| 采集日期 | 仅区域内日期区间（README 表） | ✗ 无法逐图匹配 |

**它的适用场景**（供参考）：RGB 语义分割基线（U-Net/DeepLabV3+/MFFENet）、
无人机 0.2–1 m 高分辨率滑坡提取、灾害应急快速识别；**但不适用于本项目的多光谱双骨干**。

---

## 2. Sen12Landslides

**出处**：`huggingface.co/datasets/paulhoehn/Sen12Landslides`（CC BY 4.0），
39,556 个 NetCDF，128×128 @10 m，15 个时相。

**实测**（`chimanimani_s2_1001.nc` 等 3 个文件，从 s2_part01.tar.gz 流式抽取）：

```
变量: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12  → 10 波段（无 B01/B09/B10），int16，DN(0-10000)
      SCL (场景分类/云掩膜) | MASK {0,1} | DEM | x, y (128 各一维) | time (15)
属性: crs=EPSG:32736 | event_date=2019-03-15 | pre_post={pre:8, post:9} | annotated=True
数值: B08 中位 2702、B04 中位 358 → NDVI≈0.77，符合植被物理预期
```

对照需求：

| 需求 | Sen12 | 结论 |
|---|---|---|
| AnySat 10 波段（含 B8A） | 完全一致 | ✅ **可直接用** |
| DN 0–10000 | 是 | ✅ 与两骨干预训练统计同量级 |
| 云/阴影掩膜 | SCL（4/5/6 为可用，3/8/9/10/11 排除） | ✅ |
| 坐标系 + 日期 | EPSG + `time` + `event_date` | ✅ 可回溯原始产品 |
| TiM 的 B01/B09 | 无 | ⚠️ 需回补（见下） |
| 事件/地理划分 | 按 region 前缀（chimanimani 等） | ✅ 可做事件互斥划分 |

### 2.1 B01/B09 回补（已实现并实测）

用 AWS Earth Search STAC（公开、免账号）+ `sentinel-cogs` COG：

1. 由 `crs + center + time[post]` 算出该时相的确切日期与经纬度；
2. STAC 只检索**这一天**的 `sentinel-2-l2a`（`window_days=0`）；
3. 命中 `S2A_36KVC_20190320_0_L2A`（与样本 post 时相同日），取 `coastal`/`nir09`；
4. 60 m → 10 m 双线性重采样到样本网格。

实测结果（chimanimani_s2_1001）：

```
item=S2A_36KVC_20190320_0_L2A  cloud=49.3%
B01: mean=301   B09: mean=2795   B01/B09 nodata=3.1%
```

**若同一天没有产品，代码返回 None 并跳过该样本，绝不跨日替代。**

### 2.2 踩到的坑：Sen12 的数组是转置存储的

把 STAC 的 B08 与 Sen12 的 B08 直接比，相关系数只有 0.22；**转置后**：

| 波段 | 直接比较 | 转置后 |
|---|---|---|
| B02 | 0.027 | **0.989** |
| B03 | 0.052 | **0.986** |
| B04 | −0.085 | **0.996** |
| B08 | 0.217 | **0.974** |
| B11 | −0.185 | **0.985** |
| B12 | −0.177 | **0.993** |
| B8A | 0.277 | **0.972** |

即 Sen12 的 S2 数组按 **(x, y)** 索引存储，与其 netCDF 的 x/y 维定义相反。
影像与掩膜内部是自洽的（同一存储约定），但**回补外部波段时必须转置**，
否则 12 波段输入里 B01/B09 与其余 10 个波段会错位。
该修正已写入 `read_asset_window(transpose=True)` 并注明依据。

---

## 3. 对项目状态的影响

| 项 | 之前（仅 L4S） | 现在（CAS 评估 + Sen12 接入） |
|---|---|---|
| AnySat 真实数据 | 阻塞（缺 B8A） | ✅ 真实 10 波段跑通 |
| TiM 真实数据 | 阻塞 | ✅ 同产品 B01/B09 回补后跑通 |
| 完整模型真实推理 | 阻塞 | ✅ 5 个真实样本，链路完整 |
| 完整模型真实反向 | 阻塞 | ✅ `train_full.py --source sen12` |
| 云掩膜 | 无 | ✅ SCL |
| 事件划分 | 无 | ✅ 按 region |

L4S 仍然保留作为"RGB/14 通道旁支"的历史数据（`exp/` 下的 SAM 实验），
但不再作为 V1 正式训练数据。

---

## 4. 仍需注意（不是阻塞，但会影响结论强度）

1. **B01/B09 是 60 m 插值到 10 m**，不增加真实细节；TiM 看到的高频部分仍是插值产物。
2. **Sen12 单个样本覆盖 1280 m**，对大型滑坡足够，但小滑坡像素极少（实测 27–362 px）。
3. **云量**：本次命中的产品云量 49%，SCL 已用于排除无效像素，但会显著减少有效训练面积。
4. **样本量**：当前本地只有 part01（500 个 .nc），要做 M2 稳定训练需要全量 28 个分卷（约 39 GB）。
5. **license**：Sen12 为 CC BY 4.0，可商用但需署名；CAS 为 CC BY-NC 4.0（非商用）。
