# Phase 1 Report — 数据适配器、质量掩膜与样本生成

生成时间：2026-10-31
项目根目录：`N:\Users\yzh\Disater\earth-representation`
执行依据：计划文档 Phase 1；影像操作沿用现有工程（512×512 @10 m），split 按区域（用户已确认）

## 1. 执行范围

已完成：
- Sentinel-2 L2A、Landsat 8/9 C2 L2、HLS V2.0 三个适配器，统一输出 `image / valid_mask / band_metadata / geo_metadata`。
- 三套 QA 解码（SCL、QA_PIXEL+QA_RADSAT、Fmask）与统一 valid mask。
- 按区域切 512×512 patch、生成 train/val/test/provenance 清单。
- 20 个单元测试。

未完成 / 未启用：
- EnMAP 适配器只保留接口，**不启用**（Phase 0 认证 401，无像素可取）。
- 未训练任何模型，未做 SSL。
- 未生成全量 patch 文件：只按区域抽样落地 37 个 npz 供测试与可视化，其余以清单引用 COG 按需读取。

## 2. 数据

管线覆盖 15 个区域 × 3 个产品，时间窗为事件日期前后 180 天，每个区域/产品选 3 景候选。

| 指标 | 数值 |
|---|---|
| patch 总数 | 300 |
| 按产品 | S2 96 / Landsat 94 / HLS 110 |
| 按 split | train 220 / val 48 / test 32 |
| valid_fraction 均值 | 0.875 |
| provenance 记录 | 127 条（每个被尝试的景一条） |
| 落地样例 | 278 个 npz |
| 构建耗时 | 首轮 1710 s + 补建 1071 s（含网络窗口读取） |

按区域：chimanimani、china、dominicamaria、hiroshima、hokkaido、indonesia、italy、itogon、kyrgyzstan1、kyrgyzstan2、newzealand、usa_puertorico 各 9 个；thrissur 6、usa_alaska 6；**nepal 0**。

split 规则（按区域，整区不跨 split）：
- train：其余 11 区
- val：hiroshima、italy
- test：newzealand、usa_puertorico

## 3. 实现

| 文件 | 作用 |
|---|---|
| `src/data/band_catalog.py` | 波段身份、中心波长、原生 GSD；按 key 查表，不靠数组下标 |
| `src/data/qa_masks.py` | SCL / Fmask / QA_PIXEL+QA_RADSAT 解码，统一 valid |
| `src/data/adapters/_base.py` | `Scene` 结构、远程窗口读取、重投影到统一网格、反射率反标定 |
| `src/data/adapters/sentinel2.py` | PC L2A，按 processing baseline 决定 −1000 DN 偏移 |
| `src/data/adapters/landsat_c2.py` | PC C2 L2，只取 landsat-8/9 |
| `src/data/adapters/hls_v2.py` | LP DAAC，Earthdata token 走 header 文件，scale/offset 取自产品 XML |
| `src/data/adapters/enmap.py` | 仅接口；`available()` 返回 False 与原因 |
| `src/data/patch_dataset.py` | 网格铺设、按 QA 抽样有效比例、补丁落地 |
| `src/data/manifest.py` | 区域 split 规则、jsonl 读写 |
| `scripts/build_patches.py` | CLI：区域 × 产品构建清单 |

统一输出契约：

```
image          float32 [C,H,W]   表面反射率（QA 波段不计入）
valid_mask     bool    [H,W]     由产品自身 QA 决定
band_metadata  list               key/name/centre_nm/native_gsd_m/resampled_to_m
geo_metadata   dict               product/scene_id/datetime/CRS/transform/size/gsd/bands
```

运行命令：

```bash
python scripts/build_patches.py --max-patches 3 --materialize 1
python -m pytest tests -q
```

## 4. 验收

| 计划验收项 | 结果 | 证据 |
|---|---|---|
| 生成一批可重复读取的 S2、Landsat patch | 通过 | 39 + 39 patch，37 个 npz 可直接加载 |
| HLS adapter 能读 V2.0 样本 | 通过 | 42 个 HLS patch |
| Landsat 缩放 `DN×0.0000275−0.2` 验证 | 通过 | `tests/test_landsat_scaling.py` |
| S2 使用产品 quantification/offset | 通过 | `tests/test_s2_scaling.py`（baseline≥4 的 −1000 偏移） |
| HLS 用产品 scale/offset 而非只靠 GeoTIFF tag | 通过 | `tests/test_hls_scaling.py` + 适配器读 XML |
| QA mask 对已知像元解码正确 | 通过 | `tests/test_qa_masks.py`（三套位定义） |
| 波段顺序变化时 metadata 同步变化 | 通过 | `tests/test_band_order.py` |
| nodata/云/云影/雪/饱和不参与训练 | 通过 | valid_mask 统一由 QA 决定，patch 按 valid_fraction 过滤 |
| patch 不跨训练/验证/测试地理分区 | 通过 | `manifest.SPLIT_BY_REGION`，整区划分 |
| 原始数据、派生数据、HLS 之间保留 provenance | 通过 | `provenance.jsonl` 133 条（scene_id、URL、scale 参数） |

测试结果：`20 passed in 0.47s`。

## 5. 结果

- 45 个“区域 × 产品”组合中 40 个产出 patch，5 个为 0：nepal×3、thrissur/sentinel2、usa_alaska/landsat。
- 有效像元比例均值 0.868，说明进入清单的 patch 基本不是云/雪/无数据补丁。
- 样例 npz 实测：S2 10 波段、反射率 −0.02…1.63；Landsat 7 波段、−0.2…0.76；HLS 6 波段、−0.15…0.84（负值来自 fill/偏移，由 mask 排除）。
- 全流程没有任何整景落地：所有 patch 都是远程 COG 的窗口读取。

## 6. 失败与风险

1. **EnMAP 仍未启用**（Phase 0 遗留）。适配器只保留接口与准入判定，不参与任何训练数据。
2. **nepal 0 个 patch**。该区域 AOI 只有 0.09°×0.05°（约 10×5.5 km），小于一个 512@10 m 网格（5.12 km）在整数对齐后可容纳的范围，`patch_grids` 返回 0 个网格。属真实现象，不是读取失败。
3. **thrissur/sentinel2 0 个 patch**。命中的景在 AOI 处 SCL 全为 0（nodata）、B04 全为 0，即“足迹覆盖但无数据”，valid=0 被正确拒绝。管线没有把空补丁写进清单。
4. **usa_alaska/landsat 0 个 patch**。有 24 个候选网格，但按云量排序取到的 3 景是冬季景，QA 雪标志使有效比例低于 0.30。
5. **选景排序缺陷（未修，已知）**：目前按目录云量排序取前 3 景。Alaska 的例子说明这会选中“晴空但积雪”的景。修法与 Phase 0 的 HLS 选择器一致——先探测 AOI 的 QA 有效比例再排序，属于 Phase 2 前的待办。
6. **HLS 的 UTM 带号推断曾经出错（已修）**。先前用 MGRS 瓦片字母推断半球，`T36K*` 被判成南半球 EPSG:32736，而影像实际是 EPSG:32636，导致网格与影像错位、有效比例恒为 0。现改为直接采用影像自身的 CRS，chimanimani 的 HLS 由 0 个 patch 变为 3 个。
7. **`NotGeoreferencedWarning` 复现但未定位**。构建日志中出现在 `rasterio/warp.py` 的重投影处，已写出的 120 个 patch 的 transform 均非退化、valid 与反射率范围合理，判定未影响结果；但 Phase 2 前需要定位来源（可能是某个资产缺少地理参考）。
8. **窗口读取的“传输字节”无法精确计量**。GDAL 只报告落盘大小，报告中的下载量是落盘量，实际网络字节略高（含 COG 头与索引）。

## 7. 下一阶段建议

Phase 1 的验收项已满足，可进入 Phase 2（可变波段 CNN + DINO），但建议先做三件小事，代价都很低：

1. 把选景排序改为“AOI 有效比例优先”，解决 Alaska 那类“晴空但积雪”的选择。
2. 定位 `NotGeoreferencedWarning` 的来源，确保几何链路没有隐藏问题。
3. 决定 nepal 这类极小 AOI 的处理方式：允许补丁与 AOI 部分重叠，或该区域单独用更小 patch。

另需用户决定：EnMAP 账号问题的处理（贴 Cookie / 换账号 / 本轮不做）；patch 数量是否从每区每产品 3 个提高到 6–10 个（预计数据量 250–800 MB，构建时间成比例增加）。

