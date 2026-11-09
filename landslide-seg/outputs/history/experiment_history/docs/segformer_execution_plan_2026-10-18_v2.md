# SegFormer 执行计划 v2（2026-10-18，依据当前实测状态）

> **定位**：取代 [`segformer_execution_plan_2026-10-18.md`](segformer_execution_plan_2026-10-18.md) 的"起点 / 顺序 / 预算"三部分。
> v1 的模型契约（10 通道适配、Spectral Stem、四组对照、指标定义）继续有效；本文件只改动**已被实测推翻**或**已经落地**的部分。
>
> **编写时间**：2026-10-18 07:50（America/New_York）。
> **证据原则**：§1–§4 的每个数字都来自本轮实际命令（§12 列出），不引用未复测的历史结论；未验证的内容一律标注"未验证"。
>
> **当前状态一句话**：P0 数据流水线已经跑过一次并**整体失败**（0 通过 / 5,000 隔离），根因是一处属性解析缺陷；模型封装已落地并实测可前向反向；训练器、入口脚本、测试、扩展指标全部缺失。
>
> ---
>
> **2026-10-18 执行结果（本计划已完成）**
>
> | 阶段 | 结果 |
> |---|---|
> | P0a 修复 + 重跑 | **2,107 certified**（train 1,277 / val 448 / test 382），隔离 2,893；`stats.json` 非零；bbox 泄漏 0 |
> | P0b 标签审计 | train 12,827 个对象；52.3% 截断；1–4 px 735 个；尺度档位由对象级分布确认 |
> | P0c 环境/权重锁定 | `model.safetensors` SHA-256 与 HF LFS 元数据一致；revision 已记录；许可证仍未关闭 |
> | P1 工程 | 训练器 + 指标 + 5 个入口 + 38 个 SegFormer 测试全部落地；全套 48 tests 通过 |
> | P2 过拟合 | 8 瓦片 500 步：loss −89.3%，平均 IoU 0.812，8/8 瓦片有正预测 |
> | P3/P4 先导 | A0 30 epoch val IoU 0.239；A1/B0/B1 短跑无异常 |
> | P5 正式 | A0/A1/B0/B1 × 3 seeds 全部完成；val IoU 0.257 / 0.294 / 0.274 / 0.313 |
> | P6 评测 | val/test 全量评测、聚合、尺度分解；报告 [`segformer_report_2026-10-18.md`](segformer_report_2026-10-18.md) |
> | P7 B2 | 768 维解码器结构接线 + 校验已测（随机权重）；官方 B2 权重未下载、许可证未确认，**未完成** |
>
> 核心结论：**Stem 是稳定主效应（+0.036/+0.039 val IoU）**，适配器是次效应（+0.017/+0.020）；
> **test（italy）全部组塌到 0.015–0.023**，尺度构成把参考上限压到 0.34–0.43，
> 但同档召回下降 0.31–0.38 才是主因——域差异，不是单纯小目标。
> 下面的"当前状态/下一步"保留为当时的决策记录，不再是最新状态。

> **2026-10-18 审阅修正（metric v2 + 多日期审计）**
>
> - 指标口径升级为 **metric v2**（`evaluate_tile_v2`）：先遮蔽无效区再建预测连通域、尺度像素召回独立统计、tiny 分母修正；4 个回归用例通过；12 个 best 已重评（val/test/train 的 `eval_v2/` 与 `ablation_eval_metric_v2.json`），像素指标差异 ≤1e-6、四组排名不变。
> - 多日期契约已单列审阅：**215** 个训练样本 `event_date` 含多个日期，**120** 个所选影像早于最晚日期，**114** 个 post 索引早于首个日期；清单 `manifests/s2_10band_v1_multidate_review.csv`（未改标签、未删样本）。
> - 阶段一剩余项：`s2_10band_v2_clean`、对象—日期映射核实、可靠负例与域覆盖（本次未做）。

> - **2026-10-18 T1/T2 与扩量**：T1（首层 lr=6e-4）3-seed 均值 0.2525 vs 基线 0.2483（逐 seed 混合，不成立）；T2（均衡采样）0.2560（2/3 seed 为正，弱正向，未达改协议门槛）。数据已下载至 **part20（10,000 个 nc）**，生成 `s2_10band_v3_p20_raw`（4,772）与 `s2_10band_v3_p20`（4,723；train 1,480 / val 448 / test 2,795），四组 3-seed 训练进行中。

> - **2026-10-18 裁决与 v2_clean**：215 个多日期样本按"选图晚于最晚事件 / 120 天内可重选 / 否则隔离"裁决（95 保留、80 重选、40 隔离），生成 `s2_10band_v2_clean`（2,067；train 1,237 / val 448 / test 382，val/test 与 v1 完全一致）。seed1234 筛查 ΔIoU 在 −0.011~+0.012，方向不一致，属单 seed 噪声量级；完整 3-seed 清洁基线运行中。阶段二扩量（官方其余 S2 分卷）尚未开始。



---

## 0. 一页结论

| # | 结论 | 依据 |
|---|---|---|
| 1 | **P0 失败不是数据问题**：2,161 个样本走完全部筛选后在 `float(np.ravel(str))` 处抛异常，全部被计为隔离；清单 0 行、`stats.json` 全 0 | §3 |
| 2 | **本地标注量是旧路线的 13 倍**：5,000 个 nc 中 2,217 个带标注；按修复后口径预计 **≈2,030** 可用（旧路线 156） | §2.3 |
| 3 | **模型封装已可用**：四组变体均能加载官方权重（缺失/多余/形状不符键全为空），`[2,10,128,128] → [2,1,128,128]`，Stem 初始 ΔL 严格为 0 | §2.4 |
| 4 | **训练侧全部缺失**：训练器、5 个入口脚本、SegFormer 测试、尺度/对象/边界指标、pyproject 依赖均未落地 | §1、§5 |
| 5 | **计算预算比 v1 估计小一个量级**：batch 64 仅 4.8 GB、42.8 ms/step；100 epoch ≈ 7 分钟（不预载） | §2.5、§9 |
| 6 | **两条数据口径必须现在定**：SCL 255 必须视为无效（131 个瓦片）；时相不能只用"最低云量"（与数据集给的 post 索引 1,805/2,217 不一致） | §4 D2/D3、§5 |

**下一步只做一件事**：按 §4 修 D1–D4 并重跑 P0，拿到第一份有效清单；在此之前写训练代码属于空转（没有可用 `stats.json`，也没有可用样本列表）。

---

## 1. 阶段状态总表

| 阶段 | v1 计划要求 | 实测现状 | 判定 |
|---|---|---|---|
| P0 数据 | 10 波段清单、缓存、划分、统计 | 脚本已写（351 行），运行一次，**0 通过 / 5,000 隔离**；缓存残留 2,161 个 npz（524.6 MB）；`stats.json` 均值/方差全 0 | **BLOCKED** |
| P0 依赖 | 安装并锁定 Transformers、获取官方权重 | `envs/tm` 中 transformers **4.57.6**、torch 2.11.0+cu128；B0 ADE 权重已落盘（15.0 MB） | **DONE**（缺版本锁、revision 记录、许可证结论） |
| P1 A0 工程 | 官方封装、10 通道迁移、二分类、统一损失 | `models/segformer.py`、`models/spectral_stem.py`、`data/segformer_dataset.py`、`config.py` SegFormer schema、5 个 YAML 已落地；本轮实测前向/反向/梯度/初始化等价通过 | **部分完成**（无测试、无入口、`backbone` 键未接线） |
| P2 A0 学习验证 | 2–8 瓦片过拟合 → 核定清单先导训练 | 未开始 | 未开始 |
| P3 A1 | Stem + 融合 | 未开始（代码已有） | 未开始 |
| P4 B0 | 可训练 10→3 投影 | 未开始（代码已有） | 未开始 |
| P5 B1 | B0 + Stem | 未开始 | 未开始 |
| P6 正式对照 | 四组 × 三 seed、完整指标 | 未开始 | 未开始 |
| P7 B2 兼容 | 配置切换 + 权重迁移 | `model.backbone` 配置项**没有任何代码读取**，填 `b2` 等于仍用 B0 | 未开始 |

旧路线（TerraMind + AnySat + 语义 + 粗/细）按 v1 §9 继续保留，其历史指标（val IoU 0.2344 / test IoU 0.0177）与新路线**不可比**：输入波段、标准化、模型、损失都不同，且旧结果属 `offset=0` 口径。

---

## 2. 本轮实测证据

### 2.1 环境

| 项 | 值 | 备注 |
|---|---|---|
| 解释器 | `C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe` | Python 3.13.14 |
| 系统默认 `python` | 3.13.15，**无 torch / h5py / yaml** | 直接 `python scripts/...` 会失败，必须用专用 venv |
| torch | 2.11.0+cu128，`cuda.is_available()=True` | 2 × RTX 5090，各 31.8 GiB |
| transformers | 4.57.6 | `output_loading_info`、`SegformerForSemanticSegmentation` 均可用 |
| 其它 | safetensors 0.8.0、h5py、numpy、yaml、matplotlib、pytest 有；**scikit-image 没有** | 边界/连通域指标只能用 `scipy.ndimage` 或手写 |

### 2.2 权重

| 项 | 值 |
|---|---|
| 目录 | `weights/segformer/segformer-b0-finetuned-ade-512-512/` |
| 文件 | `config.json`、`model.safetensors`（15,036,944 B）、`pytorch_model.bin`、`tf_model.h5`、`preprocessor_config.json`、`README.md` |
| HF revision | `489d5cd81a0b59fab9b7ea758d3548ebe99677da`（来自 `.cache/huggingface/download/*.metadata`） |
| `model.safetensors` 的 LFS SHA-256 | `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`（元数据记录值，需再自算一次全文件哈希） |
| 结构 | `num_channels=3`、`hidden_sizes=[32,64,160,256]`、`decoder_hidden_size=256`、`depths=[2,2,2,2]`、150 类 ADE |
| 许可证 | 模型卡写 `license: other` → 指向 NVlabs SegFormer LICENSE，**结论未确认** |
| B2 权重 | 本地**没有**，P7 需要单独下载 |

### 2.3 数据（`data/sen12/*_s2_*.nc`）

实测 5,000 个文件（v1 文档写 5,003），7 个区域前缀；每个文件 `[15,128,128]`、10 波段 int16（`0–10000`）、`SCL`、`MASK`、`DEM`、`time[15]`。

| 区域 | 文件数 | 带标注 | 整块正像素合计 | 整块正像素 <50 的瓦片 | 掩膜触边界瓦片 | 所选时相距事件 >365 天 |
|---|---:|---:|---:|---:|---:|---:|
| chimanimani | 1,133 | 423 | 167,477 | 74 | 275 | 0 |
| china | 159 | 81 | 16,523 | 16 | 47 | 26 |
| dominicamaria | 1,148 | 403 | 219,155 | 54 | 331 | 0 |
| hiroshima | 864 | 459 | 130,709 | 135 | 305 | 0 |
| hokkaido | 290 | 150 | 337,491 | 6 | 143 | 0 |
| indonesia | 573 | 300 | 177,403 | 45 | 219 | 35 |
| italy | 833 | 401 | 19,953 | 275 | 116 | 0 |
| **合计** | **5,000** | **2,217** | **1,068,711** | **605** | **1,436** | **61** |

其它已核实事实：

| 事实 | 数值 | 影响 |
|---|---|---|
| 带标注中的负瓦片（掩膜全 0） | **2**（都在 hiroshima） | 瓦片级误报率只能写 N/A，不能编造 |
| 掩膜是否逐期相同 | 抽查 51 个标注瓦片，15 期**完全一致**；取值只有 `{0,1}` | 标签是事件级编目，不是逐期观测 |
| 瓦片中心是否重复 | 2,217 个标注瓦片中心**全部唯一** | 无重复采样；但相邻瓦片可能共享同一滑坡 |
| SCL=255 的瓦片 | 131（chimanimani 109、indonesia 22） | 255 不是标准 SCL 类，必须按无效处理 |
| 平均有效比（允许/禁止 255） | 0.9718 / 0.9127 | 131 个瓦片受影响，主要在 chimanimani |
| 按"cloud≤0.5 且 valid≥0.5"估算可用量 | **2,033**（train 1,173 / val 459 / test 401） | 以修复后重跑为准 |
| 标注里给的 `pre_post.post` 与"最低云量"选择一致 | 174 / 2,217（7.8%）；有 `pre_post` 的 1,979 个中 8.8% | 现行时相规则基本没在用标注参考日 |
| `pre_post.post` 处的云量中位数 | hiroshima 1.00、hokkaido 1.00、dominicamaria 1.00、italy 0.54、chimanimani 0.03 | 给的是"事件后第一景"，经常全云；`post+1` 才转晴 |

> 说明：`MASK` 逐期相同，说明标注来自外部编目并按瓦片投影，不是在这 15 期影像上勾的。因此"选哪一期"改变的是**影像证据**，不改变标签本身；但选到灾后 1–3 年，滑坡痕迹可能已恢复，属于标签噪声来源。

### 2.4 模型封装实测（本轮直接运行）

| 变体 | 总参数 | 编码器 | 解码器 | 新增层 | 权重加载 |
|---|---:|---:|---:|---:|---|
| A0 direct10, stem=false | 3,725,377 | 3,330,368 | 395,009 | 0 ※ | 缺失/多余/形状不符键均为空 |
| A1 direct10, stem=true | 3,753,602 | 3,330,368 | 395,009 | 28,225 | 同上 |
| B0 project10to3, stem=false | 3,714,434 | 3,319,392 | 395,009 | 33 | 同上 |
| B1 project10to3, stem=true | 3,742,659 | 3,319,392 | 395,009 | 28,258 | 同上 |

※ A0 的"新增层 = 0"是因为替换后的首层仍在 `model.segformer.` 前缀下，被 `param_groups()` 计入编码器；A0 与 B0 的编码器差值 10,976 = 10×32×7×7 − 3×32×7×7。

| 检查 | 结果 | 说明 |
|---|---|---|
| 输入/输出 | `[2,10,128,128] → [2,1,128,128]` | 四组全部通过；HF 内部已上采样，封装的 `interpolate` 是恒等 |
| 初始化等价（direct10） | 解码特征 `max|Δ| = 2.6e-06`，**非逐位相等** | 10 通道卷积与 3 通道卷积的 FP 累加顺序不同；权重拷贝本身严格相等 |
| 初始化等价（project10to3） | `max|Δ| = 0`，逐位相等 | 1×1 选择器完全等价 |
| Stem 初始输出 | `ΔL` 严格为 0，`logits == base_logits` | 末层零初始化生效 |
| 梯度通路（Stem） | 第 1 步：`fusion.out` 有梯度，`fusion.body`/`stem` 为 0；第 2 步起全部非零 | 与 v1 §4 的预期一致，不是断路 |
| 首层新增通道梯度 | 非零（direct10 第 1 步 `max|g| = 6.6e-3`） | 零初始化通道确实在学习 |

### 2.5 吞吐与显存（实测，fp32，project10to3+stem）

| batch | 训练 ms/step | 吞吐 img/s | 峰值 allocated | 备注 |
|---:|---:|---:|---:|---|
| 8 | 24.5 | 327 | 642 MiB | v1 的初始候选，明显偏小 |
| 32 | 28.4 | 1,127 | 2,420 MiB | |
| 64 | 42.8 | 1,495 | 4,795 MiB | **推荐**：速度与显存平衡 |
| 128 | 78.1 | 1,638 | 9,528 MiB | 可跑，增益仅 +10% |
| 64（bf16 autocast） | 42.0 | 1,524 | 4,878 MiB | 几乎无加速，先不用 |
| 128（bf16 autocast） | 68.3 | 1,874 | 9,394 MiB | 仅在确定要省时间时考虑 |
| 评估前向，bs=64 | 14.3 | 4,491 | — | |
| npz 读取 | — | 2.13 ms/样本 | — | ≈2,033 样本 4.3 s/epoch（单进程） |

推算：train 1,173 + val 459，batch 64 → 19 + 8 步；纯计算约 0.9 s/epoch，读盘 3.5 s/epoch。**不预载 100 epoch ≈ 7 分钟；预载入内存（约 0.73 GB int16）后 ≈ 1.5 分钟。**

---

## 3. P0 失败复盘（必须先看完再动手）

**事实链**（时间 06:22–06:25，同一次运行）：

1. `manifests/s2_10band_v1.jsonl` = **0 行**；`s2_10band_v1_summary.json` 中 `n_certified=0`、`n_quarantined=5000`、三个 split 全空、`stats.mean/std` 全 0；
2. `manifests/s2_10band_v1_quarantine.jsonl` = 5,000 行，构成：
   - 2,783 `unannotated`（未标注瓦片，按规范隔离，**正确**）；
   - **2,161 `EXC ValueError: could not convert string to float: np.str_('[480640.]')`**；
   - 50 `valid ratio < 50%`、3 `cloud > 50%`、3 `滑坡像素不可观测 > 60%`（这三类判定合理）；
3. `data/s2_10band_v1/` 残留 **2,161 个 npz（524,647,149 B）**，说明这 2,161 个样本已经走完筛选、写完缓存，才在最后一步崩掉。

**根因**：`scripts/prepare_segformer_data.py` L59 用 `_decode` 把**所有** nc 属性转成 `str`，而 `center_lon/center_lat` 实际是长度 1 的数值数组（`array([480640.])`）。到 L157：

```python
cx = float(np.ravel(cx)[0]) if cx is not None else None
```

`np.ravel('[480640.]')` 得到 `array(['[480640.]'])`，`float()` 抛 `ValueError`。异常发生在 `np.savez_compressed` 之后，因此留下缓存却没有记录。

**影响**：
- 目前**没有可用的新路线数据版本**（`s2_10band_v1`）；`stats.json` 全 0，若直接训练会得到 `(x-0)/1e-6` 的荒谬输入；
- 残留 npz 与新清单不同步，必须先清理再重跑（否则会混入已被拒样本的缓存）；
- 这次失败没有产生任何"数据不可用"的证据，不能据此下调对数据量的预期。

**修复方式（最小改动，二选一）**：

```python
# 方案 A（推荐）：加一个数值解析器，保留 _decode 的字符串语义
def _num(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    if isinstance(v, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v)
        return float(m.group()) if m else None
    return float(np.ravel(v)[0])

cx = _num(attrs.get("center_lon"))
cy = _num(attrs.get("center_lat"))
```

```python
# 方案 B：只对数值型属性不解码
attrs = {k: (v if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) else _decode(v))
         for k, v in f.attrs.items()}
```

方案 A 不改变其它字段的字符串语义，改动面最小。

**重跑前必须做的三件事**：
1. 把失败的 `s2_10band_v1_summary.json` 与 `s2_10band_v1_quarantine.jsonl` 复制到 `outputs/segformer/_failed_runs/2026-10-18_prep/` 作为证据；
2. 删除 `data/s2_10band_v1/*.npz`（确认绝对路径是 `N:\Users\yzh\Disater\landslide-seg\data\s2_10band_v1` 后再删）；
3. 用 `--limit 30` 先跑一次冒烟，确认清单有行、`stats` 非 0，再跑全量。

> **旧路线数据未被波及，仍然可用**：`manifests/certified.jsonl` 156 行（train 80 / val 36 / test 40，7 个事件）、`data/certified/` 缓存（每个 npz 内同时存 `x12`、`x10`、`mask`、`valid`、`doy`）、`outputs_run2/` 的 checkpoint 与 `metrics.json` 全部完好；旧模型可原样复跑。
> 新路线要重建清单，不是因为旧数据坏了，而是因为：① 旧认证只保留 156 / 2,217 个标注瓦片（其余 2,061 个被"同日回补 B01/B09 + 跨源配准 ≥0.90"两道门槛挡掉，而 SegFormer 根本不需要这两个波段）；② 旧缓存 `valid` 被回补空洞支配（实测 0.969，而 10 波段自身可达 1.000，差的 3.1% 正是 B01/B09 空洞）；③ 时相规则与逐波段标准化统计都要按新口径重算。
> 因此 **P1/P2 可以不等 P0**：直接取旧缓存里的 `x10`（80 个训练瓦片）做训练链路联调与过拟合诊断，标注为 bootstrap、不作为正式结果。

---

## 4. 缺陷清单与修复顺序

严重度：**P0-blocking** 不修则无法产出数据；**P1** 影响结论有效性；**P2** 工程/可复现性。

| ID | 级别 | 缺陷 | 证据 | 改法 | 验证 |
|---|---|---|---|---|---|
| D1 | blocking | `center_lon/lat` 解析崩溃 | 5,000 条隔离中 2,161 条为该异常 | §3 方案 A | `--limit 30` 有清单、`stats.mean` 非 0 |
| D2 | P1 | `SCL_INVALID` 未含 255 | 131 个瓦片含 255；有效比 0.9718→0.9127 | 用**允许清单** `scl_ok = np.isin(scl, (4,5,6,7))`，不要用排除清单 | 单测：255 像素不进 valid |
| D3 | P1 | 时相只按"最低云量" | 与 `pre_post.post` 一致仅 174/2,217；有 `pre_post` 的 1,979 个中仅 174 个一致；灾后天数 p90 = 193 天，最大 1,050 天 | 规则改为：事件后 120 天窗口内**最早**满足 `cloud≤0.5 且 valid≥0.5` 的时相；窗口内无解则退化为全灾后最低云量并标 `temporal_fallback`；同时记录 `pre_post` 与差值 | 清单统计：`fallback` 数、天数分布 |
| D4 | P1 | `stats.json` 全 0 + 残留缓存 | 见 §3 | 清理后重跑；`compute_stats` 增加 p1/p50/p99 与 `n_pixels` | `stats.mean` 与手算一致 |
| D5 | P1 | `model.backbone` 配置无人读取 | `SegFormerSeg.__init__` 只接收 `pretrained_dir` | 增加 `backbone` 参数与映射表（b0→256、b2→768 维解码器），校验 `config.decoder_hidden_size` 与声明一致，不一致直接报错 | 单测：`b2` 在无 B2 权重时报明确错误，而不是静默用 B0 |
| D6 | P2 | `label_known = mask >= 0` 实际恒真 | `MASK` 为 uint8 且只有 {0,1}；若将来出现 -1 会被转成 255 并整块隔离 | 显式处理未知标签：`mask==255 → 未知`，valid 置 False，不参与损失，不当作负样本 | 单测：构造 255 掩膜样本 |
| D7 | P1 | 无 SegFormer 测试 | `tests/` 只有旧路线 2 个文件 | 新增 §6 P1a 的 6 类测试 | pytest 全绿 |
| D8 | P1 | 对象/尺度/边界指标未实现 | 配置里有 `scale_thresholds_ha` 等键，无代码 | 按 §8 实现（`scipy.ndimage`，不依赖 skimage） | 合成掩膜单测（已知 IoU/边界） |
| D9 | P2 | `pyproject.toml` 无 transformers | 依赖表里没有 | 加 `segformer = ["transformers==4.57.6", "safetensors>=0.4"]` 可选组，并输出 `pip freeze` 片段 | `pip check` |
| D10 | P2 | batch=8 过小 | bs=8 仅 327 img/s、642 MiB | 公共 batch 改为 64（§7） | 已实测 |
| D11 | P2 | 权重身份只做"首尾块哈希" | `config.file_identity` 截断 32 字符 | 对 `model.safetensors` 计算**完整 SHA-256**并写入加载报告；记录 revision 与许可证结论 | 报告与元数据一致 |
| D12 | P2 | `config.num_channels` 仍为 3 | 替换首层后未同步 | `adapt_first_conv_direct10` 内 `model.config.num_channels = in_chans` | 保存配置后重新读取为 10 |
| D13 | P2 | 每 epoch 重复解压 npz | 2.13 ms/样本 | 加 `data.preload: true`（缓存 int16 张量 0.73 GB）或 `num_workers>0`（Windows 需 main guard） | 吞吐对比 |
| D14 | P1 | "事件"其实是区域前缀 | china 有 37 个 `event_date`、indonesia 约 60 个 | 报告层把 `(region, event_date)` 作为事件单元；划分仍按 7 个区域互斥，并声明 china/indonesia 是"多事件区域组" | 汇总表按 `(region, event_date)` 输出 |

---

## 5. 必须在训练前冻结的口径

| # | 决策 | 建议值 | 理由 / 敏感性 |
|---|---|---|---|
| 1 | 数据版本名 | `s2_10band_v1`（第一次**有效**版本） | 失败运行未产出有效清单，不必升版本；但 summary 里要写 `rules_sha256`，规则再变就升 v2 |
| 2 | `valid10` | `SCL ∈ {4,5,6,7}` ∧ 全部波段有限且 >0 ∧ 标签已知 | 255 不是 SCL 类；实测影响 131 瓦片 |
| 3 | 时相 | 灾后 120 天内最早满足 `cloud≤0.5 且 valid≥0.5`；否则退回全灾后最低云量并标 `fallback` | 现行规则 p90 滞后 193 天；120 天是可申报的物理窗口，需在清单里报告实际分布 |
| 4 | 划分 | train = {chimanimani, china, dominicamaria, hokkaido, indonesia}；val = hiroshima；test = italy | 与旧路线同一事件口径，便于对照；val/test 各为单一事件，泄漏风险最低 |
| 5 | 标准化 | 训练集有效像素逐波段 z-score，val/test 复用 | 加回 +1000 DN 常数在 z-score 中被均值吸收，**新路线不再使用 `boa_offset_dn`**；物理指数（NDVI 等）若要用必须单独加回 |
| 6 | 主阈值 | 0.5 | 阈值扫描只在验证集做，单列报告 |
| 7 | 尺度档 | small <50 px(0.5 ha)、medium 50–200 px、large ≥200 px | **按对象面积**而不是按整块像素统计；档位在只看训练集分布后冻结 |
| 8 | 对象召回 | 8 邻接连通域，一对一贪心匹配，IoU≥0.25 主、≥0.5 严格 | 截断对象单列（§8.3） |
| 9 | 边界 F1 | 容差 1 px 主、2 px 补充；排除距无效区/瓦片边缘不足容差的部分 | 实测 64.8% 瓦片有掩膜触边，不排除会把瓦片边缘当滑坡边界 |
| 10 | 负瓦片误报率 | **N/A**（只有 2 个负瓦片），另报背景像素 FPR | 2 个瓦片算不出比率；写 N/A 比给一个伪数字诚实 |
| 11 | 预算单位 | 用"优化器更新步数"而非 epoch | v1 的 100 epoch × batch 8 = 14,700 步；batch 64 时 100 epoch 只有 1,900 步，不能直接照搬 |
| 12 | 精度 | 主结果 fp32；bf16 只作对照 | 实测 bf16 在 bs=64 只快 2% |
| 13 | 统计 | 3 seeds（1234/2026/3407）均值±std + 逐事件 + 逐瓦片；瓦片级 bootstrap 置信区间 | 像素不独立，不能给像素级显著性 |

**尚未定、需要在 A0 pilot 后冻结**：batch（候选 64）、epochs/总步数（候选 150 epoch ≈ 2,850 步）、是否启用 bf16。冻结后四组不得单独延长。

---

## 6. 分阶段执行计划

依赖顺序：P0a → P0b → P0c → P1 → P2 → P3 → P4 → P5 → P6 → P7。P3–P5 可并行写代码，但**训练必须按 A0→A1→B0→B1 顺序**，以便前一组暴露的问题不污染后一组。

### P0a 修复与重跑（0.5 天）

- 修 D1/D2/D3/D4；
- 清理残留 npz，先 `--limit 30` 冒烟，再全量（5,000 文件，8 线程，预计 10–20 分钟）；
- 产物：`manifests/s2_10band_v1.jsonl`、`_quarantine.jsonl`、`_summary.json`、`data/s2_10band_v1/*.npz`、`stats.json`。
- **验收**：`n_certified` 在 1,900–2,150 之间（与 §2.3 估算同量级）；三个 split 均非空；每 split 的 `n_events/n_positive_tiles/positive_px/small_objects` 有值；`split_bbox_leaks=0`；`stats.mean` 落在 100–4,000 DN、`std` 非 0；清单每行都有 `cache` 且文件存在。

### P0b 数据身份与标签审计（0.5 天）

- summary 增加：`rules_sha256`（口径文本哈希）、`script_sha256`（脚本哈希）、`manifest_sha256`、`stats_sha256`、每 split 的 `(region, event_date)` 计数、`fallback` 计数、灾后天数分布、`SCL=255` 计数；
- 新增 `scripts/audit_segformer_labels.py`：对 **train split** 做 GT 连通域，输出**对象级**面积分布（当前只有瓦片级数字）、截断对象比例、1–4 px 对象数；
- 产物：`docs/segformer_input_contract.md`（波段顺序、valid10、时相规则、划分、统计文件哈希、权重 revision）、标签审计 JSON。
- **验收**：尺度档位由**对象级**分布支撑；截断对象比例与 §2.3 的 64.8% 同量级；输入契约文档与代码常量一致。

### P0c 环境与权重锁定（0.5 天）

- 计算 `model.safetensors` 完整 SHA-256；记录 revision、许可证结论（NVlabs LICENSE 的商用限制必须写清）；
- 记录 `--set` 与 YAML 的最终生效配置；`pip freeze` 子集写进 `docs/segformer_environment_lock.md`；
- pyproject 增加 `segformer` 可选依赖（D9）。
- **验收**：加载报告里 revision/hash/license 三项齐全；别人按文档能复现同一权重。

### P1 工程验收（1 天）

| 测试 | 内容 | 通过标准 |
|---|---|---|
| `test_segformer_adapter.py` | 四组变体构造；direct10 权重拷贝逐元素相等；解码特征与"三通道模型 + 对应 RGB"对比 | 权重相等；特征 `allclose(atol=1e-5)`；project 路径逐位相等 |
| `test_segformer_model.py` | 前向尺寸、Stem 开关、初始 ΔL=0、首层新通道梯度、两步后 Stem 梯度、`state_dict` 保存到全新实例重载 | 尺寸 `[B,1,H,W]`；ΔL=0；重载后 logits `max|Δ|<1e-6` |
| `test_segformer_data.py` | 标准化只按训练统计；增强同步；关闭增强时三条路径一致；无效像素不进损失 | 逐元素一致；无效像素计数为 0 |
| `test_segformer_metrics.py` | 全无效样本、空预测、单像素对象、边界容差 | 与手算一致 |
| `tests/test_segformer_config.py` | `_base` 合并、`--set`、`backbone` 校验 | 覆盖生效、b2 报错清晰 |

- 入口 `scripts/smoke_segformer.py`：真实 batch 前向 + 反向 + 显存/延迟打印，写 `outputs/segformer/_smoke/<exp>/report.json`。
- **验收**：pytest 全绿；smoke 报告含参数量、峰值显存、延迟、加载键清单。

### P2 过拟合诊断（0.5 天）

- 选 2/4/8 个可靠训练瓦片（覆盖大/中/小目标），关闭增强，**最多 500 步**；
- 指标：loss、训练瓦片 IoU/Dice、预测正像素数、图像对照（概率图/GT）。
- **验收**：loss 显著下降（≥50%）、每个瓦片都出现正像素、预测不退化为常数；记录实际 Dice，**不以固定 Dice 作为放行门槛**（v1 §6 的约束保留）。未达标先查数据/损失/优化，不自动扩正式训练。

### P3 A0 先导 + profiler（0.5 天）

- A0 用核定清单跑短预算（如 30 epoch），同时测 batch 8/32/64/128 的显存与步时；
- 冻结 batch、epochs/总步数、精度、数据加载方式；
- **验收**：A0 的 loss 曲线与验证 IoU 有明确改善；冻结值与实测一致；把决定写进 `docs/segformer_training_protocol.md`。

### P4 A1/B0/B1 先导（0.5 天）

- 同预算短跑，检查：开关一致性、投影非 RGB 系数是否更新、Stem 梯度是否更新、显存是否仍在同一量级；
- **验收**：三组都能完成短跑且每组的"特有检查"通过；若某组异常，先修再进 P5。

### P5 正式四组 × 三 seed（0.5 天，机时约 1–2 小时）

- 公共设置冻结后：4 组 × 3 seeds = 12 次；每组从**同一份预训练权重**重新开始，A1/B1 不加载 A0/B0 的已训练权重；
- 每 epoch 验证，按验证前景 IoU 存 best，另存 last；测试集不参与任何选择；
- 两个 GPU 各跑一个进程（`CUDA_VISIBLE_DEVICES`），记录实际设备；配对比较尽量同卡，若跨卡需先验证可复现性。
- **验收**：12 个 run 目录齐全；每个 run 有 `resolved_config`、数据/划分/统计哈希、权重身份、seed、best/last、曲线、逐 epoch 指标。

### P6 评测与报告（0.5 天）

- 按 §8 输出全部指标，逐事件、逐瓦片 CSV、聚合混淆计数；
- 固定案例集，输出 RGB/假彩色/GT/valid/概率/预测/TP-FP-FN；
- 效率表（参数量、B=1 延迟 p50/p95、峰值显存、训练吞吐）；
- **验收**：所有指标要么有数、要么明确 N/A 并给理由；无法计算的项不得用 0 或估计值代替。

### P7 B2 兼容（0.5 天）

- 先修 D5（`backbone` 接线 + 维度校验），再下载 B2 ADE 权重并记录来源/revision/hash；
- 至少完成：B2 四组前向/反向/保存重载测试；
- B2 四组完整训练属后续扩展，不在本轮承诺。

### 可选 P8（时间允许再做）

留一事件交叉验证（LOEO）：只对 A0 与 B0、单 seed 跑 7 折，检验"适配器效应"是否跨事件稳定。7×2 = 14 次短跑，约 1 小时。

---

## 7. 四组公共训练协议（按实测修正）

| 项 | v1 计划 | v2 修正 | 理由 |
|---|---|---|---|
| batch | 8 | **64**（实测 4.8 GB） | bs=8 只有 327 img/s；32 GB 卡浪费 |
| epochs | 100 | **150**（≈2,850 步），pilot 后冻结 | v1 的 100 epoch 是配合 batch 8 的 14,700 步；直接搬到 batch 64 会少 7.7 倍更新 |
| 数据加载 | `num_workers=0` | 预载入内存（0.73 GB）或 4 workers | npz 解压 2.13 ms/样本，占每 epoch 大头 |
| 精度 | 先 fp32 | 保持 fp32；bf16 仅作对照 | 实测 bs=64 时 bf16 只快 2% |
| lr | 编码器 6e-5 / 新层与解码器 6e-4 | 不变（pilot 中验证） | 未训练前不改 |
| 调度 | 5% warmup + polynomial | 不变 | |
| 损失 | `0.5*masked BCE + 0.5*masked Dice` | 直接复用 `training/losses.py::bce_loss/dice_loss`（已有，且已按 valid 归一） | 不复制实现 |
| 全无效样本 | 跳过并计数 | 不变；batch 内若全无效，跳过该 batch 并记数 | |
| 增强 | 翻转 + 90° 旋转 | 不变；影像/mask/valid 同步 | |
| 选择 | 按 val 前景 IoU 存 best | 不变；测试集完全不参与 | |

**参数分组**：`SegFormerSeg.param_groups()` 已实现"编码器 / 解码器 / 新增层"三分组；新增层（首层新增通道、投影、Stem、融合头）统一用 head lr。四组共用同一规则。

**复现要求**：每次 run 保存 `resolved_config`、`manifest_sha256`、`stats_sha256`、权重 revision+完整 SHA-256、加载缺失/新增/改形状键、软件版本、seed、`best.pt`/`last.pt`、逐 epoch 曲线、预测。全新进程重载 best 后评估结果需在声明容差内复现（建议 `1e-5`）。resume 必须恢复优化器/调度器/AMP/RNG 状态。

---

## 8. 评测实现规格（P6 前必须写好）

所有实现放在 `src/landslide_seg/training/` 或新建 `evaluation/`，**只依赖 numpy/scipy**（skimage 未安装）。

### 8.1 像素指标
复用 `training/metrics.py`（已修正"无效像素不计 TN"）。输出聚合混淆计数、IoU/Dice/Precision/Recall、逐瓦片 CSV、逐事件汇总。

### 8.2 尺度分组
- 对 GT（只取 valid ∧ 已知区域）做 8 邻接连通域；
- 面积换算：1 px = 100 m²；small <50 px、medium 50–200 px、large ≥200 px；
- 分别报告对象数与**对象内像素召回**；1–4 px 对象单列计数与召回；
- 档位在只看训练集对象分布后冻结（P0b 产物），四组一致。

### 8.3 小目标 / 对象召回
- GT 与预测各做连通域，一对一匹配（按 IoU 从大到小贪心，或 Hungarian）；
- 主报告：IoU≥0.25 的对象召回；补充：IoU≥0.5；
- **截断对象**：对象像素触及瓦片边界，或与 `valid=False` 像素相邻（≤1 px）→ 单独成层；
  - 主指标用"未截断对象"的分母；
  - 同时给出"全部对象"和"截断对象"两组数字，禁止混算。

### 8.4 边界 F1
- 边界 = 掩膜 XOR 其 1 px 腐蚀；
- 排除 GT/预测边界中距无效区或瓦片外边缘 <容差 的部分；
- 主容差 1 px（10 m），补充 2 px；
- 空边界规则：GT 与预测都为空 → 不计入；只一方为空 → 记 FN/FP，不记 0 分。

### 8.5 负瓦片误报率
- 可靠负瓦片只有 **2** 个（hiroshima，均在 val）→ **N/A**，报告"n=2，不足以计算比率"；
- 另报背景像素 FPR（有效区内 `FP/(FP+TN)`），并注明它与瓦片级误报不是同一指标；
- 任何"最小对象面积后处理"必须四组一致并单列。

### 8.6 效率
- 总/可训练参数；
- B=1、128×128：预热 20 次、计时 100 次，报延迟 p50/p95 与吞吐；
- 峰值 allocated/reserved；训练峰值按公共 batch 报；
- 单 GPU、同精度；模型计算与读盘分开计时。

### 8.7 案例
固定集合：小/中/大目标、低对比、纯背景裸地、云边界；保存 RGB、假彩色（B12/B8A/B04）、GT、valid、概率、预测、TP/FP/FN。按验证集规则选漏检/误检；测试案例只在实验冻结后分析。

### 8.8 比较口径
- 输入适配效应：A0 vs B0、A1 vs B1；
- Stem 效应：A1−A0、B1−B0；
- 交互项：两者差值之差；
- 必须同时看 precision/recall，防止"扩大预测范围"换来小目标召回虚高。

---

## 9. 预算与时间表

| 工作 | 机时 | 人力/工程 |
|---|---|---|
| P0a 修复 + 全量重跑 | 10–20 min | 0.5 天 |
| P0b 身份与标签审计 | 分钟级 | 0.5 天 |
| P0c 环境/权重锁定 | 分钟级 | 0.5 天 |
| P1 测试与 smoke | 分钟级 | 1 天 |
| P2 过拟合（≤500 步 × 若干组） | <10 min | 0.5 天 |
| P3 A0 pilot + profiler | ~30 min | 0.5 天 |
| P4 三组先导 | ~1 h | 0.5 天 |
| P5 正式 12 run（150 epoch） | **1–2 h**（预载后 <40 min） | 0.5 天 |
| P6 评测与报告 | ~30 min | 0.5 天 |
| P7 B2 兼容 | 取决于下载 | 0.5 天 |

> 单次 run 估算：train 1,173 样本 × 150 epoch ÷ 64 ≈ 2,850 步 × 42.8 ms ≈ 2 分钟计算；读盘 3.5 s/epoch ≈ 9 分钟。合计约 11 分钟；预载后约 3 分钟。

---

## 10. 风险与"不能主张"清单

| # | 风险 | 现状 | 处理 |
|---|---|---|---|
| R1 | 负瓦片只有 2 个 | 都在 hiroshima（val） | 瓦片级误报率写 N/A；需要外部"确认无滑坡"数据源才能补 |
| R2 | 64.8% 的正瓦片掩膜触边界 | 实测 | 对象召回/边界指标必须分层；截断对象单列 |
| R3 | 标签来自事件编目、逐期相同 | 抽查 51/51 一致 | 选图只影响证据不影响标签；但灾后过久会引入"痕迹已恢复"噪声，靠 120 天窗口抑制 |
| R4 | 数据集给的 post 时相经常全云 | hiroshima/hokkaido/dominicamaria 中位云量 1.0 | 时相规则改为"最早可用"，并记录 `fallback` |
| R5 | test 只有单一事件（italy） | 401 瓦片 | 泛化结论只能写"跨到 italy"；要强结论需 LOEO 或新增外部事件 |
| R6 | italy 已被旧路线多轮查看 | 历史事实 | 记为"历史固定留出集"，不称盲测；仍固定该划分做配对比较 |
| R7 | ADE 预训练看到的不是自然 RGB | z-score 后的 DN | 明确不声称"卫星影像等同自然图像"；RGB 拷贝只用于保留初始化响应 |
| R8 | china/indonesia 是"多事件区域组" | 37 / ~60 个 event_date | 报告按 `(region, event_date)` 分解；划分仍区域互斥 |
| R9 | 旧指标与新路线不可比 | 输入/模型/损失全变 | 不引用旧数字做对比基线 |
| R10 | 项目没有 git 仓库 | `git status` 报 not a repository | 用配置/脚本/清单哈希做身份；建议后续 `git init`（低优先） |
| R11 | B2 权重与许可证 | 未下载 / 未确认 | P7 前完成；许可证结论写进 P0c 报告 |

**明确不能主张**：任何精度/提升结论（尚未训练）；"Stem 有效"（只有结构）；"泛化好"（单事件测试）；"空图拒检可用"（2 个负瓦片）；"辐射已完全正确"（只解决了常数偏移，场景分布差异仍在）。

---

## 11. 命令清单（P0a 之后可用）

```powershell
$PY = "C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe"

# P0a 冒烟 → 全量
& $PY scripts/prepare_segformer_data.py --config configs/segformer/base.yaml --limit 30
& $PY scripts/prepare_segformer_data.py --config configs/segformer/base.yaml --workers 8

# P0b 标签审计
& $PY scripts/audit_segformer_labels.py --split train

# P1 测试与冒烟
& $PY -m pytest tests/test_segformer_*.py -q
& $PY scripts/smoke_segformer.py --config configs/segformer/a0.yaml

# P2 过拟合
& $PY scripts/overfit_segformer.py --config configs/segformer/a0.yaml --tiles 8 --max-steps 500

# P3/P4 先导
& $PY scripts/train_segformer.py --config configs/segformer/a0.yaml --set segformer.train.epochs=30

# P5 正式
& $PY scripts/run_segformer_ablation.py --order A0 A1 B0 B1 --seeds 1234 2026 3407

# P6 评测
& $PY scripts/evaluate_segformer.py --checkpoint outputs/segformer/s2_10band_v1/A0/1234/best.pt --split test
```

> 除 `prepare_segformer_data.py` 外，以上脚本**尚不存在**（§1），属于 P1–P6 的实现项。

---

## 12. 本轮核对方式（可复核）

| 核对项 | 方式 |
|---|---|
| 目录/文件清单 | `Get-ChildItem -Recurse -File` |
| 依赖与版本 | 专用 venv `python -c "import ...; print(__version__)"` |
| 权重 revision | `weights/segformer/.../.cache/huggingface/download/*.metadata` 首行 |
| nc 结构/属性 | `h5py` 读 attrs、`time`、`SCL`、`MASK` |
| 标注量与分布 | 遍历 5,000 个 nc 的 `annotated` / `event_date` |
| 可用量估算 | 遍历 2,217 个标注瓦片的 `SCL` + `MASK`，按 `cloud≤0.5 & valid≥0.5` 统计 |
| 时相规则对比 | 用 `pre_post_dates.post` 与"灾后最低云量"索引逐瓦片对比 |
| 模型封装 | `SegFormerSeg` 四变体加载 + 前向 + 反向 + 梯度检查 |
| 初始化等价 | 与三通道 `SegformerForSemanticSegmentation` 的解码特征逐元素比较 |
| 吞吐/显存 | 预热 5 步、计时 10 步，`torch.cuda.max_memory_allocated` |
| 失败复盘 | 解析 `s2_10band_v1_quarantine.jsonl` 的 5,000 条 reason |

---

## 13. 与 v1 计划的差异

| v1 内容 | v2 处理 | 原因 |
|---|---|---|
| 状态"PLAN，尚未实施" | 改为"P0 已试跑并失败；P1 部分落地" | 实测 |
| 数据起点"156 条认证清单" | 改为"2,217 标注、预计 ≈2,030 可用" | 10 波段路线不需要 B01/B09 回补，旧认证的 C1/C5 门槛不再适用 |
| "优先检查本地可用样本，再决定下载" | 本地 5,000 已足够第一轮，不必再下载 | 2,030 ≫ 第一轮所需 |
| batch=8、100 epoch | batch=64、150 epoch（按更新步数定义预算） | 实测吞吐 |
| SCL 有效判定 | 改为允许清单 `{4,5,6,7}` | 255 混入 |
| 时相"最低云量" | 改为"灾后 120 天内最早可用，失败再退化" | 有 `pre_post` 的 1,979 个瓦片中 91% 与标注参考日不一致 |
| "无可靠负瓦片则 N/A" | 保留，且明确 n=2 | 实测 |
| 尺度档按瓦片像素 | 改为按**对象**面积 | 瓦片级数字会误导 |
| B2 配置支持 | 标记为"配置键未接线"，需先修 D5 | 实测 |
| 权重"官方获取是显式验收项" | 已完成下载；补 revision/hash/license | 实测 |
| `boa_offset_dn` | 新路线不使用（z-score 吸收常数），保留旧路线开关 | 数学等价 |

> - **2026-10-18 v3 结果**：扩充到 parts 1–20（10,000 nc）后，val 排序变为 **A1 0.300 > B1 0.293 > B0 0.253 > A0 0.246**；Stem 仍稳定（A +0.054、B +0.040），适配器效应变为 ±0.007 且随 Stem 变号 → "project10to3 更好"不成立；扩量不是均匀增益（A0/B0/B1 下降，A1 上升）。test 集合扩到 2,795 张，与 v1/v2 不可比。
