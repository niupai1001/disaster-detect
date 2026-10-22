# 阻塞清单

本文件记录**已实际验证**的阻塞事实，不做推测性描述。每条包含：现象、证据、影响、已尝试、解除条件。

> **更新（2026-10-17 晚）**：B1 已通过**更换数据源**解除，不是靠放宽校验。
> - CAS 滑坡数据集（中科院成都山地所）实测为 RGB 三通道、无坐标系 → **不可用**；
> - Sen12Landslides 实测含 `B02–B08 / B8A / B11 / B12`（10 个，**无 B01、B09、B10**）+ SCL + 日期 + 坐标系 → **可用**；
> - TiM 缺的 B01/B09 已用同一 L2A 产品按同一天回补并跑通。
>   详见 `docs/data_source_review.md`。
>
> 因此"真实数据是否可训练"的结论已从 BLOCKED 变为 **DONE**；
> B1 对 Landslide4Sense 仍然成立（该数据集本身没有 B8A，且无法回溯）。

---

## B1. 真实滑坡数据缺 B8A，无法构造合法的双骨干输入（L4S 仍然成立；已换数据源解决）

**现象**
现有唯一滑坡数据集是 Landslide4Sense（`N:/Users/yzh/Disater/images|annotations`，14 通道）。
官方数据卡与实测都表明其 14 个通道为：
`B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,SLOPE,DEM`
——**包含 B10（卷云），不包含 B8A**。

而两条骨干的官方输入协议都要求 B8A：

| 骨干 | 要求波段 | 缺 |
|---|---|---|
| TerraMind `S2L2A`（TiM 必须完整 12 波段） | B01–B12 + **B8A**，无 B10 | B8A |
| AnySat `s2` | B02,B03,B04,B05,B06,B07,B08,**B8A**,B11,B12 | B8A |

**证据**
```bash
python scripts/audit_data_contract.py --probe 16
```
```json
"band_contract": {
  "terramind_missing": ["B8A"],
  "anysat_missing":    ["B8A"]
},
"strict_mode": {
  "terramind": "refused: ['B8A']",
  "anysat":    "refused: ['B8A']"
}
```
`python scripts/train_full.py --source real` → 退出码 **2**，打印 `[BLOCKED]`。

**影响**
- M1 验收第 1、8 项无法通过；一切"真实数据分割结果"都不能给出。
- 现有 `exp/` 下 SAM3 / RemoteSAM 的零样本实验属于另一条技术路线（RGB/PCA 输入），
  与本文档的双骨干架构不可混为一谈，也不能替代本项验收。

**已尝试（均被判定为不可接受）**
- 补零填 B8A：TerraMind 官方在 [issue #36](https://github.com/IBM/terramind/issues/36) 明确说明
  缺波段属于 OOD 输入，会显著劣化；文档 §3.1 也禁止。
- 用 B08（842nm，宽带）近似 B8A（865nm，窄带）：二者物理相关但不是同一观测，
  会让"12 波段"这一前提失真，同样禁止。
- 代码中已提供 `policy="synthetic_smoke"`，但强制在样本 provenance 里写
  `synthetic_bands={"B8A": "copied_from:B08"}` 与 `fully_observed=False`，
  仅用于验证张量形状与梯度通路。

**解除条件（二选一）**
1. 取得**含地理坐标 + 采集日期 + 产品 ID** 的滑坡标注（如 Sen12Landslides），
   按 `data_contract.md` §3 的流程从 Copernicus Data Space 下载同一时相原始 L2A，
   补齐 B8A 后重投影到 10m 网格；
2. 更换为本身就提供完整 12 波段 L2A（含 B8A）且有滑坡掩膜的数据源。

---

## B2. L4S 存储数值不是 DN/反射率，无法对齐两个骨干的预训练统计

**现象**
官方 baseline（`iarai/Landslide4Sense-2022/dataset/landslide_dataset.py`）对**已存储的数值**
再做一次逐波段仿射归一化，mean ≈ [-0.49 … 0.78]，std ≈ [0.88 … 1.61]。
实测各波段中位数集中在 1.14–1.62，连 `SLOPE`（1.43）与 `DEM`（1.20）也在同一量级，
`NDVI=(B08-B04)/(B08+B04)` 的均值约 **-0.016**（真实陆地应显著为正）。

**结论**
存储值已被数据集制作者归一化/混合处理，**无法反解为反射率或 DN**。
因此即便补齐 B8A，也无法合法套用
TerraMind `S2L2A`（mean 1390–3163, std 1334–2141）或 AnySat（DN 尺度，mean 2320–4837）
的预训练标准化。

**影响** 与 B1 叠加，使 L4S 完全无法作为双骨干的正式训练输入。

**解除条件**
改用原始 L2A 产品（DN，harmonized offset 已处理）；即与 B1 的解除条件一致。

---

## B3. 缺少地理/日期/事件标识，无法做事件互斥划分与泄漏控制

**现象** L4S 每个 `image_i.h5` 只有 `img` 键，无 CRS、bbox、日期、event id、产品 ID。

**影响**
- 无法按 `split_by: [event_id, geography]` 划分，只能沿用官方 noise-prone 的随机划分，
  相邻图块/同事件泄漏风险无法量化；
- 无法生成 `data_contract.md` 要求的完整清单字段（`scene_id/date/bbox/crs/cloud_fraction`）；
- 无法回溯原始影像（B1 的根因）。

**当前处理** `SampleMeta` 中这些字段保持 `None`，并在 `label_status`
与 `unusable_reason` 中显式标注，不做任何猜测性填充。

---

## B4. 环境问题：会话初期 CUDA 无法初始化（已自行恢复）

**现象**
会话开始时任意 CUDA 调用（`torch.cuda.is_available()`、官方 `deviceQuery.exe`、
直接 `ctypes` 调 `nvcuda.dll!cuInit`）都会触发 access violation：

```
OSError: exception: access violation writing 0x000000000000293C
```

**排查**
- `nvidia-smi` 正常（驱动 595.97，两块 RTX 5090，GPU1 空闲）；
- `nvcuda.dll` 版本 32.0.15.9597 与驱动一致；
- 最小 PATH、非沙箱执行、指定单卡均复现 → 判定为驱动层临时状态问题。

**后续**
约 40 分钟后重测，`cuInit` 返回 0，CUDA 恢复正常（`torch.cuda.is_available()=True`），
本报告所有 GPU 结果均在此恢复之后取得。

**影响** 无遗留影响，但说明该机器存在偶发 CUDA 初始化失败；
如再次出现，先重启再排查环境，不要误判为 torch 安装问题。

---

## B5. 网络：GitHub 直连不可用（已绕过）

`raw.githubusercontent.com` / `codeload.github.com` 直连超时（HTTP 000）。
可用替代（均已实测）：
- GitHub 代码：`https://ghfast.top/https://github.com/<org>/<repo>/archive/refs/heads/main.zip`
- GitHub raw：`https://gh-proxy.com/https://raw.githubusercontent.com/...`
- GitHub API：`https://api.github.com/...`（可用）
- HuggingFace：`https://hf-mirror.com`（需 `HF_ENDPOINT`）

**额外坑**：Windows 上 `huggingface_hub` 可能只把文件写进 `blobs/` 而没有在
`snapshots/<sha>/` 建链接，导致 `hf_hub_download` 返回的路径不存在。
`encoders/terramind_tim.py::ensure_hf_cache` 已做兜底（blob 复制 / 直接下载）。

---

## B6. 本数据集没有"已验证的无滑坡图块"（影响 presence 与空图拒检）

**现象**
Sen12Landslides 的属性满足严格的等价关系（400 个文件实测）：

```
annotated=False  有掩膜=False  有event_date=False : 178
annotated=True   有掩膜=True   有event_date=True  : 222
```

即 `annotated=False` 的图块**既没有掩膜也没有事件日期**——它们是"未标注"，
不是"经核实无滑坡"。把它们当负样本等于把"未知"当 0，正是规范禁止的做法。

**影响**
- 无法训练/验证 tile 级 `presence` 头与"空图拒检"；
- `lambda_presence` 已设为 0，`predict()` 的 `use_presence_gate` 默认关闭
  （否则会用随机分数清零预测）。

**仍然可用的负信号**
- **像素级**：已认证图块内的非滑坡像素（掩膜为 0 且 SCL 有效）仍作为背景监督，
  这部分是可靠的，因为该图块确实经过标注；
- 语义分支的前景/背景对照用同一图块内的 FG/BG 池化实现，也不依赖负样本图块。

**解除条件**：引入带"确认无滑坡"标注的数据源，或对未标注图块做人工抽查后转为负样本。

---

## B7. 27 个样本因跨源配准相关不足被隔离（原因未完全定位）

**现象** C5 判据（STAC B08 与 Sen12 B08 相关 ≥0.90）在 28 个隔离样本中拦下 27 个，
相关值分布 0.33–0.89；通过样本的中位数是 0.980。

**已知**
- 相关计算已确认需要转置（转置后通过样本从 0.22 升到 0.97），不是方向问题；
- 被拦下的样本多为"所选时相距事件日期较远"的情形（如 dominicamaria 事件 2017-09-23，
  所选观测 2018-02-26），季节变化会显著降低跨源一致性。

**处理** 保守隔离，不用未通过配准校验的样本。这会损失约 27% 的候选，
但不影响已认证样本的正确性。

**待查** 是否存在景/轨道差异或 bbox 偏移导致的系统性错位；建议对隔离样本做
逐波段相关 + 位移搜索，若确认是季节性差异而非几何错位，可放宽判据到分波段阈值。
