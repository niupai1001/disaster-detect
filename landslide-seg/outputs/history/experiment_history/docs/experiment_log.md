# 实验日志（M0–M1）

按时间顺序记录**实际执行**的命令、结果与失败原因。失败也保留，避免重复踩坑。

---

## 2026-10-17

### 1. 环境侦察

| 动作 | 结果 |
|---|---|
| `python -V` / `venv` | 已存在 venv `envs/default`（torch 2.11.0+cu128，用于早期 SAM 实验） |
| `torch.cuda.is_available()` | **segfault**；`nvidia-smi` 正常；直接 `ctypes` 调 `nvcuda.dll!cuInit` → access violation |
| 官方 `deviceQuery.exe` | segfault（证明与 torch 无关） |
| 40 分钟后复测 | `cuInit=0`，CUDA 恢复；RTX 5090 31.8 GB |

→ 结论：驱动层临时故障，非环境配置问题（`blockers.md` B4）。

### 2. 数据核查

- Landslide4Sense 全量在本地：train 3799+3799、validation 245+245、test 800+800。
- 14 通道语义经官方数据卡 + 相关性结构 + 量级确认：
  `B01…B12 + SLOPE + DEM`，**无 B8A**。
- 严格模式构造 TiM/AnySat 输入均抛 `MissingBandError(['B8A'])`。
- 存储值量级不可解释（所有波段中位数 1.14–1.62，NDVI≈0）→ 不可反解为 DN。

→ **判定：现有数据无法进入正式训练。** 见 `data_contract.md`、`blockers.md` B1/B2。

### 3. 依赖与权重获取

| 目标 | 过程 | 结果 |
|---|---|---|
| terratorch | 直接 `pip install terratorch` 15 分钟无进展（依赖解析/源码构建） | 改为 `--no-deps`，再补 backbone 路径所需依赖 |
| GitHub 代码 | raw/codeload 超时 | `ghfast.top` 拉 zip 成功 |
| TerraMind 权重 | `hf-mirror.com` | 504 MB，350 个键，可严格加载 |
| AnySat 权重 | `hf-mirror.com/g-astruc/AnySat` | 504 MB，`load_state_dict` 无 missing/unexpected |
| RemoteCLIP | `hf-mirror.com/chendelong/RemoteCLIP` | 605 MB，`clip.load` 成功 |

### 4. TerraMind 接入中的 3 个真实故障

1. `import terratorch` → `No module named 'torchgeo'`
   （registry → smp_registry → datasets → torchgeo 链）
   → 写 `_terramind_loader.py`：占位包 + 空注册表桩，直接按文件加载官方模块。
2. `ValueError: Shape for encoder_embeddings.tok_lulc@224.pos_emb ([1,196,384]) does not match ([1,64,384])`
   → TiM 的 tokenizer 位置编码固定为 14×14，输入必须是 224×224。
   适配器改为内部重采样到 224，并记录 `resampled_from`。
3. `hf_hub_download` 返回的路径不存在
   → Windows 上只写 `blobs/` 未建 `snapshots/` 链接；`ensure_hf_cache` 增加 blob 复制兜底。

首个成功前向：
```
[TiM ON ] tokens=[(1,196,384) ×4] grid=(14,14) latency=0.43s
[TiM OFF] latency=0.01s
[TiM check] mean|Δ|=0.4665  relative=78.75%
```

### 5. AnySat 接入中的 2 个故障

1. `ModuleNotFoundError: No module named 'src'`
   → 官方 `hubconf.py` 内部 `from src.models...`，需要把**仓库根目录**也加入 `sys.path`
   （`torch.hub.load` 会自动做，本地按文件加载必须手动加）。
2. `RuntimeError: shape '[163840, 1]' is invalid for input of size 16384`
   → 我自己写的 `normalize_anysat` 用 `(C,1,1,1)` 广播，把 `[B,10,H,W]` 撑成了 `[10,10,H,W]`。
   改为按 `ndim` 构造 `[1,…,C,…,1]`。**这是本项目自己代码的 bug，与 AnySat 无关。**

成功输出（三尺度）：
```
[scale  40m] patches=1024 dense=(1,128,128,1536) latency=0.35s
[scale  80m] patches= 256 dense=(1,128,128,1536) latency=0.14s
[scale 160m] patches=  64 dense=(1,128,128,1536) latency=0.10s
```
（官方 dense 输出是 NHWC，适配器转成 NCHW 后送入卷积。）

### 6. 完整模型打通的 4 个故障

1. `ImportError: 需要 openai CLIP` → 安装 `clip-anytorch`。
2. `AttributeError: 'CLIP' object has no attribute 'dtype'`
   → CLIP 的 `dtype` 是从 `self.visual.dtype` 读的属性，**不能**把 visual 置空（原本想省内存）。
3. `KeyError: module name can't contain "."` → `ModuleDict` 键名做 `.`/`@` → `_` 映射。
4. `RuntimeError: Input type (cuda) and weight type (cpu)`
   → 融合投影原本在 `forward` 里懒建，`model.to(device)` 之后新建的层留在 CPU。
   改为在 `__init__` 用骨干官方维度预建（TerraMind 384、AnySat 2×768），forward 只做校验。

另外两处损失/融合逻辑修正：
- `presence` 目标 `sum(dim=(1,2))` 少了通道维 → 形状 `[256,1]`，改为按 `ndim` 求和；
- 分支丢弃有极小概率把所有分支都丢掉 → 增加"至少保留一路"的兜底。

### 7. M1 闭环结果

```
[params] total=526.0M trainable=3.5M
[step 1] total=1.4502  fine=0.8362 coarse=0.8127 presence=0.6977 region_text=0.6808 lat=0.26s
[step 2] total=1.3896  fine=0.8177 coarse=0.8030 presence=0.6947 region_text=0.3145 lat=0.24s
[step 3] total=1.3433  fine=0.7945 coarse=0.7946 presence=0.6910 region_text=0.1335 lat=0.22s
[update check] 8/8 个被抽样参数发生了更新
[frozen check] terramind: trainable=0 | anysat: trainable=0
[ckpt] outputs/checkpoints/m1_smoke.pt (41.1 MB)
[reload check] max|Δlogits| = 0.000e+00（eval + 固定种子）
[predict] prob=(2,1,128,128) presence=[0.465, 0.465]
```

**输入是合成张量**，这些 loss 不代表任何精度。

### 8. 测试

```
python -m pytest tests/ -q   →   10 passed
```
覆盖：波段表与缺失检测、严格/合成策略、重排正确性、金字塔几何、脉冲对齐、
粗/细头形状、细化器确实使用预测粗 mask、语义模块必须加载真实文本权重。

---

## 尚未完成（明确未验证）

- 真实滑坡数据上的任何前向/反向/指标 → 被 B1 阻塞；
- TiM ON/OFF 在真实数据上的贡献、AnySat 多尺度选择、语义三阶段各自增益 → 属 M3 消融；
- LoRA / 局部解冻、混合精度、多卡 → 未尝试；
- 大图（>224）滑窗推理、对象级（连通域）评估 → 未实现。

---

## 2026-10-17 晚：数据源评审与真实数据接入

### 1. CAS 滑坡数据集（中科院成都山地所）——不可用

- 出处：Zenodo `10.5281/zenodo.10294997`（CC BY-NC 4.0），20,865 张 512×512。
- 因单包 24 MB–2 GB，改用 **HTTP Range + ZIP 中央目录**只取每个子集首张影像：
  `remotezip` 远程读 `Tiburon(Sentinel)`、`Moxitaidi(SAT)`、`LongxiRiver(SAT)`、`Wenchuan`。
- 实测四个子集全部：**3 通道、uint8 0–255、CRS=None、恒等仿射**。
  README 亦自述 "RGB images in TIF format"。
- 结论：缺 9/8 个波段，且无法地理定位 → 不能用于 TiM/AnySat，**否决**。

### 2. Sen12Landslides——可用

- 从 `s2_part01.tar.gz` 用 HTTP Range 拉前 40 MB，流式解出 3 个 .nc 先验证，再下整卷（1.4 GB，503 个样本）。
- 实测：`B02…B12`（**含 B8A**，int16 DN 0–10000）+ `SCL` + `MASK` + `DEM` + `x/y` + `time`；
  attrs 有 `crs=EPSG:32736`、`event_date`、`pre_post`。
- AnySat 真实 10 波段前向成功（40m/160m，输出 finite）。

### 3. B01/B09 回补

- AWS Earth Search STAC（公开免账号）按 crs+center+time 反算经纬度与日期，**只查同一天**；
  命中 `S2A_36KVC_20190320_0_L2A`，取 `coastal`/`nir09` COG，60m→10m 双线性重投影。
- 若同一天无产品则返回 None 并跳过，不做跨日替代。

### 4. 发现并修正的坐标坑

- STAC 的 B08 与 Sen12 的 B08 直接比只有 0.22 相关；**转置后 0.972–0.996**（7 个波段一致）。
- 结论：Sen12 的 S2 数组按 (x, y) 索引存储，与 netCDF 的 x/y 维定义相反。
  影像与掩膜内部自洽，但**外部回补必须转置**，已写入 `read_asset_window(transpose=True)`。
- 附带修正：`sen12.py` 里 `center_lat/center_lon` 实为 northing/easting，命名与含义相反。

### 5. 真实数据跑通

```
[sen12] 构造 6 个真实批次（12 波段含 B01/B09 回补）
[step 1/5] loss=1.4510 → [step 5/5] loss=1.3531   (0.13 s/step)
[ckpt] outputs/checkpoints/last.pt
```

另修复一个真实场景 bug：区域池化后特征为 `[B,D,1,1]`，batch=1 时 `BatchNorm2d` 直接报错
（`Expected more than 1 value per channel`）→ `vis_proj` 改用 `GroupNorm(32, D)`。

### 6. 仍未做

- 全量 28 个分卷（≈39 GB）未下载；当前仅 503 个样本、一个事件（chimanimani）；
- 训练指标无意义（5 步、未收敛），M2 才评估；
- 跨事件/跨区域泛化未验证。

---

## 2026-10-17 深夜：可视化复查（由用户提问触发）

### 1. 发现图注错误（自己代码的 bug）

`scripts/real_inference.py` 调用 `save_panel(rgb, gt, prob, pmask)`，
而 `save_panel` 的签名是 `(rgb, gt, coarse, fine)` —— 于是：

- 标着 **"coarse mask"** 的面板，实际画的是**细阶段概率图** `prob`（连续值）；
- 标着 **"fine mask"** 的面板，实际是**阈值化后的二值结果**。

已修正为 6 面板：`RGB | SCL 云/阴影 | GT | coarse prob | fine prob | fine mask`，
并在 `smoke_full_model.py` 同步更新调用。

### 2. 网格状纹理的来源

| 量 | 数值范围 | 说明 |
|---|---|---|
| coarse prob | 0.476 – 0.551 | 动态范围仅 0.075 |
| fine prob | 0.483 – 0.485 | 几乎是常数 |

旧图用 `imshow` 默认自动归一化，把 0.076 的区间拉伸到整幅灰度，
**微弱纹理被放大成肉眼可见的"网格"**。

FFT 分析（chimanimani_s2_1003）：

```
coarse 水平: 16px(1.00) > 5px(0.75) > 43px > 64px > 8px(0.51)
coarse 垂直: 16px(1.00) > 43px > 4px(0.41) > 8px(0.33)
P0     水平/垂直: 16px, 4px 均显著
```

**16 px = 160 m**，正是 AnySat `patch_size=160m` 分支的 token 网格
（每个 token 覆盖 16×16 像素，dense 特征在块内恒等）；4 px = 40 m 分支。
即：网格是 AnySat 多尺度 dense 特征的分块结构经融合后残留的周期性，
在未训练、门控均匀（各 1/3）时最明显。

处理：概率图改为**固定 vmin=0 / vmax=1** 显示，并在标题写出真实取值范围。
修正后同一张图几乎看不出网格 —— 说明之前看到的主要是显示放大效应。

### 3. 云是数据本身就有的

chimanimani_s2_1003 的 post 时相（t=9, 2019-03-20）：

```
SCL: vegetation 84.8% | bare 3.0% | water 1.7% | cloud_med 2.4% | cloud_high 4.4%
     | cloud_shadow 1.7% | dark 0.1% | unclassified 2.0%
云+阴影 8.5%   →   valid 91.5%
```

光谱佐证（真实云的特征，不是渲染假象）：

| 波段 | 云区均值 DN | 非云区均值 DN | 比值 |
|---|---|---|---|
| B02 | 2383 | 393 | **6.06** |
| B04 | 2260 | 491 | **4.60** |
| B08 | 3304 | 2619 | 1.26 |
| B11 | 2833 | 1792 | 1.58 |

可见光区亮度是背景的 4.6–6 倍、近红外仅 1.26 倍 —— 典型云光谱。
RGB 用 2–98% 分位拉伸，云被推到 255 饱和，所以**视觉上比 8.5% 看起来更多**。

### 4. 顺带发现的一个数据质量问题

该样本的 `MASK` 在 15 个时相上**完全相同**（每个时相都是 242 px）：

```
mask 各时相像素数: [242]*15
```

即 Sen12 的掩膜是**一次事件滑坡编目**，被复制到所有时相，而不是逐期标注。
影响：
- 用 post 时相评估是对的（灾后才有滑坡），但**绝不能把 pre 时相当作负样本**；
- 也不能做"变化检测"式的逐期标签；
- 我们目前只用 post 时相，符合这一约束，但需要在 M2 的采样策略里显式禁止 pre 时相进负样本池。

### 5. 未改动的结论

模型仍是未训练状态（fine prob ≈ 0.484 常数），IoU≈0 没有意义；
本次只修正**展示方式**与**记录事实**，没有为了让图"好看"去改模型。
