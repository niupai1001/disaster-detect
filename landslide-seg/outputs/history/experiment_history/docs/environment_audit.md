# 环境审计

> 实测时间：2026-10-17，全部数值来自 `scripts/audit_environment.py` 的实际输出
> （原始 JSON：`outputs/logs/environment_audit.json`）。

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| OS | Windows 11 (10.0.26200) |
| GPU | NVIDIA GeForce RTX 5090 ×2，各 32607 MiB（31.8 GB 可用） |
| 驱动 | 595.97（CUDA 13.2） |
| 磁盘 | N: 7.45 TB 总 / 2.67 TB 可用 |

## 2. Python 与关键包

| 项 | 值 |
|---|---|
| Python | 3.13.14 |
| 解释器 | `C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe`（专用 venv） |
| torch | 2.11.0+cu128 |
| torch CUDA 构建 | 12.8，`cuda.is_available()=True` |
| terratorch | 1.2.13（已安装；**顶层 import 不可用**，见 §3） |
| timm | 1.0.30 |
| einops | 0.8.2 |
| huggingface_hub | 1.33.0 |
| clip (openai, clip-anytorch) | 已安装 |
| numpy | 2.5.2 |
| h5py | 3.16.0 |
| matplotlib | 3.11.2 |

## 3. 为什么单独建 `tm` 环境 + 为什么 terratorch 只做"部分导入"

`terratorch/__init__.py` 会连带导入 `datasets`、`datamodules`、`tasks` 与全部 backbone，
从而依赖 `torchgeo`、`lightly`、`jsonargparse`、`scikit-image` 等一整条地球观测栈。
本项目只需要 TerraMind 这一个 backbone，因此：

1. 以 `--no-deps` 安装 terratorch 本身，再单独装齐它 backbone 路径真正需要的依赖；
2. `src/landslide_seg/encoders/_terramind_loader.py` 采用两级策略：
   - 先尝试正常的 `import terratorch...`；
   - 失败时注册若干只带 `__path__` 的占位包，直接按文件加载
     `terratorch/models/backbones/terramind/**` 的**官方源码**，
     并提供空的注册表桩（官方代码用 `@REGISTRY.register` 装饰器，本项目直接调用构建函数）。

**没有复制或改写任何模型实现**，加载的就是 pip 包内的官方文件。

同一环境里同时容纳了 AnySat（官方 repo，`vendor/AnySat-main`）与 RemoteCLIP
（openai CLIP 格式 `clip.load`），因此三者可以在**同一进程**里做真正的端到端前向/反向，
不需要跨进程缓存特征。

## 4. 依赖获取路径（网络受限时的实际做法）

| 资源 | 直连 | 实际使用 |
|---|---|---|
| PyPI | 200 OK | 直连 |
| download.pytorch.org | 200 OK | 直连 |
| GitHub raw / codeload | 超时（000） | `gh-proxy.com` / `ghfast.top` |
| api.github.com | 200 OK | 直连 |
| huggingface.co | 403/超时 | `hf-mirror.com`（`HF_ENDPOINT`） |

## 5. 模型权重（全部为官方发布，非自制）

| 模型 | 文件 | 大小 | 来源 |
|---|---|---|---|
| TerraMind V1 Small + TiM | `weights/TerraMind_v1_small.pt` | 504.2 MB | `ibm-esa-geospatial/TerraMind-1.0-small` |
| ↳ S2L2A Tokenizer | HF 缓存 | — | `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A` |
| ↳ LULC Tokenizer | HF 缓存 | — | `ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC` |
| AnySat Base | `weights/AnySat_base.pth` | 503.8 MB | `g-astruc/AnySat/resolve/main/models/AnySat.pth` |
| RemoteCLIP ViT-B-32 | `weights/RemoteCLIP-ViT-B-32.pt` | 605.2 MB | `chendelong/RemoteCLIP` |

代码版本：
- TerraMind 示例仓 `IBM/terramind@main`（仅取 configs 作参考，模型代码来自 terratorch 1.2.13）
- AnySat `gastruc/AnySat@main`（`vendor/AnySat-main`；未锁定 commit，属于待办）
- terratorch 1.2.13（PyPI）

## 6. 内存/耗时实测（batch=1~2, 128×128）

| 模块 | 参数 | 前向耗时（RTX 5090） |
|---|---|---|
| TerraMind small + TiM ON | 245.35 M | 0.43 s（batch 1） |
| TerraMind small + TiM OFF | 245.35 M | 0.01 s（batch 1） |
| AnySat base @40m dense | 125.91 M | 0.21–0.35 s |
| AnySat base @80m dense | 125.91 M | 0.14 s |
| AnySat base @160m dense | 125.91 M | 0.15 s |
| 完整模型（3 分支融合） | 526.0 M（可训练 3.5 M） | 0.22–0.37 s/step（batch 2，含反向） |

## 7. 环境层面的已知问题

- 会话初期 CUDA 完全不可用（`cuInit` access violation），约 40 分钟后自行恢复，
  详见 `blockers.md` B4。若复现，优先重启机器，不要先怀疑 torch 安装。
- Windows 上 `huggingface_hub` 可能只在 `blobs/` 落盘而不建 `snapshots/` 链接，
  `ensure_hf_cache()` 已做兜底。
