# SegFormer 环境与权重锁定

> 记录时间：2026-10-18。所有数值来自本机实测（`python -m pip list`、`nvidia-smi`、
> `scripts/smoke_segformer.py`）。任何一项变化都必须重跑四组并更新本文件。

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| OS | Windows 11（10.0.26200） |
| GPU | NVIDIA GeForce RTX 5090 ×2，各 32,607 MiB |
| 驱动 | 595.97（CUDA 13.2） |
| CUDA 运行时（torch 构建） | 12.8 |

## 2. 解释器与依赖

| 项 | 值 |
|---|---|
| 解释器 | `C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe` |
| Python | 3.13.14 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| transformers | **4.57.6** |
| safetensors | 0.8.0 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| huggingface_hub | 0.36.2 |
| timm | 1.0.30 |

> 系统默认 `python`（3.13.15）**没有** torch/h5py/yaml；所有命令必须用上面的 venv 解释器。
> `scikit-image` **未安装**，评测实现只依赖 numpy + scipy（`scipy.ndimage`）。

`pyproject.toml` 已补 `segformer = ["transformers==4.57.6", "safetensors>=0.4"]` 可选依赖，
以及 `[tool.pytest.ini_options] pythonpath = ["src"]`。

## 3. 预训练权重

| 项 | 值 |
|---|---|
| 目录 | `weights/segformer/segformer-b0-finetuned-ade-512-512/` |
| 来源 | `nvidia/segformer-b0-finetuned-ade-512-512` |
| HF revision | `489d5cd81a0b59fab9b7ea758d3548ebe99677da` |
| `model.safetensors` 大小 | 15,036,944 B |
| `model.safetensors` 完整 SHA-256 | `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9` |
| 加载方式 | `SegformerForSemanticSegmentation.from_pretrained(local_dir)` |
| 加载结果 | `missing/unexpected/mismatched/error` 四个列表均为空（见 `outputs/segformer/_smoke/*/report.json`） |

> 该 SHA-256 与 HF 元数据 `.cache/huggingface/download/model.safetensors.metadata`
> 中记录的 LFS 哈希**完全一致**，说明本机文件与仓库发布版本同一份。

### 许可证状态（未关闭）

模型卡写 `license: other`，并指向 `https://github.com/NVlabs/SegFormer/blob/master/LICENSE`。
该许可证的具体条款（尤其商用限制）**尚未逐条确认**，在对外使用/发布前必须复核。
本轮实验只用于内部研究，不据此主张任何再分发许可。

## 4. B2 权重（已获取）

| 项 | 值 |
|---|---|
| 目录 | `weights/segformer/segformer-b2-finetuned-ade-512-512/` |
| 来源 | `nvidia/segformer-b2-finetuned-ade-512-512` |
| HF revision | `de01bae28967510f9ddd496c60a969357195400c` |
| `pytorch_model.bin` 大小 | 110,000,905 B |
| `pytorch_model.bin` SHA-256 | `187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3` |
| `config.json` SHA-256 | `ee7400840fdb1e5045f0b2eba78bf053df8e33a309c4acec31705a48c8cc5c00` |
| 解码器维度 | 768（B0 为 256，`BACKBONE_DECODER_HIDDEN` 校验通过） |
| 加载结果 | `mismatched_keys` 为空；`[1,10,128,128] → [1,1,128,128]` |
| 参数量 | 无细节分支 27,369,345；+细节残差 27,430,338；+条件精分头 27,518,659 |

下载与哈希记录在 `weights/segformer/segformer-b2-finetuned-ade-512-512/download_meta.json`，
可复现命令：`python scripts/download_segformer_weights.py --repo nvidia/segformer-b2-finetuned-ade-512-512`。
许可证状态与 B0 相同（`license: other`，对外使用前需复核）。

## 5. 可复现命令

```powershell
$PY = "C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe"

& $PY scripts/prepare_segformer_data.py --config configs/segformer/base.yaml --workers 8
& $PY -m pytest tests/test_segformer_config.py tests/test_segformer_data.py tests/test_segformer_metrics.py tests/test_segformer_adapter.py tests/test_segformer_model.py -q
& $PY scripts/smoke_segformer.py --config configs/segformer/a0.yaml
& $PY scripts/run_segformer_ablation.py --order A0 A1 B0 B1 --seeds 1234 2026 3407
```
