"""环境审计：OS / Python / GPU / 关键包 / 权重 / 磁盘。

用法:
    python scripts/audit_environment.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def probe_cuda() -> dict:
    """直接调用驱动 API，把"CUDA 不可用"与"torch 装错"区分开。"""
    import ctypes

    out = {"cuInit_rc": None, "cuInit_error": None}
    try:
        lib = ctypes.WinDLL(r"C:\WINDOWS\System32\nvcuda.dll")
        rc = lib.cuInit(0)
        out["cuInit_rc"] = int(rc)
    except Exception as exc:  # access violation 表现为 OSError
        out["cuInit_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    report = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except Exception as exc:
        report["torch"] = f"IMPORT_FAIL: {exc}"

    report["driver_probe"] = probe_cuda()

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        report["nvidia_smi"] = r.stdout.strip().splitlines()
    except Exception as exc:
        report["nvidia_smi"] = f"FAIL: {exc}"

    pkgs = {}
    for name in ("terratorch", "timm", "einops", "transformers", "huggingface_hub",
                 "open_clip", "clip", "h5py", "numpy", "matplotlib"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            # 顶层 import 失败时退回读包元数据（terratorch 的 __init__ 会拉整套 EO 栈）
            try:
                from importlib.metadata import version as _v

                pkgs[name] = f"{_v(name)} (installed, top-level import unavailable)"
            except Exception:
                pkgs[name] = None
    report["packages"] = pkgs

    weights = {
        "terramind_small": ROOT / "weights" / "TerraMind_v1_small.pt",
        "anysat_base": ROOT / "weights" / "AnySat_base.pth",
        "remoteclip": ROOT / "weights" / "RemoteCLIP-ViT-B-32.pt",
    }
    report["weights"] = {}
    for k, p in weights.items():
        if p.is_file():
            report["weights"][k] = {
                "path": str(p),
                "size_mb": round(p.stat().st_size / 1024**2, 1),
            }
        else:
            report["weights"][k] = {"path": str(p), "exists": False}

    total, used, free = shutil.disk_usage(str(ROOT))
    report["disk"] = {
        "root": str(ROOT),
        "total_tb": round(total / 1024**4, 2),
        "free_tb": round(free / 1024**4, 2),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = ROOT / "outputs" / "logs" / "environment_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
