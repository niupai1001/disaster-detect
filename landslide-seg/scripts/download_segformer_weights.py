"""下载 SegFormer 官方权重（默认 B2 ADE）并记录来源、revision 与哈希。

走 ``HF_ENDPOINT``（默认 https://hf-mirror.com），Range 断点续传，不依赖
``hf_hub_download`` 的 metadata（镜像上曾失败）。

用法::

    python scripts/download_segformer_weights.py \
        --repo nvidia/segformer-b2-finetuned-ade-512-512 \
        --dest weights/segformer/segformer-b2-finetuned-ade-512-512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests  # noqa: E402


def _head(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=30)
    if r.status_code == 200 and r.headers.get("content-length"):
        return int(r.headers["content-length"])
    return None


def _download(url: str, target: Path, attempts: int = 4) -> int:
    expected = _head(url)
    for attempt in range(1, attempts + 1):
        have = target.stat().st_size if target.exists() else 0
        if expected is not None and have >= expected:
            return have
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=180) as r:
                if r.status_code == 416:
                    return target.stat().st_size if target.exists() else 0
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0
                with open(target, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            got = target.stat().st_size
            if expected is None or got >= expected:
                return got
            print(f"[retry] {target.name} {got} < {expected}", flush=True)
        except Exception as exc:
            print(f"[retry] {target.name} {attempt}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"下载失败: {url}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--dest", default=str(ROOT / "weights" / "segformer" / "segformer-b2-finetuned-ade-512-512"))
    ap.add_argument("--files", nargs="+", default=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
    ap.add_argument("--mirror", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = requests.get(f"{args.mirror}/api/models/{args.repo}", timeout=30)
    api.raise_for_status()
    info = api.json()
    meta = {
        "repo": args.repo,
        "revision": info.get("sha"),
        "mirror": args.mirror,
        "downloaded_on": date.today().isoformat(),
        "files": {},
    }

    for name in args.files:
        url = f"{args.mirror}/{args.repo}/resolve/main/{name}"
        target = dest / name
        print(f"[download] {url}", flush=True)
        size = _download(url, target)
        meta["files"][name] = {"bytes": size, "sha256": _sha256(target)}
        print(f"  {name}: {size/2**20:.1f} MiB", flush=True)

    cfg = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    meta["config"] = {
        "model_type": cfg.get("model_type"),
        "num_labels": cfg.get("num_labels"),
        "hidden_sizes": cfg.get("hidden_sizes"),
        "decoder_hidden_size": cfg.get("decoder_hidden_size"),
        "depths": cfg.get("depths"),
    }
    (dest / "download_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
