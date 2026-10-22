"""下载 Sen12Landslides 的 S2 harmonized 分卷并解压（默认 11–20）。

直连 ``HF_ENDPOINT``/镜像的 resolve URL（HEAD 实测 200 + 支持 Range），
用 Range 断点续传，避免 ``hf_hub_download`` 在镜像上的 metadata 失败。

用法::

    python scripts/download_sen12_parts.py --start 11 --end 20
    python scripts/download_sen12_parts.py --start 11 --end 11 --no-extract
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests  # noqa: E402


def _tar_ok(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as tf:
            return len(tf.getnames()) > 0
    except Exception:
        return False


def _remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code == 200 and r.headers.get("content-length"):
            return int(r.headers["content-length"])
    except Exception:
        pass
    return None


def _download_resume(url: str, target: Path, attempts: int = 4) -> int:
    """Range 断点续传；返回最终大小。"""
    expected = _remote_size(url)
    for attempt in range(1, attempts + 1):
        have = target.stat().st_size if target.exists() else 0
        if expected is not None and have >= expected:
            return have
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=120) as r:
                if r.status_code == 416:  # 已完整
                    return target.stat().st_size if target.exists() else 0
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0
                mode = "ab" if have else "wb"
                last_log = time.time()
                with open(target, mode) as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        if time.time() - last_log > 20:
                            done = target.stat().st_size
                            pct = f" ({done/expected*100:.1f}%)" if expected else ""
                            print(f"    {done/2**30:.2f} GiB{pct}", flush=True)
                            last_log = time.time()
            got = target.stat().st_size
            if expected is None or got >= expected:
                return got
            print(f"[retry] {target.name} 大小 {got} < {expected}，第 {attempt} 次", flush=True)
        except Exception as exc:
            print(f"[retry] {target.name} 第 {attempt} 次失败: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"下载失败: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="paulhoehn/Sen12Landslides")
    ap.add_argument("--prefix", default="data_harmonized/s2")
    ap.add_argument("--start", type=int, default=11)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--dest", default=str(ROOT / "data" / "sen12"))
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--mirror", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    summary = {"repo": args.repo, "prefix": args.prefix, "mirror": args.mirror, "parts": []}
    started = time.time()

    for i in range(args.start, args.end + 1):
        name = f"s2_part{i:02d}.tar.gz"
        remote = f"{args.prefix}/{name}"
        url = f"{args.mirror}/datasets/{args.repo}/resolve/main/{remote}"
        target = dest / name
        entry = {"part": i, "name": name, "url": url, "status": None, "bytes": None}

        if target.is_file() and target.stat().st_size > 0 and _tar_ok(target):
            print(f"[skip] {name} 已存在且可读（{target.stat().st_size/2**30:.2f} GiB）", flush=True)
            entry.update({"status": "existing", "bytes": target.stat().st_size})
        else:
            print(f"[download] {url}", flush=True)
            t0 = time.time()
            size = _download_resume(url, target)
            dt = time.time() - t0
            entry.update({"status": "downloaded", "bytes": size, "seconds": round(dt, 1)})
            print(f"[download] {name} 完成 {size/2**30:.2f} GiB，{dt/60:.1f} min", flush=True)

        if not args.no_extract:
            t0 = time.time()
            n_new = 0
            with tarfile.open(target, "r:gz") as tf:
                for m in tf.getmembers():
                    if not m.name.endswith(".nc"):
                        continue
                    out = dest / Path(m.name).name
                    if out.exists() and out.stat().st_size > 0:
                        continue
                    with tf.extractfile(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n_new += 1
            entry["extracted_nc"] = n_new
            entry["extract_seconds"] = round(time.time() - t0, 1)
            print(f"[extract] {name}: 新增 {n_new} 个 nc", flush=True)

        summary["parts"].append(entry)
        summary["elapsed_seconds"] = round(time.time() - started, 1)
        summary["total_nc"] = len(list(dest.glob("*_s2_*.nc")))
        (dest / "download_summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    summary["total_nc"] = len(list(dest.glob("*_s2_*.nc")))
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    (dest / "download_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] 本地 nc 总数: {summary['total_nc']}")


if __name__ == "__main__":
    main()
