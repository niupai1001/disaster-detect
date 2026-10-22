"""Read-only official GitHub source snapshots, never import downloaded code."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v4r2_methods/sources"
REPOS = {
    "NVlabs/SegFormer": ("master", ["README.md", "mmseg/models/decode_heads/segformer_head.py"]),
    "zhu-xlab/DOFA": ("master", ["README.md", "dofa_v1.py", "wave_dynamic_layer.py"]),
    "HRNet/HRNet-Semantic-Segmentation": ("master", ["README.md", "lib/models/seg_hrnet.py"]),
    "JunchuanYu/TransLandSeg": ("main", ["README.md", "translandseg.yaml", "models/sam.py"]),
    "VMarsocci/pangaea-bench": ("main", ["README.md"]),
}


def read(url):
    request = urllib.request.Request(url, headers={"User-Agent": "landslide-research-source-audit"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def capture(item):
    repo, (branch, paths) = item
    result = {"repo": repo, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "branch": branch, "commit": None, "files": []}
    try:
        result["commit"] = json.loads(read(f"https://api.github.com/repos/{repo}/commits/{branch}"))["sha"]
    except Exception as exc:
        result["commit_error"] = str(exc)
    revision = result["commit"] or branch
    for name in paths:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{name}"
        file = {"path": name, "url": url}
        try:
            data = read(url)
            destination = OUT / repo.replace("/", "__") / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != data:
                raise FileExistsError("Refuse source snapshot overwrite")
            destination.write_bytes(data)
            file.update(local_path=destination.relative_to(ROOT).as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as exc:
            file["error"] = str(exc)
        result["files"].append(file)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-name", default="source_index.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        sources = list(pool.map(capture, REPOS.items()))
    destination = OUT.parent / args.index_name
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(sources, stream, indent=1)
    print(json.dumps(sources, indent=1))
