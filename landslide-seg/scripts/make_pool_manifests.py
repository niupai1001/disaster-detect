"""从 v4 清单派生 M2 需要的训练/评测清单。

训练器只按 ``split`` 字段取数据（train/val/test），因此这里把 ``split_v4`` 的角色
"翻译"成派生清单，不改动原始 manifest，也不改训练代码：

    manifests/v4/pool_25.jsonl ... pool_100.jsonl
        train = 该档嵌套池（9 个事件）
        val   = hiroshima（SEEN_DEV，与 v1–v3 可比，用于早停/选 checkpoint）
        test  = newzealand（FINAL，训练全程不可见）

    manifests/v4/dev_thrissur.jsonl   # 第二个开发事件，单独评测
    manifests/v4/diag_italy.jsonl     # 固定诊断集，不参与任何选择

用法::

    python scripts/make_pool_manifests.py --version s2_10band_v4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FRACTIONS = (25, 50, 75, 100)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(records: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="s2_10band_v4")
    ap.add_argument("--out-dir", default=str(ROOT / "manifests" / "v4"))
    args = ap.parse_args()

    rows = {r["sample_id"]: r for r in read_jsonl(ROOT / "manifests" / f"{args.version}.jsonl")}
    split = json.loads((ROOT / "manifests" / "split_v4.json").read_text(encoding="utf-8"))
    design = split["design"]

    pool: dict[str, set[str]] = {k: set() for k in FRACTIONS}
    with open(ROOT / "manifests" / "allowed_train_pool.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in FRACTIONS:
                if int(r[f"in_{k}"]):
                    pool[k].add(r["sample_id"])

    val_ids = [sid for sid, r in rows.items() if r["event_id"] in design["dev"] and r["event_id"] == "hiroshima"]
    final_ids = [sid for sid, r in rows.items() if r["event_id"] in design["final"]]

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Do not overwrite existing manifests; use a new revision directory")
    summary = {}
    for k in FRACTIONS:
        recs = []
        for sid in sorted(pool[k]):
            r = dict(rows[sid])
            r["split"] = "train"
            r["pool_fraction"] = k
            recs.append(r)
        for sid in val_ids:
            r = dict(rows[sid])
            r["split"] = "val"
            recs.append(r)
        for sid in final_ids:
            r = dict(rows[sid])
            r["split"] = "test"
            recs.append(r)
        p = out_dir / f"pool_{k}.jsonl"
        write_jsonl(recs, p)
        ev = Counter(r["event_id"] for r in recs if r["split"] == "train")
        summary[f"pool_{k}"] = {
            "path": str(p).replace("\\", "/"),
            "train": len(pool[k]),
            "val": len(val_ids),
            "test": len(final_ids),
            "train_events": len(ev),
            "per_event": dict(sorted(ev.items())),
        }
        print(f"[pool] {k:>3}% → train {len(pool[k]):>5} | val {len(val_ids)} | test {len(final_ids)} | events {len(ev)}")

    # 额外两组评测清单：只保留一个事件，全部标成 test，供 evaluate_segformer.py --manifest 使用
    for name, event in (("dev_thrissur", "thrissur"), ("diag_italy", "italy")):
        recs = []
        for sid, r in rows.items():
            if r["event_id"] == event:
                rr = dict(r)
                rr["split"] = "test"
                recs.append(rr)
        p = out_dir / f"{name}.jsonl"
        write_jsonl(recs, p)
        summary[name] = {"path": str(p).replace("\\", "/"), "test": len(recs)}
        print(f"[eval] {name:>12}: {len(recs):>5} tiles")

    (out_dir / "pool_summary.json").write_text(
        json.dumps(
            {
                "data_version": args.version,
                "split_sha256": split.get("sha256"),
                "split_frozen": split.get("frozen"),
                "pools": summary,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"[done] → {out_dir}")


if __name__ == "__main__":
    main()
