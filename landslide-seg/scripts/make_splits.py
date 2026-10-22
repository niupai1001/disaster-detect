"""划分训练/验证/测试：事件互斥 + 空间块互斥。

策略（在 4 个事件上）：
- **test**：整段事件互斥（默认 italy 全部留给测试）——衡量"换一个事件"的泛化；
- **val**  ：另一个事件互斥（默认 hiroshima）——用于选模型，不参与训练；
- **train**：其余事件 + 按 5 km 空间块切出一部分做 val 的内部替代（可选）。

空间块互斥的动机：相邻 1.28 km 图块高度重叠，随机切分会让同一片滑坡同时出现在
训练和验证里。这里用 ``floor(center_x / block)`` 与 ``floor(center_y / block)``
把样本归到块，块整体只属于一个 split。

用法：
  python scripts/make_splits.py --test-event italy --val-event hiroshima
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def block_key(rec: dict, size_m: float) -> tuple[int, int]:
    cx, cy = rec["center_xy"]
    return int(cx // size_m), int(cy // size_m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests" / "certified.jsonl"))
    ap.add_argument("--test-event", default="italy")
    ap.add_argument("--val-event", default="hiroshima")
    ap.add_argument("--block-m", type=float, default=5000.0)
    ap.add_argument("--val-block-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("清单为空")

    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)

    for r in rows:
        if r["event_id"] == args.test_event:
            r["split"] = "test"
        elif r["event_id"] == args.val_event:
            r["split"] = "val"

    # 剩余事件按空间块切一小部分给 val
    rest = [r for r in rows if r.get("split") is None]
    blocks = defaultdict(list)
    for r in rest:
        blocks[(r["event_id"], *block_key(r, args.block_m))].append(r)

    import numpy as np

    rng = np.random.default_rng(args.seed)
    keys = sorted(blocks)
    rng.shuffle(keys)
    n_val_blocks = int(round(len(keys) * args.val_block_frac))
    val_keys = set(keys[:n_val_blocks])
    for k in keys:
        split = "val" if k in val_keys else "train"
        for r in blocks[k]:
            r["split"] = split

    Path(args.manifest).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    stats = Counter()
    per_event = defaultdict(Counter)
    for r in rows:
        stats[r["split"]] += 1
        per_event[r["event_id"]][r["split"]] += 1
    print("[splits]", dict(stats))
    for ev, c in sorted(per_event.items()):
        print(f"  {ev:16s} {dict(c)}")

    # 泄漏检查：同一空间块是否跨 split（只在同事件内检查）
    leak = 0
    seen = defaultdict(set)
    for r in rows:
        k = (r["event_id"], *block_key(r, args.block_m))
        seen[k].add(r["split"])
    for k, s in seen.items():
        if len(s) > 1:
            leak += 1
    print(f"[check] 跨 split 的空间块: {leak}")
    print("[check] 事件是否跨 split:",
          {ev: sorted({r['split'] for r in rs}) for ev, rs in by_event.items()})


if __name__ == "__main__":
    main()
