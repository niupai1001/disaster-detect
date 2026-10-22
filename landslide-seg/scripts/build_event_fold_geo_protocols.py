"""为四折事件协议生成带地形通道的冻结协议（stats 换成本折 DEM 变体）。

原 ``protocol.json`` 只登记了十波段 ``stats_train.json``；训练入口的
``validate_protocol`` 会核对"配置里的 stats 路径已登记且哈希一致"，所以每个
新增统计都必须在协议里登记并重算整体哈希。本脚本只新增协议文件，不改原协议。

产出（每折）：
- ``protocol_dem.json``     登记 ``stats_dem.json``
- ``protocol_geoboth.json`` 登记 ``stats_geoboth.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256  # noqa: E402
from landslide_seg.training.protocol import canonical_hash  # noqa: E402

FOLDS = ("F1", "F2", "F3", "F4")
VARIANTS = {
    "dem": ("stats_dem.json", "十波段 + DEM（本折 TRAIN 拟合）"),
    "geoboth": ("stats_geoboth.json", "十波段 + DEM + 坡度（本折 TRAIN 拟合）"),
}


def posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def build(fold: str, variant: str) -> Path:
    d = ROOT / "manifests" / "event_cv_v5" / fold
    base = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
    stats_name, desc = VARIANTS[variant]
    stats_path = d / stats_name
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    payload = {k: v for k, v in base.items() if k != "sha256"}
    # 把十波段 stats 的登记替换成新变体；其余文件不变
    keys = [k for k in payload["files"] if k.endswith("stats_train.json")]
    if len(keys) != 1:
        raise RuntimeError(f"{fold}: 协议里应当恰好登记一个 stats_train.json，得到 {keys}")
    payload["files"].pop(keys[0])
    payload["files"][posix(stats_path)] = file_sha256(stats_path)
    payload["version"] = f"EVENT_CV_V5/{fold}+{variant}"
    payload["design"] = dict(payload.get("design", {}), extra_channel=variant)
    payload["notes"] = dict(
        payload.get("notes", {}),
        stats=f"十波段沿用本折 stats_train.json；{desc}",
        terrain="DEM 为静态底图，坡度由 DEM 按 10 m 像元求梯度，不含事件信息",
    )
    payload["sha256"] = canonical_hash(payload)
    out = d / f"protocol_{variant}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="all", choices=("all", *FOLDS))
    ap.add_argument("--variant", default="all", choices=("all", *VARIANTS))
    args = ap.parse_args()
    folds = FOLDS if args.fold == "all" else (args.fold,)
    variants = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    for f in folds:
        for v in variants:
            p = build(f, v)
            print(f"[built] {p.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
