"""Build Phase 1 patch manifests over the configured regions.

Every patch is read from the remote COGs by window; the region is never
downloaded whole. Run with a subset first if the network is slow.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import manifest, patch_dataset  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--only", nargs="*", help="subset of region names")
    ap.add_argument("--products", nargs="*",
                    default=["sentinel2", "landsat", "hls"])
    ap.add_argument("--max-patches", type=int, default=6,
                    help="stop after this many patches per region/product")
    ap.add_argument("--min-valid", type=float, default=0.30)
    ap.add_argument("--materialize", type=int, default=1,
                    help="npz patches to save per region/product for tests")
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--out", default=str(ROOT / "data" / "manifests"))
    ap.add_argument("--merge", action="store_true",
                    help="keep existing rows for regions not rebuilt")
    args = ap.parse_args(argv)

    regions = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))["regions"]
    names = args.only or list(regions)
    all_rows, provenance, summaries = [], [], []
    t0 = dt.datetime.now()

    for name in names:
        region = regions[name]
        dates = sorted({dt.date.fromisoformat(d) for d in region["event_dates"]})
        if not dates:
            continue
        start = (dates[0] - dt.timedelta(days=args.window_days)).isoformat()
        end = (dates[-1] + dt.timedelta(days=args.window_days)).isoformat()
        for product in args.products:
            try:
                rows, summary, scenes = patch_dataset.build_region(
                    name, region["bbox"], product, start, end,
                    min_valid=args.min_valid, max_patches=args.max_patches)
            except Exception as exc:
                summary = {"region": name, "product": product,
                           "error": f"{type(exc).__name__}: {exc}"}
                rows, scenes = [], {}
                print(f"{name:16s} {product:10s} FAILED {exc}")
            summaries.append(summary)
            all_rows.extend(rows)
            for sid, scene in scenes.items():
                provenance.append({"region": name, "product": product,
                                   "scene_id": sid, "datetime": scene.datetime,
                                   "crs": f"EPSG:{scene.epsg}",
                                   "assets": scene.assets,
                                   "extra": {k: v for k, v in scene.extra.items()
                                             if isinstance(v, (str, int, float))}})
            if rows and args.materialize:
                for row in rows[:args.materialize]:
                    scene = scenes.get(row["scene_id"])
                    if scene is None:
                        continue
                    try:
                        path = patch_dataset.save_patch(
                            scene, row, ROOT / "data" / "interim" / "patches", product)
                        if path:
                            row["npz"] = str(path.relative_to(ROOT))
                    except Exception as exc:
                        summary["materialize_error"] = f"{type(exc).__name__}: {exc}"
            n = summary.get("patches", 0)
            tail = f"  {summary['error']}" if "error" in summary else ""
            print(f"{name:16s} {product:10s} scenes={summary.get('scenes', 0):3d} "
                  f"patches={n:3d}{tail}", flush=True)

    if args.merge and (Path(args.out) / "train.jsonl").exists():
        keep_rows, keep_prov = [], []
        for split in ("train", "val", "test"):
            for row in manifest.read_jsonl(Path(args.out) / f"{split}.jsonl"):
                if row["region"] not in names:
                    keep_rows.append(row)
        for row in manifest.read_jsonl(Path(args.out) / "provenance.jsonl"):
            if row["region"] not in names:
                keep_prov.append(row)
        all_rows = keep_rows + all_rows
        provenance = keep_prov + provenance

    counts = manifest.write_split_manifests(args.out, all_rows, provenance)
    summary_path = Path(args.out) / "patch_summary.json"
    summary_path.write_text(json.dumps({
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round((dt.datetime.now() - t0).total_seconds(), 1),
        "regions": names, "products": args.products,
        "patches": len(all_rows), "splits": counts,
        "valid_fraction_mean": round(
            sum(r["valid_fraction"] for r in all_rows) / max(len(all_rows), 1), 4),
        "per_region_product": summaries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\npatches {len(all_rows)} | splits {counts} | "
          f"elapsed {(dt.datetime.now() - t0).total_seconds():.0f}s")
    print("manifest:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

