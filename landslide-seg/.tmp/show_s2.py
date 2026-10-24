import json
import pathlib

base = pathlib.Path("outputs/segformer/s2_10band_v4r2_spectral_mapping")
for exp in ("SegFormer-B0-LateDetail-OODv4-S2WavelengthMap",
            "SegFormer-B0-LateDetail-OODv4-S2FreeMap"):
    for tag in ("best", "last", "step1000", "step1500", "step2000"):
        p = base / exp / "1234" / "eval" / f"eval_{tag}_val.json"
        if not p.is_file():
            print(exp.split("-")[-1], tag, "missing")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        by = {k: round(v["pixel"]["iou"], 4) for k, v in d["by_event"].items()}
        print(f"{exp.split('-')[-1]:14s} {tag:9s} mean={d['event_mean_iou']:.4f} "
              f"AP={d['overall']['pixel']['average_precision']:.4f} events={by}")
