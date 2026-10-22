"""Read-only source weight changes; nonzero weights do not establish band utility."""
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(4)
from landslide_seg.config import load_segformer_config, file_sha256


def main():
    models = []
    for item in json.loads((ROOT / "configs/segformer/v4r2_probe/experiments.json").read_text()):
        cfg = load_segformer_config(ROOT / item["config"])
        probe = cfg["segformer"]["model"]["probe"]
        path = ROOT / probe["checkpoint"]
        assert file_sha256(path) == probe["checkpoint_sha256"]
        models.append(torch.load(path, map_location="cpu", weights_only=False)["model"])
    rows = []
    for name, best in models[0].items():
        if not name.startswith("model.segformer."):
            continue
        last = models[1][name]
        a, b = best.float().flatten(), last.float().flatten()
        denominator = float(a.norm())
        rows.append({"name": name, "parameters": a.numel(), "best_l2": denominator,
                     "last_l2": float(b.norm()), "delta_l2": float((b-a).norm()),
                     "relative_delta_l2": float((b-a).norm()) / denominator if denominator else None,
                     "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0))})
    out = ROOT / "reports/v4r2_probe_results"
    out.mkdir(exist_ok=True)
    for filename, data in (("encoder_tensor_changes.csv", rows),):
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    key = "model.segformer.encoder.patch_embeddings.0.proj.weight"
    a, b = models[0][key], models[1][key]
    bands = [{"band": name, "best_rms": float(a[:, i].square().mean().sqrt()),
              "last_rms": float(b[:, i].square().mean().sqrt()),
              "delta_rms": float((b[:, i]-a[:, i]).square().mean().sqrt()),
              "is_non_rgb": i > 2, "interpretation": "weight update only; not causal band contribution"}
             for i, name in enumerate(names)]
    with (out / "first_conv_band_weights.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bands[0]))
        writer.writeheader()
        writer.writerows(bands)
    print(json.dumps(bands, indent=1))


if __name__ == "__main__":
    main()
