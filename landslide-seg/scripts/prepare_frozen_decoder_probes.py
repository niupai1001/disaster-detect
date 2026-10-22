"""Bind two probes to immutable v4r1 source checkpoints; preserve existing configs."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256, load_segformer_config


def main():
    config = load_segformer_config(ROOT / "configs/segformer/v4r1/segformer_b0_direct10_n100_u3600_s1234.yaml")
    source = ROOT / "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n100_u3600_s1234/1234"
    meta = json.loads((source / "run_meta.json").read_text(encoding="utf-8"))
    target = ROOT / "configs/segformer/v4r2_probe"
    target.mkdir(exist_ok=True)
    registry = []
    for tag, step in (("best", 1200), ("last", 3600)):
        cfg = copy.deepcopy(config)
        s = cfg["segformer"]
        s["experiment"] = f"segformer_b0_direct10_frozen_{tag}_decodeprobe_n100_u3600_s1234"
        s["data"]["version"] = "s2_10band_v4r2_probe"
        path = source / f"{tag}.pt"
        s["model"]["probe"] = {
            "checkpoint": path.relative_to(ROOT).as_posix(), "checkpoint_sha256": file_sha256(path),
            "source_step": step, "stats_sha256": meta["stats_sha256"],
            "manifest_sha256": meta["manifest"]["sha256"],
        }
        dest = target / f"{s['experiment']}.yaml"
        content = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        if dest.exists() and dest.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refuse changed probe configuration: {dest}")
        dest.write_text(content, encoding="utf-8")
        registry.append({"experiment": s["experiment"], "source": tag, "source_step": step,
                         "config": dest.relative_to(ROOT).as_posix(), "config_sha256": file_sha256(dest)})
    path = target / "experiments.json"
    content = json.dumps(registry, indent=1) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError("Refuse changed probe registry")
    path.write_text(content, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
