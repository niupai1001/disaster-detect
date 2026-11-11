"""Create a new source-frozen release for an explicitly authorized V3-H-R run."""
import ssl
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import argparse
import json
import shutil

from encoder_v3.data import sha256
from encoder_v3.hierarchical_revised.train import validate_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--release",type=Path,required=True)
    args = parser.parse_args()
    document = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(document)
    release = args.release.resolve()
    if release.exists():
        parser.error("A new release directory is required")
    release.mkdir(parents=True)
    runtime = release/"runtime"
    sources = list((ROOT/"src/encoder_v3").glob("*.py"))
    for folder in ("shared_mae","latent_response","hierarchical","hierarchical_revised"):
        sources.extend((ROOT/"src/encoder_v3"/folder).glob("*.py"))
    sources.extend(ROOT/path for path in (
        "scripts/train_v3_h_revised.py", "src/encoder_v3/config_optical_selfsup_v1.json",
        "configs/config_lr_e0_v1.json", "outputs/encoder_v3/latent_response/channel_catalog_frozen.json",
        "data/physics/sensor_physics.json"))
    manifest = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = runtime/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        manifest[relative.as_posix()] = sha256(target)
    (release/"config.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (release/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("FROZEN",release,len(manifest),"files")


if __name__ == "__main__":
    main()
