"""Download the checkpoint named by the pinned official repository, with identity."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "weights/dofa/XShadow_DOFA_base"


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=False)
    info = api.model_info("XShadow/DOFA", files_metadata=True)
    filename = "DOFA_ViT_base_e100.pth"
    entry = next(file for file in info.siblings if file.rfilename == filename)
    checkpoint = Path(hf_hub_download(repo_id="XShadow/DOFA", filename=filename, revision=info.sha,
                                     local_dir=DEST, cache_dir=ROOT / "weights/.hf_cache", token=False))
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    lfs = getattr(entry, "lfs", None)
    expected = getattr(lfs, "sha256", None) if lfs is not None else None
    if isinstance(lfs, dict):
        expected = lfs.get("sha256")
    if expected and digest.hexdigest() != expected:
        raise ValueError("Downloaded checkpoint hash differs from official HF LFS metadata")
    identity = {"repo_id": "XShadow/DOFA", "revision": info.sha, "filename": filename,
                "size_bytes": checkpoint.stat().st_size, "sha256": digest.hexdigest(),
                "remote_lfs_sha256": expected, "official_code_commit": "0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9"}
    path = DEST / "weight_identity.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != identity:
        raise FileExistsError("Refuse changed DOFA weight identity")
    path.write_text(json.dumps(identity, indent=1), encoding="utf-8")
    print(json.dumps(identity, indent=1))
