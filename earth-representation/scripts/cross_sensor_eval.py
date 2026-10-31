"""Cross-sensor retrieval: does the encoder put the same place together?

For a frozen encoder each sample appears three times (S2, Landsat, HLS) on the
same ground footprint. If the representation transfers across sensors, an S2
feature should sit closer to its own Landsat/HLS counterpart than to other
locations.

Top-1 is a harsh statistic, so the report also carries recall@10, the median
rank of the true match, and the cosine gap between true pairs and random pairs.
All features are computed in large GPU batches.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "dino"))

from vision_transformer import DINOHead  # noqa: E402

from src.data import triples  # noqa: E402
from src.models.unified_encoder import UnifiedEncoder  # noqa: E402
from train_dino_official import BandBackbone  # noqa: E402


def build(embed_dim=512, out_dim=256):
    enc = UnifiedEncoder(width=32)
    head = DINOHead(embed_dim, out_dim, nlayers=3, hidden_dim=512, bottleneck_dim=64)
    import utils as dino_utils
    return dino_utils.MultiCropWrapper(BandBackbone(enc), head)


@torch.no_grad()
def features(model, items, key, device, batch=256):
    enc = model.backbone.encoder
    enc.eval()
    imgs = torch.from_numpy(np.stack([it[key]["image"] for it in items])).to(device)
    centre = torch.from_numpy(np.stack([it[key]["centre"] for it in items])).to(device)
    gsd = torch.from_numpy(np.stack([it[key]["gsd"] for it in items])).to(device)
    out = []
    for i in range(0, len(imgs), batch):
        f = enc(imgs[i:i + batch], centre[i:i + batch], gsd[i:i + batch])["global"]
        out.append(torch.nn.functional.normalize(f, dim=1))
    return torch.cat(out)


def retrieval_stats(query, gallery, chunk=256):
    """Recall@1/@10, median rank and cosine contrast for the true pair."""
    n = len(query)
    ranks, same, other = [], [], []
    for i in range(0, n, chunk):
        q = query[i:i + chunk]
        sim = q @ gallery.T                       # [chunk, n]
        truth = torch.arange(i, min(i + chunk, n), device=q.device)
        true_sim = sim[torch.arange(len(truth)), truth]
        # rank = number of gallery items strictly closer than the true match
        ranks.append((sim > true_sim.unsqueeze(1)).sum(dim=1))
        same.append(true_sim)
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask[torch.arange(len(truth)), truth] = False
        other.append(sim[mask].mean())
    ranks = torch.cat(ranks).float()
    same = torch.cat(same).mean()
    return {
        "recall@1": round(float((ranks < 1).float().mean()), 4),
        "recall@10": round(float((ranks < 10).float().mean()), 4),
        "median_rank": int(ranks.median().item()),
        "cos_true": round(float(same), 4),
        "cos_random": round(float(torch.stack(other).mean()), 4),
        "n": n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = args.device if torch.cuda.is_available() else "cpu"
    items = triples.load_triples(limit=args.n)
    print(f"triples: {len(items)} | device {device}", flush=True)

    model = build().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("teacher", state.get("student", state)))

    feats = {k: features(model, items, k, device) for k in ("s2", "landsat", "hls")}
    out = {"checkpoint": Path(args.checkpoint).name, "pairs": {}}
    for src, dst in (("s2", "landsat"), ("s2", "hls"), ("landsat", "hls")):
        st = retrieval_stats(feats[src], feats[dst])
        out["pairs"][f"{src}->{dst}"] = st
        print(f"{src}->{dst}: R@1 {st['recall@1']} R@10 {st['recall@10']} "
              f"median rank {st['median_rank']} | cos true {st['cos_true']} "
              f"vs random {st['cos_random']}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
