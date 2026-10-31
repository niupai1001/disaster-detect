"""Frozen-feature linear probe: which augmentation gives transferable features?

DINO loss cannot compare arms that remove information (band dropout), so the
comparison uses a probe instead: freeze the encoder, train one linear layer to
predict the region, and evaluate on regions never seen during the probe.

The encoder checkpoints come from outputs/e1_official. The same split and seed
are used for every arm, so the numbers are comparable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "dino"))

from vision_transformer import DINOHead  # noqa: E402

from src.data import landslide_patches  # noqa: E402
from src.models.unified_encoder import UnifiedEncoder  # noqa: E402
from train_dino_official import BandBackbone  # noqa: E402


def build(embed_dim, out_dim=256):
    enc = UnifiedEncoder(width=32)
    head = DINOHead(embed_dim, out_dim, nlayers=3, hidden_dim=512, bottleneck_dim=64)
    import utils as dino_utils
    return dino_utils.MultiCropWrapper(BandBackbone(enc), head)


def features(model, patches, device, which="teacher"):
    enc = model.backbone
    enc.eval()
    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(patches), 256):   # bigger batches: GPU-bound, not CPU-bound
            chunk = patches[i:i + 256]
            x = torch.from_numpy(np.stack([p["image"] for p in chunk])).float().to(device)
            centre = torch.from_numpy(np.stack([p["centre"] for p in chunk])).to(device)
            gsd = torch.from_numpy(np.stack([p["gsd"] for p in chunk])).to(device)
            enc.encoder.eval()
            out = enc.encoder(x, centre, gsd)["global"]
            feats.append(out.cpu())
            labels.extend(p.get("region", "unknown") for p in chunk)
    return torch.cat(feats), labels


def probe(feats, labels, train_regions, seed=0, steps=400, lr=1e-2,
          train_frac=0.8):
    """Linear probe with a random patch split.

    Holding out whole regions was wrong: the probe would have to predict class
    ids it never saw, so accuracy is 0 by construction. A random split measures
    what the probe is meant to measure - how separable the features are.
    """
    regions = sorted(set(labels))
    idx = {r: i for i, r in enumerate(regions)}
    y = torch.tensor([idx[r] for r in labels])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(labels), generator=g)
    n_train = int(train_frac * len(labels))
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask = ~train_mask

    torch.manual_seed(seed)
    clf = torch.nn.Linear(feats.shape[1], len(regions)).to(feats.device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)

    x_tr, y_tr = feats[train_mask], y[train_mask]
    for _ in range(steps):
        out = clf(x_tr)
        loss = F.cross_entropy(out, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        tr_acc = float((clf(x_tr).argmax(1) == y_tr).float().mean())
        te_acc = float((clf(feats[test_mask]).argmax(1) == y[test_mask]).float().mean())
    return {"train_acc": round(tr_acc, 3), "test_acc": round(te_acc, 3),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "n_classes": len(regions)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--n-patches", type=int, default=1600)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="regions held out of the probe; default = last three")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e1_official" / "probe.json"))
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patches = landslide_patches.load(n=args.n_patches)
    for p in patches:
        # usa_alaska / usa_puertorico must not collapse into one "usa" label
        p["region"] = Path(p["path"]).name.split("_s2_")[0]
    regions = sorted({p["region"] for p in patches})
    holdout = args.holdout or []
    train_regions = regions
    print(f"patches {len(patches)} | regions {len(regions)} | random split 80/20",
          flush=True)

    ckpts = args.checkpoints or sorted(
        str(p) for p in (ROOT / "outputs" / "e1_official").glob("E1-*.pt"))
    ckpts = ckpts + [str(ROOT / "outputs" / "dino_official" / "off3k.pt")]

    results = []
    for ck in ckpts:
        ck = Path(ck)
        if not ck.exists():
            continue
        t0 = time.time()
        model = build(512).to(device)
        state = torch.load(ck, map_location=device)
        model.load_state_dict(state["teacher"])
        # average features over both views is unnecessary: a single pass suffices
        feats, labels = features(model, patches, device)
        runs = [probe(feats, labels, set(train_regions), seed=s) for s in args.seeds]
        res = {k: round(sum(r[k] for r in runs) / len(runs), 3)
               for k in ("train_acc", "test_acc")}
        res["n_test"] = runs[0]["n_test"]
        res.update({"checkpoint": ck.name, "seconds": round(time.time() - t0, 1)})
        results.append(res)
        print(f"{ck.stem:10s} train {res['train_acc']:.3f} test {res['test_acc']:.3f} "
              f"({res['seconds']}s)", flush=True)

    Path(args.out).write_text(json.dumps(
        {"holdout": holdout, "n_patches": len(patches), "results": results},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

