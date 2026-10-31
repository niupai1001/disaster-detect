"""DINO training loop over locally materialised patches.

Patches are grouped by product so a batch always has one band count; the model
itself supports any band count, but batching needs a common shape.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..augment import band_dropout
from ..data import band_catalog
from ..models.dino_head import DINOHead
from ..models.unified_encoder import UnifiedEncoder
from .dino_loss import DINOLoss
from .ema_teacher import EMATeacher
from .batching import mixed_batches
from .multi_crop import multi_crop

PROJECT = Path(__file__).resolve().parents[2]
NAME_TO_BAND = {b.name: b for bands in band_catalog.BY_PRODUCT.values() for b in bands}


def load_patches(npz_dir, limit=0):
    files = sorted(Path(npz_dir).glob("*.npz"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        # npz written before the QA band was dropped carries QA names too
        bands = [str(b) for b in d["bands"]
                 if NAME_TO_BAND.get(str(b)) and NAME_TO_BAND[str(b)].kind == "reflectance"]
        image = d["image"]
        if len(bands) != image.shape[0]:
            bands = bands[:image.shape[0]]
        out.append({"image": image, "valid": d["valid"], "bands": bands,
                    "centre": np.array([NAME_TO_BAND[b].centre_nm for b in bands],
                                       dtype=np.float32),
                    "gsd": np.array([NAME_TO_BAND[b].native_gsd_m for b in bands],
                                    dtype=np.float32),
                    "path": str(f)})
    return out


def descriptor(bands):
    centre = [NAME_TO_BAND[b].centre_nm for b in bands]
    gsd = [NAME_TO_BAND[b].native_gsd_m for b in bands]
    return torch.tensor(centre, dtype=torch.float32), torch.tensor(gsd, dtype=torch.float32)


def batches(items, batch_size, rng):
    """Group by band signature, so [B,C,H,W] is always well defined."""
    groups = {}
    for it in items:
        groups.setdefault(tuple(it["bands"]), []).append(it)
    keys = list(groups)
    rng.shuffle(keys)
    for k in keys:
        g = groups[k]
        rng.shuffle(g)
        for i in range(0, len(g), batch_size):
            yield g[i:i + batch_size]


def make_model(cfg):
    enc = UnifiedEncoder(width=cfg["model"]["width"],
                         aggregate=cfg["model"]["aggregate"])
    head = DINOHead(in_dim=enc.backbone.out_channels["s32"],
                    out_dim=cfg["dino"]["out_dim"],
                    hidden=cfg["dino"].get("hidden", 1024),
                    bottleneck=cfg["dino"].get("bottleneck", 128))
    return torch.nn.ModuleDict({"encoder": enc, "head": head})


def forward_model(model, images, centre, gsd, device, band_mask=None):
    levels = model["encoder"](images.to(device), centre, gsd, band_mask)
    return model["head"](levels["global"])


def augment_view(x, mode, t_rng):
    """Apply the configured physical (or control) augmentation to one view."""
    from ..augment import reflectance_perturb as rp
    from ..augment import spatial_degradation as sd

    params = {}
    if mode in ("generic", "physical", "physical_band", "physical_band_radiometry"):
        if float(torch.rand(1, generator=t_rng, device=x.device)) < 0.5:
            factor = float(torch.tensor([1.5, 2.0, 3.0], device=x.device)[
                int(torch.randint(0, 3, (1,), generator=t_rng, device=x.device))])
            if mode == "generic":
                x = sd.generic_resize(x, factor)
                params = {"resize": factor}
            else:
                sigma = float(torch.tensor([0.6, 1.0, 1.5], device=x.device)[
                    int(torch.randint(0, 3, (1,), generator=t_rng, device=x.device))])
                x = sd.spatial_degrade(x, sigma, factor)
                params = {"sigma": sigma, "factor": factor}
    if mode == "physical_band_radiometry":
        x, p = rp.apply_gain_offset(x, t_rng)
        params.update(p)
    return x, params


def run(cfg, patches, outdir, seed=0, max_steps=0, tag="run"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(cfg).to(device)
    total_steps = max_steps or cfg["train"]["steps"]
    ema = EMATeacher(model, base_momentum=cfg["train"]["ema_base"],
                     total_steps=total_steps,
                     cap=cfg["train"].get("ema_cap", 0.999))
    teacher_model = ema.teacher.to(device)

    head_loss = DINOLoss(out_dim=cfg["dino"]["out_dim"],
                         teacher_temp=cfg["dino"]["teacher_temp"],
                         student_temp=cfg["dino"]["student_temp"],
                         center_momentum=cfg["dino"].get("center_momentum", 0.9)).to(device)
    # Weight decay on the prototype layer shrinks its weight-norm scale toward
    # zero, which makes every prototype identical and the teacher target
    # uniform. DINO therefore excludes the last layer (and bias/norm params).
    # The head collapses to rank ~1 if it trains at the backbone's rate: every
    # input then maps to the same normalised vector, which makes the teacher's
    # targets identical across samples. A separate, slower rate prevents that.
    head_scale = cfg["train"].get("head_lr_scale", 0.1)
    decay, no_decay, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(p)
        elif p.ndim <= 1 or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0},
         {"params": head_params, "weight_decay": 0.0}],
        lr=cfg["train"]["lr"])
    head_group = 2
    warmup = max(1, int(0.1 * total_steps))

    def lr_at(step):
        base = cfg["train"]["lr"]
        if step < warmup:
            return base * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return base * 0.5 * (1 + math.cos(math.pi * progress))

    # The teacher runs in eval mode, so its BatchNorm uses running statistics.
    # At initialisation those are the default (0, 1) and never see real data,
    # which makes the teacher's output almost constant and its targets uniform.
    # Warm the statistics up by running the teacher in train mode for a few
    # steps, then freeze it to eval.
    teacher_model.train()
    bn_warmup = cfg["train"].get("teacher_bn_warmup", 20)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"{tag}.jsonl"
    step, t0 = 0, time.time()
    model.train()
    torch_rng = torch.Generator().manual_seed(seed)
    rec = {"loss": None}

    while step < total_steps:
        for batch in mixed_batches(patches, cfg["train"]["batch_size"], rng):
            if step >= total_steps:
                break
            images = batch["image"]            # [B, C, H, W], padded to C_max
            centre, gsd, avail = batch["centre"], batch["gsd"], batch["mask"]
            n_b, n_c = images.shape[0], images.shape[1]
            n_g, n_l = 2, cfg["aug"]["n_local"]
            mode = cfg["aug"].get("mode", "standard")

            gs_list, ls_list, m_list = [], [], []
            for b in range(n_b):
                g_raw, l_raw = multi_crop(images[b], cfg["aug"]["global_size"],
                                          cfg["aug"]["local_size"], n_l, rng=torch_rng)
                m = avail[b].clone()
                if mode in ("physical_band", "physical_band_radiometry"):
                    idx = torch.nonzero(m).squeeze(1)
                    keep = band_dropout.sample_band_mask(
                        len(idx), torch_rng,
                        keep_min=cfg["aug"].get("keep_min_bands", 3))
                    m = torch.zeros_like(m)
                    m[idx] = keep
                gs, ls, params_seen = [], [], []
                for v in g_raw:
                    out_v, p = augment_view(v, mode, torch_rng)
                    gs.append(out_v * m.view(-1, 1, 1))
                    params_seen.append(p)
                for v in l_raw:
                    out_v, p = augment_view(v, mode, torch_rng)
                    ls.append(out_v * m.view(-1, 1, 1))
                    params_seen.append(p)
                if cfg["aug"].get("log_params") and b == 0:
                    from ..augment import augmentation_log as alog
                    alog.record(Path(outdir) / f"{tag}_aug.jsonl", step, "sample0",
                                mode, {"views": params_seen}, seed)
                gs_list.append(torch.stack(gs))
                ls_list.append(torch.stack(ls))
                m_list.append(m)

            # all views share the padded band count, so one forward covers the batch
            g_all = torch.cat(gs_list)         # [B*n_g, C, H, W]
            l_all = torch.cat(ls_list)         # [B*n_l, C, H, W]
            m_g = torch.cat([m.unsqueeze(0).expand(n_g, -1) for m in m_list])
            m_l = torch.cat([m.unsqueeze(0).expand(n_l, -1) for m in m_list])
            c_g = centre.repeat_interleave(n_g, 0)
            d_g = gsd.repeat_interleave(n_g, 0)
            c_l = centre.repeat_interleave(n_l, 0)
            d_l = gsd.repeat_interleave(n_l, 0)

            s_g = forward_model(model, g_all, c_g, d_g, device, m_g.to(device))
            s_l = forward_model(model, l_all, c_l, d_l, device, m_l.to(device))
            s = torch.cat([s_g.view(n_b, n_g, -1), s_l.view(n_b, n_l, -1)], dim=1)
            s = s.permute(1, 0, 2)             # [V, B, D]
            with torch.no_grad():
                t_g = forward_model(teacher_model, g_all, c_g, d_g, device,
                                    m_g.to(device))
            t = t_g.view(n_b, n_g, -1).permute(1, 0, 2)   # [G, B, D]
            loss = head_loss(s, t, n_global=len(t))
            if os.environ.get("EARTH_DBG") and step < 3:
                with torch.no_grad():
                    tt = torch.softmax((t - head_loss.center) / head_loss.teacher_temp, -1)
                    ent = float(-(tt * tt.clamp_min(1e-9).log()).sum(-1).mean())
                print(f"DBG step{step} s{tuple(s.shape)} t{tuple(t.shape)} "
                      f"s_std={float(s.std()):.4f} t_std={float(t.std()):.4f} "
                      f"t_entropy={ent:.4f}", flush=True)

            base_lr = lr_at(step)
            for i, group in enumerate(opt.param_groups):
                group["lr"] = base_lr * (head_scale if i == head_group else 1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step < cfg["dino"].get("freeze_last_steps", 0):
                # DINO freezes the prototype layer early so the targets cannot
                # collapse onto a handful of prototypes before features form
                model["head"].last.weight.grad = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            m = ema.update_from(model)
            if step == bn_warmup:
                teacher_model.eval()

            step += 1
            if step % 5 == 0 or step == 1:
                with torch.no_grad():
                    # separation between prototypes: goes to 0 when the teacher collapses
                    proto_std = float(t.std(dim=-1).mean())
                    sample_std = float(t.std(dim=1).mean())
                rec = {"step": step, "loss": round(float(loss.detach()), 4),
                       "lr": round(opt.param_groups[0]["lr"], 7), "ema_m": round(m, 5),
                       "proto_std": round(proto_std, 5), "sample_std": round(sample_std, 5),
                       "sec": round(time.time() - t0, 1), "tag": tag}
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[{tag}] step {step}/{total_steps} loss {rec['loss']} "
                      f"proto_std {rec['proto_std']:.4f} m {rec['ema_m']} "
                      f"{rec['sec']}s", flush=True)

    ckpt = outdir / f"{tag}.pt"
    torch.save({"model": model.state_dict(),
                "teacher": teacher_model.state_dict(),
                "cfg": cfg, "step": step}, ckpt)
    return {"steps": step, "checkpoint": str(ckpt), "log": str(log_path),
            "final_loss": rec["loss"] if step else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patches", default=str(PROJECT / "data" / "interim" / "patches"))
    ap.add_argument("--source", choices=["npz", "landslide"], default="npz",
                    help="npz: Phase 1 patches; landslide: existing 12k netCDF patches")
    ap.add_argument("--n-patches", type=int, default=2048)
    ap.add_argument("--out", default=str(PROJECT / "outputs" / "dino"))
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="key=value overrides")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for kv in args.set:
        k, v = kv.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(v)

    if args.source == "landslide":
        from ..data import landslide_patches
        patches = landslide_patches.load(n=args.n_patches or args.limit or 2048)
    else:
        patches = load_patches(args.patches, limit=args.limit)
    if not patches:
        raise SystemExit(f"no patches under {args.patches}")
    print(f"patches: {len(patches)}", flush=True)
    result = run(cfg, patches, args.out, seed=args.seed,
                 max_steps=args.steps, tag=args.tag)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())












