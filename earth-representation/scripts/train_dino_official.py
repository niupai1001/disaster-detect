"""DINO training driven by the official facebookresearch/dino implementation.

Only the domain-specific parts stay ours: the variable-band encoder, the
Sentinel-2/Landsat/HLS data loading and the physical augmentations. The DINO
head, loss, momentum schedule and multi-crop wrapper come from the vendored
official code (vendor/dino), because re-implementing them was where the
collapse came from.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "dino"))

import main_dino  # noqa: E402  official DINOLoss
import utils as dino_utils  # noqa: E402  official MultiCropWrapper / cosine_scheduler
from vision_transformer import DINOHead  # noqa: E402  official head

from src.models.unified_encoder import UnifiedEncoder  # noqa: E402
from src.ssl import train_dino as base  # noqa: E402


def ensure_single_process_group():
    """Official DINOLoss.update_center calls dist.all_reduce unconditionally.

    Initialising a gloo group on this Windows build segfaults, so the two
    collectives are replaced by their single-process equivalents. The loss math
    itself stays untouched.
    """
    if not dist.is_available():
        return
    dist.all_reduce = lambda tensor, *a, **k: tensor
    dist.get_world_size = lambda *a, **k: 1


class BandBackbone(torch.nn.Module):
    """Official MultiCropWrapper calls backbone(x); this injects band metadata.

    MultiCropWrapper concatenates crops along the batch axis, so the descriptors
    of the B base samples are repeated for every crop.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = torch.nn.Identity()
        self.head = torch.nn.Identity()
        self._centre = None
        self._gsd = None
        self._mask = None
        self._n_base = 1

    def set_meta(self, centre, gsd, n_base, mask=None):
        self._centre, self._gsd, self._n_base, self._mask = centre, gsd, n_base, mask

    def _meta(self, x):
        rep = max(1, x.shape[0] // self._n_base)
        centre = self._centre.repeat_interleave(rep, dim=0)
        gsd = self._gsd.repeat_interleave(rep, dim=0)
        mask = None if self._mask is None else self._mask.repeat_interleave(rep, dim=0)
        return centre, gsd, mask

    def forward(self, x):
        centre, gsd, mask = self._meta(x)
        return self.encoder(x, centre, gsd, mask)["global"]

    def levels(self, x, key="s16"):
        centre, gsd, mask = self._meta(x)
        return self.encoder(x, centre, gsd, mask)[key]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model_cnn.yaml"))
    ap.add_argument("--source", choices=["npz", "landslide", "mixed"],
                    default="landslide")
    ap.add_argument("--n-patches", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--tag", default="official")
    ap.add_argument("--tv-root", default=None,
                    help="if set, also write a TVLP 1.0 run under this log root")
    ap.add_argument("--tv-project", default="earth-representation")
    ap.add_argument("--scale-mode", default="none",
                    choices=["none", "uniform", "fixed", "random", "selective"])
    ap.add_argument("--scale-lambda", type=float, default=0.1)
    ap.add_argument("--scale-level", default="s16")
    ap.add_argument("--aug-mode", default="standard",
                    choices=["standard", "generic", "physical", "physical_band",
                             "physical_band_radiometry"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dino_official"))
    args = ap.parse_args(argv)

    ensure_single_process_group()
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.source == "mixed":
        from src.data import triples
        patches = triples.load_mixed(n_per_product=max(1, args.n_patches // 3))
    elif args.source == "landslide":
        from src.data import landslide_patches
        patches = landslide_patches.load(n=args.n_patches)
    else:
        patches = base.load_patches(ROOT / "data" / "interim" / "patches")
    if not patches:
        raise SystemExit("no patches")
    print(f"patches: {len(patches)} | steps {args.steps} | batch {args.batch_size}",
          flush=True)

    encoder_s = UnifiedEncoder(width=cfg["model"]["width"])
    encoder_t = UnifiedEncoder(width=cfg["model"]["width"])
    embed_dim = encoder_s.backbone.out_channels["s32"]
    head_s = DINOHead(embed_dim, args.out_dim, nlayers=3,
                      hidden_dim=512, bottleneck_dim=64)
    head_t = DINOHead(embed_dim, args.out_dim, nlayers=3,
                      hidden_dim=512, bottleneck_dim=64)
    student = dino_utils.MultiCropWrapper(
        BandBackbone(encoder_s), head_s).to(device)
    teacher = dino_utils.MultiCropWrapper(
        BandBackbone(encoder_t), head_t).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_crops = 2 + 2
    dino_loss = main_dino.DINOLoss(
        out_dim=args.out_dim, ncrops=n_crops,
        warmup_teacher_temp=0.04, teacher_temp=0.07,
        warmup_teacher_temp_epochs=max(1, args.steps // 10),
        nepochs=args.steps, student_temp=0.1,
        center_momentum=0.9).to(device)

    params = dino_utils.get_params_groups(student)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.04)
    lr_sched = dino_utils.cosine_scheduler(args.lr, 1e-6, args.steps, 1,
                                           warmup_epochs=max(1, args.steps // 10))
    wd_sched = dino_utils.cosine_scheduler(0.04, 0.4, args.steps, 1)
    mom_sched = dino_utils.cosine_scheduler(0.99, 0.9999, args.steps, 1)

    tv = None
    if args.tv_root:
        # direct TVLP file writing: the official writer's threads crash with CUDA
        from src.ssl.tvlp_writer import TvlpRun
        tv = TvlpRun(
            args.tv_root, project=args.tv_project, name=args.tag, seed=args.seed,
            metric_definitions={"train_loss": {"direction": "min"},
                                "lr": {"direction": "none"}},
            config={"steps": args.steps, "batch_size": args.batch_size,
                    "lr": args.lr, "out_dim": args.out_dim,
                    "aug_mode": args.aug_mode, "scale_mode": args.scale_mode,
                    "n_patches": args.n_patches, "seed": args.seed},
            dataset={"name": "landslide-sen12", "patches": args.n_patches},
            model={"name": "variable-band CNN + DINOHead"})
        print("tvlp run dir:", tv.dir, flush=True)

    rng = np.random.default_rng(args.seed)
    t_rng = torch.Generator(device=device).manual_seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / f"{args.tag}.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    step = 0
    rec = {"loss": None}

    while step < args.steps:
            for batch in base.mixed_batches(patches, args.batch_size, rng):
                if step >= args.steps:
                    break
            # keep the whole batch on the GPU so augmentation runs there too
            images = batch["image"].to(device)
            centre, gsd = batch["centre"].to(device), batch["gsd"].to(device)
            n_b = images.shape[0]

            crops = []
            masks = []
            for b in range(n_b):
                gs, ls = base.multi_crop(images[b], cfg["aug"]["global_size"],
                                         cfg["aug"]["local_size"],
                                         cfg["aug"]["n_local"], rng=t_rng)
                views = [base.augment_view(v, args.aug_mode, t_rng)[0]
                         for v in gs + ls]
                if args.aug_mode.startswith("physical_band"):
                    from src.augment import band_dropout
                    m = band_dropout.sample_band_mask(images.shape[1], t_rng, keep_min=3)
                    views = [v * m.view(-1, 1, 1) for v in views]
                    masks.append(m)
                else:
                    masks.append(None)
                crops.append([v.unsqueeze(0) for v in views])
            # MultiCropWrapper wants one tensor per resolution, batch = n_b
            per_view = [torch.cat([crops[b][v] for b in range(n_b)]).to(device)
                        for v in range(n_crops)]

            bmask = (torch.stack(masks).to(device)
                     if any(m is not None for m in masks) else None)
            student.backbone.set_meta(centre, gsd, n_b, bmask)
            teacher.backbone.set_meta(centre, gsd, n_b, bmask)
            teacher_input = per_view[:2]

            for i, pg in enumerate(opt.param_groups):
                pg["lr"] = lr_sched[step]
                if i == 0:
                    pg["weight_decay"] = wd_sched[step]

            student_out = student(per_view)
            with torch.no_grad():
                teacher_out = teacher(teacher_input)
            loss = dino_loss(student_out, teacher_out, step)
            if args.scale_mode != "none":
                from src.augment import spatial_degradation as sd
                from src.ssl import scale_consistency as sc
                hi = per_view[0]
                lo = sd.spatial_degrade(hi, 1.0, 2.0)
                f_hi = student.backbone.levels(hi, args.scale_level)
                f_lo = student.backbone.levels(lo, args.scale_level)
                sl = sc.scale_loss(f_hi, f_lo, r_list=(2, 4), mode=args.scale_mode,
                                   generator=t_rng)
                loss = loss + args.scale_lambda * sl

            opt.zero_grad(set_to_none=True)
            if step < max(1, args.steps // 10):
                dino_utils.cancel_gradients_last_layer(step, student, 1)
            loss.backward()
            dino_utils.clip_gradients(student, 3.0)
            opt.step()

            with torch.no_grad():
                m = mom_sched[step]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                for bs, bt in zip(student.buffers(), teacher.buffers()):
                    if bt.dtype.is_floating_point:
                        bt.mul_(m).add_(bs.detach(), alpha=1 - m)
                    else:
                        bt.copy_(bs)

            step += 1
            if step % 5 == 0 or step == 1:
                rec = {"step": step, "loss": round(float(loss.detach()), 4),
                       "lr": round(lr_sched[step - 1], 7), "m": round(m, 5),
                       "sec": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{args.tag}] step {step}/{args.steps} loss {rec['loss']} "
                      f"lr {rec['lr']} {rec['sec']}s", flush=True)

    log.close()
    if tv is not None:
        tv.finish("completed")
    torch.save({"student": student.state_dict(), "teacher": teacher.state_dict()},
               outdir / f"{args.tag}.pt")
    print(json.dumps({"steps": step, "final_loss": rec["loss"],
                      "checkpoint": str(outdir / f"{args.tag}.pt")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








