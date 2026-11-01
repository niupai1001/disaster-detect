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
import os
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


class LocalDINOLoss(main_dino.DINOLoss):
    @torch.no_grad()
    def update_center(self, teacher_output):
        self.center.mul_(self.center_momentum).add_(
            teacher_output.mean(dim=0, keepdim=True), alpha=1-self.center_momentum)


    def validation(self, student_output, teacher_output, step):
        center, ncrops = self.center.clone(), self.ncrops
        self.ncrops = 4
        try:
            return main_dino.DINOLoss.forward(self, student_output, teacher_output, step)
        finally:
            self.center.copy_(center)
            self.ncrops = ncrops


class SensorDINOLoss(LocalDINOLoss):
    def __init__(self, *args, cross_sensor=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_sensor = cross_sensor

    def forward(self, student_output, teacher_output, step):
        from src.ssl.sensor_routes import observation_dino_loss
        loss = observation_dino_loss(student_output, teacher_output, self.center,
            self.teacher_temp_schedule[step], self.student_temp, self.cross_sensor)
        self.update_center(teacher_output)
        return loss


from src.ssl.sensor_routes import ViewMetadataMixin


class BandBackbone(ViewMetadataMixin, torch.nn.Module):
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
        self._view_meta = None
        self._centre, self._gsd, self._n_base, self._mask = centre, gsd, n_base, mask

    def _meta(self, x):
        if getattr(self, "_view_meta", None) is not None:
            return ViewMetadataMixin._meta(self, x)
        assert x.shape[0] % self._n_base == 0
        rep = x.shape[0] // self._n_base
        centre = self._centre.repeat(rep, 1)
        gsd = self._gsd.repeat(rep, 1)
        mask = None if self._mask is None else self._mask.repeat(rep, 1)
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
    ap.add_argument("--augmentation-calibration", default=None)
    ap.add_argument("--sensor-release", default=None)
    ap.add_argument("--route", choices=["single", "mixed", "paired"], default=None)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--stop-after", type=int, default=0)
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
    ap.add_argument("--out", default=str(ROOT / "outputs" / "verified_dino"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if args.augmentation_calibration:
        os.environ["EARTH_AUGMENTATION_CALIBRATION"] = args.augmentation_calibration
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.scale_mode != "none" and (args.source != "landslide" or args.route):
        raise ValueError("corrective scale run is restricted to native-grid S2")

    if args.route:
        from src.data.sensor_release import load_release
        patches, validation, sensor_manifest = load_release(args.sensor_release, args.route)
        if args.steps>100 and sensor_manifest["status"]!="accepted":
            raise ValueError("formal sensor training requires an accepted frozen data release")
    elif args.source == "mixed":
        from src.data import triples
        patches = triples.load_mixed(n_per_product=max(1, args.n_patches // 3))
    elif args.source == "landslide":
        from src.data import landslide_patches
        patches = [] if args.adaptive else landslide_patches.load(n=args.n_patches)
    else:
        patches = base.load_patches(ROOT / "data" / "interim" / "patches")
    if not patches and not args.adaptive:
        raise SystemExit("no patches")
    print(f"patches: {len(patches)} | steps {args.steps} | batch {args.batch_size}",
          flush=True)

    if not args.route:
        validation = []
    if args.adaptive and not args.route:
        if args.source != "landslide":
            raise ValueError("adaptive mixed training requires accepted v2 membership release")
        records = [json.loads(x) for x in Path(os.environ.get("EARTH_PROTOCOL_MANIFEST", ROOT.parent / "landslide-seg/manifests/v4r1/pool_100.jsonl")).read_text().splitlines() if x.strip()]
        train_ids = {r["sample_id"] for r in records if r.get("split") == "train"}
        val_ids = {r["sample_id"] for r in records if r.get("split") == "val"}
        patches = landslide_patches.load(n=args.n_patches, allowed_ids=train_ids)
        validation = landslide_patches.load(n=128, allowed_ids=val_ids)
        if not patches or not validation:
            raise ValueError("empty frozen train or development membership")

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
    # CNN teacher uses batch statistics, as in official DINO training.
    teacher.train()
    for p in teacher.parameters():
        p.requires_grad_(False)

    freeze_steps = max(1, math.ceil(len(patches)/args.batch_size)) if args.adaptive else max(1,args.steps//10)
    if args.route and (cfg["aug"]["n_local"] != 2 or args.aug_mode != "standard"):
        raise ValueError("sensor routes use two local crops per observation and standard augmentation")
    n_crops = 8 if args.route else 2 + cfg["aug"]["n_local"]
    loss_type = SensorDINOLoss if args.route else LocalDINOLoss
    loss_kwargs = {"cross_sensor": args.route == "paired"} if args.route else {}
    dino_loss = loss_type(
        out_dim=args.out_dim, ncrops=n_crops,
        warmup_teacher_temp=0.04, teacher_temp=0.07,
        warmup_teacher_temp_epochs=max(1, min(args.steps,10*math.ceil(len(patches)/args.batch_size))) if args.adaptive else max(1,args.steps//10),
        nepochs=args.steps, student_temp=0.1,
        center_momentum=0.9, **loss_kwargs).to(device)

    params = dino_utils.get_params_groups(student)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=.5,
        patience=9, threshold=.005, threshold_mode="rel", min_lr=1e-6)
    epoch = 0
    best = float("inf")
    stale = 0
    stop_reason = "update_limit"

    lr_sched = dino_utils.cosine_scheduler(args.lr, 1e-6, args.steps, 1,
                                           warmup_epochs=max(1, args.steps // 10))
    wd_sched = dino_utils.cosine_scheduler(0.04, 0.4, args.steps, 1)
    mom_sched = dino_utils.cosine_scheduler(0.99, 0.9999, args.steps, 1)

    tv = None
    if args.tv_root:
        # direct TVLP file writing: the official writer's threads crash with CUDA
        from src.ssl.tvlp_writer import TvlpRun
        tv = TvlpRun(
            args.tv_root, project=args.tv_project, name=args.tag, seed=args.seed, run_id=args.tag, resume=bool(args.resume),
            metric_definitions={"train/loss": {"direction": "min"},
                                "train/lr": {"direction": "none"}},
            config={"steps": args.steps, "batch_size": args.batch_size,
                    "lr": args.lr, "out_dim": args.out_dim,
                    "aug_mode": args.aug_mode, "scale_mode": args.scale_mode,
                    "n_patches": len(patches), "seed": args.seed},
            dataset={"name": "stable S2-Landsat locations" if args.route else "landslide-sen12", "patches": len(patches)},
            model={"name": "variable-band CNN + DINOHead"})
        print("tvlp run dir:", tv.dir, flush=True)

    rng = np.random.default_rng(args.seed)
    t_rng = torch.Generator(device=device).manual_seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / f"{args.tag}.jsonl", "a" if args.resume else "x", encoding="utf-8")
    t0 = time.time()
    step = 0
    cursor = 0
    exposure = 0
    epoch_rng = rng.bit_generator.state
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        opt.load_state_dict(state["optimizer"])
        dino_loss.load_state_dict(state["dino_loss"])
        step, cursor, exposure = state["step"], state["cursor"], state["exposure"]
        epoch, best, stale = state.get("epoch",0), state.get("best",float("inf")), state.get("stale",0)
        if "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
        epoch_rng = state["epoch_rng"]
        rng.bit_generator.state = epoch_rng
        t_rng.set_state(state["augmentation_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])

    if not args.resume:
        torch.save({"student":student.state_dict(),"teacher":teacher.state_dict()}, outdir/f"{args.tag}.initial.pt")

    def save():
        checkpoint = {"student": student.state_dict(), "teacher": teacher.state_dict(),
            "optimizer": opt.state_dict(), "dino_loss": dino_loss.state_dict(),
            "step": step, "cursor": cursor, "exposure": exposure, "epoch_rng": epoch_rng,
            "augmentation_rng": t_rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "stale": stale, "stop_reason": stop_reason,
            "config": vars(args), "sampling_resume_version": 2, "members": [p.get("path") for p in patches]}
        tmp = outdir / f"{args.tag}.tmp"
        torch.save(checkpoint, tmp)
        tmp.replace(outdir / f"{args.tag}.pt")

    import hashlib, subprocess
    provenance = {"git_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("src","scripts","configs","vendor/dino") for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".py",".yaml")},
        "members": [p.get("path") for p in patches], "arguments": vars(args),
        "validation_members": [p.get("path") for p in validation], "teacher_mode":"train BN; parameter and buffer EMA"}
    (outdir/f"{args.tag}.provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    provenance["runtime_contract"] = dict(
        model=dict(stem_width=cfg["model"]["width"],aggregation="masked mean",descriptor="wavelength + native GSD FiLM",student_parameters=sum(p.numel() for p in student.parameters())),
        optimizer=dict(type="AdamW",lr=args.lr,regularized_weight_decay=.04,nonregularized_weight_decay=0.,gradient_clip=3.),
        dino=dict(out_dim=args.out_dim,hidden_dim=512,bottleneck_dim=64,student_temp=.1,teacher_temp_start=.04,teacher_temp_end=.07,center_momentum=.9,teacher_views=4 if args.route else 2,student_views=n_crops,teacher_mode="train batch statistics; EMA parameters and buffers"),
        validation=dict(members=len(validation),view_seed=999,BN="eval",center="restored",selection="relative loss improvement >=0.5%",not_semantic_labels=True),
        adaptive=dict(warmup_epochs=10,lr_patience=10,lr_factor=.5,min_lr=1e-6,min_epochs=100,stop_stale_epochs=30,stop_lr_fraction=1/16,max_epochs=args.max_epochs,max_updates=args.steps))
    (outdir/f"{args.tag}.provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    if args.route:
        provenance["sensor_release"] = sensor_manifest
        provenance["locations_per_batch"] = args.batch_size
        provenance["observation_groups_per_batch"] = args.batch_size*2
        provenance["distinct_sensor_observations_per_batch"] = args.batch_size*(1 if args.route=="single" else 2)
        provenance["student_views_per_step"] = args.batch_size*8
        (outdir/f"{args.tag}.provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    rec = {"loss": None}
    session_start_exposure = exposure

    try:
        while step < args.steps:
                epoch_rng = rng.bit_generator.state
                skip = cursor
                for batch_index, batch in enumerate(base.mixed_batches(patches, args.batch_size, rng)):
                    if batch_index < skip:
                        continue
                    if step >= args.steps:
                        break
                    # keep the whole batch on the GPU so augmentation runs there too
                    images = batch["image"].to(device)
                    centre, gsd = batch["centre"].to(device), batch["gsd"].to(device)
                    n_b = images.shape[0]

                    crops = []
                    masks = []
                    first_valid_views = []
                    for b in range(n_b):
                        gs, ls = base.multi_crop(torch.cat([images[b], batch["valid_mask"][b].to(device).unsqueeze(0)], dim=0), cfg["aug"]["global_size"],
                                                 cfg["aug"]["local_size"], cfg["aug"]["n_local"], rng=t_rng)
                        first_valid_views.append(gs[0][-1:] >= .999)
                        if args.route:
                            # Shared crop geometry; each model view still has only ten band slots.
                            views = ([v[:10] * (v[-1:] >= .999) for v in gs]
                                + [v[10:20] * (v[-1:] >= .999) for v in gs]
                                + [v[:10] * (v[-1:] >= .999) for v in ls]
                                + [v[10:20] * (v[-1:] >= .999) for v in ls])
                            masks.append(batch["mask"][b].to(device))
                        else:
                            views = [base.augment_view(v[:-1], args.aug_mode, t_rng)[0] * (v[-1:] >= .999) for v in gs + ls]
                            if args.aug_mode.startswith("physical_band"):
                                from src.augment import band_dropout
                                available = batch["mask"][b].to(device)
                                m = available.clone()
                                ids = available.nonzero().flatten()
                                m[ids] = band_dropout.sample_band_mask(len(ids), t_rng, keep_min=min(3, len(ids)))
                                views = [v * m.view(-1, 1, 1) for v in views]
                                masks.append(m)
                            else:
                                masks.append(batch["mask"][b].to(device))
                        crops.append([v.unsqueeze(0) for v in views])
                    per_view = [torch.cat([crops[b][v] for b in range(n_b)]) for v in range(n_crops)]
                    bmask = torch.stack(masks)
                    teacher_input = per_view[:4 if args.route else 2]
                    if args.route:
                        halves = (0, 0, 1, 1, 0, 0, 1, 1)
                        centres = [centre[:, i*10:(i+1)*10] for i in halves]
                        gsds = [gsd[:, i*10:(i+1)*10] for i in halves]
                        band_masks = [bmask[:, i*10:(i+1)*10] for i in halves]
                        student.backbone.set_view_meta(centres, gsds, band_masks, n_b)
                        teacher.backbone.set_view_meta(centres[:4], gsds[:4], band_masks[:4], n_b)
                    else:
                        student.backbone.set_meta(centre, gsd, n_b, bmask)
                        teacher.backbone.set_meta(centre, gsd, n_b, bmask)

                    for i, pg in enumerate(opt.param_groups):
                        if not args.adaptive:
                            pg["lr"] = lr_sched[step]
                        elif epoch < 10:
                            pg["lr"] = args.lr * (epoch + 1) / 10
                        if i == 0 and not args.adaptive:
                            pg["weight_decay"] = wd_sched[step]

                    student_out = student(per_view)
                    with torch.no_grad():
                        teacher_out = teacher(teacher_input)
                    loss = dino_loss(student_out, teacher_out, step)
                    scale_metrics = {}
                    if args.scale_mode != "none":
                        from src.ssl.common_scale import coarse_view, common_scale_loss, auxiliary_batch_statistics
                        hi = per_view[0]
                        valid_hi = torch.stack(first_valid_views)
                        lo, valid_lo = coarse_view(hi, valid_hi)
                        with auxiliary_batch_statistics(student):
                            f_hi = student.backbone.levels(hi, args.scale_level)
                            f_lo = student.backbone.levels(lo, args.scale_level)
                        joint_valid = valid_hi & (torch.nn.functional.interpolate(valid_lo.float(),size=hi.shape[-2:],mode="nearest")>.5)
                        sl, scale_metrics = common_scale_loss(f_hi, f_lo, hi, lo, joint_valid, args.scale_mode, t_rng)
                        scale_metrics = {"scale_loss":float(sl.detach()), **{k:float(v) for k,v in scale_metrics.items()}}
                        loss = loss + args.scale_lambda * sl

                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss at {step}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    dino_utils.cancel_gradients_last_layer(step, student, freeze_steps)
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
                    exposure += n_b
                    cursor = batch_index + 1
                    if tv is not None:
                        elapsed = max(time.time()-t0, 1e-6)
                        resources = {"session_seconds": elapsed,
                                     "session_locations_per_second": (exposure-session_start_exposure)/elapsed}
                        if device == "cuda":
                            resources.update(gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2,
                                             gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
                        tv.log("train", {"loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"], "exposure": exposure, "sensor_observation_exposure":exposure*(2 if args.route in ("mixed","paired") else 1), "logit_std": float(student_out.detach().std()), "sample_logit_std":float(student_out.detach().std(dim=0).mean()), **scale_metrics, **resources}, global_step=step, wall_time_sec=elapsed)
                        tv.heartbeat(step)
                    if step % 25 == 0:
                        save()
                    if args.stop_after and step >= args.stop_after:
                        save()
                        log.close()
                        if tv: tv.finish("cancelled", "engineering pause for resume")
                        return 0
                    if step % 5 == 0 or step == 1:
                        rec = {"step": step, "loss": round(float(loss.detach()), 4),
                               "lr": round(opt.param_groups[0]["lr"], 7), "m": round(m, 5),
                               "sec": round(time.time() - t0, 1)}
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        print(f"[{args.tag}] step {step}/{args.steps} loss {rec['loss']} "
                              f"lr {rec['lr']} {rec['sec']}s", flush=True)

                cursor = 0
                epoch_rng = rng.bit_generator.state
                epoch += 1
                if args.adaptive:
                    # Fixed views; no teacher center, BN or RNG changes during validation.
                    center = dino_loss.center.clone()
                    student.eval(); teacher.eval()
                    vrng = torch.Generator(device=device).manual_seed(999)
                    losses = []
                    with torch.no_grad():
                        for vb in base.mixed_batches(validation, args.batch_size, np.random.default_rng(999)):
                            vx=vb["image"].to(device); vc=vb["centre"].to(device); vg=vb["gsd"].to(device); vm=vb["mask"].to(device)
                            student.backbone.set_meta(vc,vg,len(vx),vm)
                            teacher.backbone.set_meta(vc,vg,len(vx),vm)
                            views=[base.multi_crop(x,cfg["aug"]["global_size"],cfg["aug"]["local_size"],cfg["aug"]["n_local"],rng=vrng) for x in vx]
                            crops=[torch.stack([(v[0]+v[1])[j] for v in views]) for j in range(4)]
                            losses.append(float(dino_loss.validation(student(crops),teacher(crops[:2]),step-1)))
                            dino_loss.center.copy_(center)
                    student.train(); teacher.train()
                    score=sum(losses)/len(losses)
                    if epoch >= 10: scheduler.step(score)
                    improved=score < best*.995
                    stale=0 if improved else stale+1
                    if improved: best=score
                    if tv: tv.log("val",{"loss":score},global_step=step)
                    save()
                    if improved:
                        import shutil
                        shutil.copyfile(outdir/f"{args.tag}.pt",outdir/f"{args.tag}.best_unlabeled.pt")
                    if epoch >= 100 and stale >= 30 and opt.param_groups[0]["lr"] <= args.lr/16:
                        stop_reason="validation_plateau"; break
                    if epoch >= args.max_epochs:
                        stop_reason="epoch_limit"; break

    except BaseException as exc:
        if tv is not None:
            tv.finish("failed", type(exc).__name__ + ": " + str(exc))
        log.close()
        raise

    log.close()
    if tv is not None:
        tv.finish("completed")
    save()
    print(json.dumps({"steps": step, "final_loss": rec["loss"],
                      "checkpoint": str(outdir / f"{args.tag}.pt")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








