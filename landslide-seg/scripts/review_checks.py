"""Independent review probes. Synthetic probes are interface checks only.

Does not modify production code, original logs, or checkpoints.
"""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.metadata
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import h5py
from landslide_seg.training.losses import bce_loss, CombinedLoss
from landslide_seg.training.metrics import confusion
from landslide_seg.data.sen12 import load_sen12, list_sen12_files

out = ROOT / "outputs" / "review"
out.mkdir(parents=True, exist_ok=True)
report = {"scope": "independent review; synthetic tests do not validate real data"}

def save():
    (out / "checks.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

report["environment"] = {"python": sys.version, "executable": sys.executable,
    "torch": torch.__version__, "cuda": torch.cuda.is_available(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "versions": {p: importlib.metadata.version(p) for p in
                 ["terratorch", "timm", "numpy", "h5py", "rasterio", "pyproj"]}}
z = torch.zeros(1, 1, 2, 2, requires_grad=True)
y = torch.zeros_like(z)
report["bce_without_valid"] = {"shape": list(bce_loss(z, y).shape)}
report["invalid_confusion"] = confusion(np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2)))
loss = CombinedLoss()(fine_logits=z, coarse_logits=z,
    presence_logits=torch.zeros(1, 1, requires_grad=True), target=y,
    valid=torch.zeros_like(z))
report["all_invalid_loss"] = loss.as_dict()
samples = []
for p in list_sen12_files(ROOT / "data" / "sen12", limit=6):
    s = load_sen12(p)
    _, i = s.event_index()
    with h5py.File(p) as f:
        attrs = {k: str(v) for k, v in f.attrs.items()}
        coordinates = {k: {"shape": list(f[k].shape), "first": np.asarray(f[k][:]).ravel()[:4].tolist(),
                            "last": np.asarray(f[k][:]).ravel()[-4:].tolist()} for k in ["x", "y"] if k in f}
        dims = {k: [list(d.keys()) for d in f[k].dims] for k in ["B02", "MASK"]}
    date = s.attrs["_time_ref"] + dt.timedelta(days=int(s.time[i]))
    samples.append({"file": p.name, "post_index": i, "actual_date": str(date),
        "actual_zero_based_doy": date.timetuple().tm_yday - 1, "loader_doy": s.doy(i),
        "mask_px": int(s.mask[i].sum()), "valid_positive_px": int((s.mask[i] * s.valid_mask(i)).sum()),
        "attrs": attrs, "coordinates": coordinates, "dimensions": dims})
report["samples"] = samples
save()
print("CPU probes saved", flush=True)

from landslide_seg.models.build import build_model
from landslide_seg.data.pipeline import make_synthetic_batch
from landslide_seg.training.trainer import Trainer

torch.manual_seed(123)
random.seed(123)
model = build_model(device="cuda", branch_dropout=0)
report["parameter_report"] = model.parameter_report()
model.train()
report["after_train_modes"] = {n: {"training": m.training,
    "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
    "active_dropout": [(k, v.p) for k, v in m.named_modules()
                       if isinstance(v, torch.nn.Dropout) and v.training and v.p > 0
                       and (n != "anysat" or "projector_s2." in k)][:8]}
    for n, m in [("terramind", model.terramind), ("anysat", model.anysat), ("text", model.semantic.text)]}
save()

counts = {"tim_generate": 0}
original_generate = model.terramind.model.sampler.generate
def counted_generate(*args, **kwargs):
    counts["tim_generate"] += 1
    return original_generate(*args, **kwargs)
model.terramind.model.sampler.generate = counted_generate
batch = make_synthetic_batch(batch_size=1, size=128, device="cuda")
trainer = Trainer(model, CombinedLoss(), device="cuda", out_dir=out)
before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
log = trainer.train_step(batch)
torch.cuda.synchronize()
report["synthetic_train_step"] = log
report["updated_by_module"] = {group: sum(not torch.equal(before[n], p.detach())
    for n,p in model.named_parameters() if n in before and n.startswith(group + "."))
    for group in ["fusion", "semantic", "coarse", "refiner"]}
report["tim_call_counts"] = dict(counts)
report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
save()
model.eval()
with torch.no_grad():
    torch.manual_seed(321)
    random.seed(321)
    a = model(batch.x12, batch.x10, batch.s2_dates, region_mask=batch.mask[:, None])
    torch.manual_seed(321)
    random.seed(321)
    b = model(batch.x12, batch.x10, batch.s2_dates, region_mask=1-batch.mask[:, None])
    report["gt_prediction_delta"] = (a.fine_logits-b.fine_logits).abs().max().item()
    report["output_shapes"] = {k: list(getattr(a,k).shape) for k in ["fine_logits", "coarse_logits", "presence_logits"]}
    empty_emb = model.semantic.region_embedding(a.features["pyramid"][0], torch.zeros_like(batch.mask))
    report["empty_region_embedding_norm"] = empty_emb.norm().item()
ckpt = trainer.save(out / "review_synthetic.pt", extra={"synthetic": True})
trainer.load(ckpt)
report["checkpoint_same_instance_load"] = "passed; not an independent rebuild verification"
report["tim_call_counts"] = dict(counts)
report["weights_sha256"] = {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    for p in (ROOT/"weights").glob("*.p*")}
save()
print(json.dumps({k: report[k] for k in ["after_train_modes", "synthetic_train_step", "updated_by_module", "tim_call_counts", "gt_prediction_delta", "empty_region_embedding_norm"]}, indent=2))

# Optional real-data diagnostic: preserve the loader's current behavior, but do
# not certify source identity/radiometric/geometric compliance by running it.
from landslide_seg.data.sen12 import assemble_full
from landslide_seg.data.pipeline import ModelBatch
try:
    real = assemble_full(load_sen12(list_sen12_files(ROOT / "data" / "sen12")[1]))
    if real is None:
        report["real_diagnostic"] = {"status": "BLOCKED", "reason": "no same-day item"}
    else:
        rb = ModelBatch(x12=torch.from_numpy(real["x12"])[None],
            x10=torch.from_numpy(real["x10"])[None], s2_dates=torch.tensor([[float(real["doy"])]]),
            mask=torch.from_numpy(real["mask"])[None], valid=torch.from_numpy(real["valid"])[None],
            sample_ids=[real["provenance"]["file"]], provenance=[real["provenance"]])
        real_log = trainer.train_step(rb)
        model.eval()
        real_pred = model.predict(rb.x12, rb.x10, rb.s2_dates)
        real_ckpt = trainer.save(out / "review_real_diagnostic.pt", extra={"certified_compliant": False})
        trainer.load(real_ckpt)
        report["real_diagnostic"] = {"status": "RAN_BUT_COMPLIANCE_UNVERIFIED", "log": real_log,
            "provenance": real["provenance"], "valid_fraction": float(real["valid"].mean()),
            "pred_shape": list(real_pred["mask"].shape)}
except Exception as exc:
    report["real_diagnostic"] = {"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)}
save()
print("Real diagnostic:", json.dumps(report["real_diagnostic"], indent=2), flush=True)
