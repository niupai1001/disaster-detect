"""Runtime diagnostic on fetched real data, not a data-compliance acceptance."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.models.build import build_model
from landslide_seg.data.pipeline import ModelBatch
from landslide_seg.training.trainer import Trainer
from landslide_seg.training.losses import CombinedLoss
folder = ROOT / "outputs/review"
with np.load(folder / "real_diagnostic_input.npz") as a:
    b = ModelBatch(x12=torch.from_numpy(a["x12"])[None], x10=torch.from_numpy(a["x10"])[None],
        s2_dates=torch.tensor([[float(a["doy"])]]), mask=torch.from_numpy(a["mask"])[None],
        valid=torch.from_numpy(a["valid"])[None], sample_ids=["chimanimani_s2_1001.nc"])
torch.manual_seed(44)
random.seed(44)
m = build_model(device="cuda", branch_dropout=0)
t = Trainer(m, CombinedLoss(), device="cuda", out_dir=folder)
log = t.train_step(b)
torch.cuda.synchronize()
p = t.save(folder / "review_real_diagnostic.pt", extra={"certified_compliant": False})
def fixed():
    torch.manual_seed(99)
    random.seed(99)
    return m.predict(b.x12,b.x10,b.s2_dates)
before = fixed()
t.load(p)
after = fixed()
result = {"status": "REAL_RUNTIME_PASSED_DATA_COMPLIANCE_UNVERIFIED", "sample": b.sample_ids,
    "step": log, "reload_same_instance_max_prob_delta": (before["prob"]-after["prob"]).abs().max().item(),
    "fine_shape": list(after["mask"].shape), "checkpoint": str(p),
    "limitation": "Source identity, radiometry and pixel geometry have not passed formal acceptance; not performance evidence."}
(folder/"real_runtime.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result, indent=2))
