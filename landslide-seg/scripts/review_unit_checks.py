"""Small independent review probes; no official weights required."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("spatial_review", ROOT / "tests/test_spatial_alignment.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
failed = []
for seed in range(30):
    torch.manual_seed(seed)
    try:
        tests.test_spatial_alignment_impulse()
    except AssertionError:
        failed.append(seed)
from landslide_seg.models.coarse import CoarseLocator
head = CoarseLocator(channels=16, hidden=16, n_levels=4)
pyr = [torch.randn(1,16,n,n) for n in [32,16,8,4]]
o = head(pyr, torch.zeros(1,1,32,32))
(o["coarse_logits"].mean() + o["presence_logits"].mean()).backward()
result = {"spatial_test_failed_seeds": failed, "tested_seeds": 30,
    "coarse_parameters_without_gradient": [n for n,p in head.named_parameters() if p.grad is None]}
path = ROOT / "outputs/review/unit_checks.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
