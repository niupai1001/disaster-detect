"""Isolated CUDA runtime probe and environment snapshot; no parent CUDA context."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
PY = sys.executable


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONDONTWRITEBYTECODE="1")
    code = (
        "import json,torch;"
        "print(json.dumps(dict(device_count=torch.cuda.device_count(),"
        "name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "ok=bool(torch.ones(8,device='cuda:0').sum().item()==8))))"
    )
    try:
        child = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60, env=env)
        probe = dict(returncode=child.returncode, stdout=child.stdout[-5000:], stderr=child.stderr[-5000:], timed_out=False)
    except subprocess.TimeoutExpired:
        probe = dict(returncode=None, stdout="", stderr="TIMEOUT", timed_out=True)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"],
                         capture_output=True, text=True, timeout=60)
    report = dict(
        schema="encoder_v3_formal_readiness_runtime_probe_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, visible_devices="1",
        probe=probe, nvidia_smi=dict(returncode=smi.returncode, stdout=smi.stdout, stderr=smi.stderr),
        status="CUDA_RUNTIME_OK" if probe.get("returncode") == 0 and "\"ok\": true" in (probe.get("stdout") or "").lower() else "BLOCKED_RUNTIME",
        notes=["Child process only; no global driver/environment change.", "Prior native access-violation reports remain historical evidence."],
    )
    out = AUDIT / "evidence" / "cuda_probe.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], probe=probe, output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
