"""Run covered engineering checks and save evidence, not a stage-completion claim."""
import argparse
import json
from pathlib import Path
import time
import unittest
from .data import sha256
from .gates import require_training
from .cuda_probe import probe_cuda


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parent;start=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(root),pattern='test_*.py',top_level_dir=str(root.parent))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    config=json.loads((root/'config.json').read_text());gate_message=None
    try:require_training(config)
    except ValueError as exc:gate_message=str(exc)
    report=dict(status=('ENGINEERING_TESTS_PASS_WITH_SKIPS' if result.skipped else 'ENGINEERING_TESTS_PASS') if result.wasSuccessful() and gate_message else 'FAILED',tests=result.testsRun,skipped=[dict(test=str(case),reason=reason) for case,reason in result.skipped],cuda_initialization=probe_cuda(),failures=[str(x[0]) for x in result.failures],errors=[str(x[0]) for x in result.errors],seconds=time.perf_counter()-start,formal_gate_rejected=gate_message,source_hashes={str(path.relative_to(root)):sha256(path) for path in sorted(root.glob('*.py'))},limitations=['Unit tests include tiny synthetic optimizer fixtures; they do not consume or replace the separate real-data pilot ledger','Skipped GPU cases are unverified, never GPU acceptance','Not a replacement for full data, GPU equivalence or per-stage acceptance'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(0 if report['status'].startswith('ENGINEERING_TESTS_PASS') else 1)
