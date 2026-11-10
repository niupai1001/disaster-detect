"""V3-H entrypoint; preload stdlib ssl before adding legacy project's src/ssl."""
import ssl
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(ROOT/'src'))
if len(sys.argv)<2 or sys.argv[1] not in ('check','train'):
    raise SystemExit('Usage: python scripts/run_v3_h.py check|train [arguments]')
mode=sys.argv.pop(1)
runpy.run_module('encoder_v3.hierarchical.'+mode,run_name='__main__')
