"""Run isolated V3-H-R checks and engineering profiles."""
import ssl
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
runpy.run_module("encoder_v3.hierarchical_revised.profile", run_name="__main__")
