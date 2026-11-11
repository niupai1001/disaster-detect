"""Launch the independent frozen V3-H-R formal runner."""
import ssl
import runpy
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
runpy.run_module("encoder_v3.hierarchical_revised.train",run_name="__main__")
