"""Run a source module from the project environment."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if len(sys.argv)<2:raise SystemExit('Usage: python scripts/run_module.py module [arguments]')
module=sys.argv.pop(1)
runpy.run_module(module,run_name='__main__')
