"""Run the existing Viewer CLI without a terminal-owned console."""
from pathlib import Path
import sys


def main():
    root=Path(__file__).resolve().parent.parent.parent
    log=(root/'observation_model/outputs/encoder_v2/viewer_scheduled_service.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout=sys.stderr=log
    from training_viewer.cli import main as viewer
    sys.argv=['training-viewer','serve','--port','8766','--host','127.0.0.1','--no-browser','--data-dir',str(root/'observation_model/outputs/repair_v2/viewer')]
    raise SystemExit(viewer())


if __name__=='__main__':main()
