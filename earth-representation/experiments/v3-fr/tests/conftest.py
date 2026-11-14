import os
from pathlib import Path
import ssl
import sys

import torch


runtime = Path(os.environ["V3_FR_RUNTIME"])
sys.path[:0] = [str(runtime / "scripts"), str(runtime / "src")]
import encoder_v3
encoder_v3.__path__.append(str(Path(__file__).resolve().parents[1] / "src/encoder_v3"))
torch.set_num_threads(2)
