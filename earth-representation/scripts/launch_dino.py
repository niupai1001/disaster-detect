"""Initialize CUDA before domain DLLs/large arrays on this Windows runtime."""
import argparse
from pathlib import Path
import runpy
import sys

import torch


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--release",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device",default="auto",choices=["auto","cuda","cpu"])
    options,remaining=parser.parse_known_args()
    if "--help" not in remaining and options.device!="cpu" and torch.cuda.is_available():
        torch.cuda.init()
    sys.argv=[str(options.release/"scripts/train_dino_official.py"),"--device",options.device,*remaining]
    runpy.run_path(sys.argv[0],run_name="__main__")


if __name__=="__main__":main()
