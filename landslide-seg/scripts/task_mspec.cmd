@echo off
cd /d N:\Users\yzh\Disater\landslide-seg
C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe -X utf8 scripts\train_temporal.py --config configs\segformer\v6_main\temporal_multi_spectral.yaml --device cuda:0 --resume >> reports\v6_execution\task_mspec.log 2>&1
