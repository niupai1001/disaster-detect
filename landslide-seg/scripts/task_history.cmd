@echo off
cd /d N:\Users\yzh\Disater\landslide-seg
C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe -X utf8 scripts\train_utae_v6.py --mode history --device cuda:1 --resume >> reports\v6_execution\task_history.log 2>&1
