@echo off
cd /d N:\Users\yzh\Disater\landslide-seg
C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe -X utf8 scripts\run_v6_remaining.py --gpu0 cuda:0 --gpu1 cuda:1 >> reports\v6_execution\run_remaining.log 2>&1
