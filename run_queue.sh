#!/bin/bash
# Serialised experiment queue -- 2 cores, so nothing runs concurrently.
cd /root/diffusion-distill
while pgrep -f "02_gauss_power.py" > /dev/null; do sleep 10; done
echo "[queue] exp02 finished, starting exp03" >> results/queue.log
python3 exp/_pw.py > results/exp03_raw.log 2>&1
echo "[queue] exp03 finished, starting exp04" >> results/queue.log
python3 exp/04_lambda_ladder.py > results/exp04_raw.log 2>&1
echo "[queue] exp04 finished" >> results/queue.log
