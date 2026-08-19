#!/bin/bash
cd /root/diffusion-distill
# wait for queue 1 (exp03 -> exp04) to finish
while pgrep -f "exp/_pw.py|04_lambda_ladder.py" > /dev/null; do sleep 20; done
echo "[queue2] starting exp06 d=32" >> results/queue.log
python3 exp/06_toy_distill.py --d 32 --steps 3000 --bs 256 \
    --modes teacher robust fixed data --seeds 0 1 > results/exp06_d32.log 2>&1
echo "[queue2] exp06 d=32 finished" >> results/queue.log
python3 exp/05_analyze.py >> results/queue.log 2>&1
echo "[queue2] figures regenerated" >> results/queue.log
