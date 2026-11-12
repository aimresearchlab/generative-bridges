#!/bin/bash
# The posterior-variance restoration, then the paused A/B/C queue resumes.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=runs/.gpu.lock
while pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; do sleep 30; done
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"; exit' EXIT INT TERM

$PY -m bridges.variance_restore --n_samples 10000 --steps 10 50 --compile \
    --variants bridge bridge+var euler euler+var \
    --out runs/variance_restore.json >> runs/variance_restore.log 2>&1
echo "=== VARIANCE TEST COMPLETE ===" >> runs/variance_restore.log

rm -f "$LOCK"
./run_A_B_C.sh >> runs/A_B_C_queue.log 2>&1
