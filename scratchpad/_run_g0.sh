#!/bin/sh
# G-0 secuencial: dos brazos NO caben en RAM a la vez (cada bigX ~2 GB).
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
python -u scratchpad/train_context_heads.py --contexts MAIN --profile ALAKAZAM_MAIN2 \
    --seeds 1 --l2 0.0001 > scratchpad/_m46_g0_main2.log 2>&1
echo "=== brazo MAIN2 terminado ===" >> scratchpad/_m46_g0_main2.log
python -u scratchpad/train_context_heads.py --contexts MAIN --profile ALAKAZAM \
    --seeds 1 --l2 0.0001 > scratchpad/_m46_g0_ctrl.log 2>&1
echo "=== brazo CONTROL terminado ===" >> scratchpad/_m46_g0_ctrl.log
