#!/bin/bash
cd "c:/Users/ASUS/Desktop/S/Pokemon TCG"
echo "===== THIRD(hand) vs kanga1052 ====="
PYTHONIOENCODING=utf-8 uv run --group dev python scratchpad/head_to_head.py decks/thirdptcgclub.csv data/models/bc_third_ptcg.json 12 --baseline "decks/懒惰的金枪鱼.csv" data/models/bc_kangaskhan_1052.json --label THIRDhand --baseline-label kanga1052 2>&1 | grep -aE "profile=|macro WR over field|paired delta|VERDICT"
echo "===== THIRD(hand) vs v1 ====="
PYTHONIOENCODING=utf-8 uv run --group dev python scratchpad/head_to_head.py decks/thirdptcgclub.csv data/models/bc_third_ptcg.json 12 --label THIRDhand 2>&1 | grep -aE "macro WR over field|paired delta|VERDICT"
echo "ALLDONE_THIRD"
