#!/bin/bash
cd "c:/Users/ASUS/Desktop/S/Pokemon TCG"
declare -a rows=(
  "decks/twshin.csv|data/models/twshin_screen.json|twshin"
  "decks/driestufalabs.csv|data/models/driestufalabs_screen.json|dries"
  "decks/yudaiueno.csv|data/models/yudaiueno_screen.json|yudai"
  "decks/junlee789.csv|data/models/junlee789_screen.json|junlee"
)
for r in "${rows[@]}"; do
  IFS='|' read deck wts lbl <<< "$r"
  echo "===== $lbl vs kanga1052 ====="
  PYTHONIOENCODING=utf-8 uv run --group dev python scratchpad/head_to_head.py "$deck" "$wts" 6 --baseline "decks/懒惰的金枪鱼.csv" data/models/bc_kangaskhan_1052.json --label "$lbl" --baseline-label kanga1052 2>&1 | grep -aE "profile=|macro WR over field|paired delta|VERDICT"
done
echo "ALLDONE_M23H2H"
