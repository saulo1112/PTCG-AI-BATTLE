#!/bin/bash
cd "c:/Users/ASUS/Desktop/S/Pokemon TCG"
declare -a rows=(
  "decks/thirdptcgclub.csv|data/models/thirdptcgclub_screen.json|thirdptcg"
  "decks/teamname.csv|data/models/teamname_screen.json|teamname"
)
for r in "${rows[@]}"; do
  IFS='|' read deck wts lbl <<< "$r"
  echo "===== $lbl vs kanga1052 ====="
  PYTHONIOENCODING=utf-8 uv run --group dev python scratchpad/head_to_head.py "$deck" "$wts" 6 --baseline "decks/懒惰的金枪鱼.csv" data/models/bc_kangaskhan_1052.json --label "$lbl" --baseline-label kanga1052 2>&1 | grep -aE "paired delta|VERDICT"
done
echo "ALLDONE_LAST2"
