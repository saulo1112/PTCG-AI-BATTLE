"""M42 G-A2 — did the ACTIVATE head actually change BEHAVIOUR, or just a metric?

This is the check that was missing from seven earlier mispredictions (M11/M14/M31/
M32) and that M34 introduced: re-score the teacher's OWN decisions with the real
`ImitationPolicy` built from the candidate payload, and confirm the intervention
moves the behaviour it was designed to move. An offline accuracy number has lied
about the ladder every single time it was trusted alone.

WHAT IT MEASURES. ACTIVATE is "use this optional Ability?" (Psychic Draw on evolve,
Dudunsparce's Run Away Draw, Fezandipiti's Flip the Script). Measured on Yushin's
2,330-game corpus:

  teacher declines  6.6% of 14,750 ACTIVATE decisions
  champion declines 0 of 760 on our own ladder replays (55145833)

The champion cannot decline at all: ACTIVATE has no head, so it falls to
`GreedyPolicy._safe_default`, which returns `list(range(minCount))` = option 0 = YES,
with no look at the state. So the gate is not "is the model more accurate" — it is
"does the clone now decline at a rate resembling the teacher, without declining
where the teacher would not".

Scoring goes through the REAL policy, so the live lowest-index tie-break and the
per-context profile resolution are reproduced by construction rather than
re-implemented (the trap `semantic_fidelity.py` falls into).

Run:
  PYTHONPATH=src python scratchpad/analyze_activate_divergence.py \
      --weights data/models/bc_alakazam_ctx.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.parser import ObservationParser

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
DECK = Path("decks/yushinito.csv")
CHAMPION = Path("data/models/bc_alakazam_fetch.json")
CONTEXT = "ACTIVATE"

# Pre-registered gate (fixed before running, per docs/handoff.md discipline).
GATE_MIN_DECLINE = 0.03    # candidate must decline on >= 3% of decisions
GATE_MAX_FALSE = 0.02      # ... while wrongly declining on <= 2% of teacher-YES


def _evaluate(policy, rows, cards, parser, label):
    """Agreement + decline behaviour of one policy on the teacher's ACTIVATE states."""
    n = agree = 0
    t_no = c_no = both_no = false_no = 0
    for r, obs in rows:
        chosen = [a for a in r.action if 0 <= a < len(obs.select.option)]
        if not chosen:
            continue
        try:
            pred = policy.choose(DecisionContext(raw=r.raw_observation, observation=obs, cards=cards))
        except Exception:  # noqa: BLE001
            pred = []
        n += 1
        teacher_no = chosen != [0]      # option 0 is always YES in this context
        clone_no = pred != [0]
        agree += set(pred) == set(chosen)
        t_no += teacher_no
        c_no += clone_no
        both_no += teacher_no and clone_no
        false_no += clone_no and not teacher_no
    return {
        "label": label, "n": n, "agree": agree / max(n, 1),
        "teacher_no": t_no, "clone_no": c_no, "both_no": both_no, "false_no": false_no,
        "clone_no_rate": c_no / max(n, 1),
        "recall": both_no / max(t_no, 1),
        "false_no_rate": false_no / max(n - t_no, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="candidate payload")
    ap.add_argument("--baseline", type=Path, default=CHAMPION)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--split", choices=["test", "all"], default="test",
                    help="'test' = the held-out 20%% the head never saw (default)")
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    deck = [int(x) for x in DECK.read_text(encoding="utf-8").split()]

    all_rows = list(read_decision_dataset(args.dataset))
    _, test_rows = split_by_game(all_rows, val_fraction=0.2, seed=0)
    rows = test_rows if args.split == "test" else all_rows

    # parse once, share across both policies (same states, paired comparison)
    parsed = []
    for r in rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context.name != CONTEXT or F.is_prize_pick(obs.select):
            continue
        parsed.append((r, obs))
    print(f"dataset={args.dataset.name}  split={args.split}  {CONTEXT} decisions={len(parsed)}")

    results = []
    for path, label in ((args.baseline, "champion"), (args.weights, "candidate")):
        pol = ImitationPolicy(path, deck=deck)
        has_head = CONTEXT in json.loads(Path(path).read_text(encoding="utf-8"))["contexts"]
        results.append(_evaluate(pol, parsed, cards, parser, f"{label} ({'head' if has_head else 'greedy'})"))
        print(f"  {label:<10} bc_failures={pol._bc_failures}")

    t_no = results[0]["teacher_no"]
    n = results[0]["n"]
    print(f"\nteacher declines: {t_no}/{n} = {t_no/max(n,1):.4f}\n")
    print(f"{'policy':<22}{'agree':>8}{'declines':>11}{'rate':>9}{'recall':>9}{'false-NO':>10}")
    for r in results:
        print(f"{r['label']:<22}{r['agree']:>8.4f}{r['clone_no']:>11}{r['clone_no_rate']:>9.4f}"
              f"{r['recall']:>9.3f}{r['false_no_rate']:>10.4f}")

    cand = results[-1]
    g1 = cand["clone_no_rate"] >= GATE_MIN_DECLINE
    g2 = cand["false_no_rate"] <= GATE_MAX_FALSE
    print(f"\nG-A2a decline rate >= {GATE_MIN_DECLINE:.2f} : "
          f"{'PASS' if g1 else 'FAIL'} ({cand['clone_no_rate']:.4f})")
    print(f"G-A2b false declines <= {GATE_MAX_FALSE:.2f} : "
          f"{'PASS' if g2 else 'FAIL'} ({cand['false_no_rate']:.4f})")
    print(f"G-A2 OVERALL: {'PASS' if g1 and g2 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
