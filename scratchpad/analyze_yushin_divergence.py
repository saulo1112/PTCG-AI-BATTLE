"""M33 §5 — WHERE exactly does the clone disagree with the teacher?

Fidelity is ~0.78 on MAIN, so roughly 1 in 5 of the teacher's own decisions is
one the clone would make differently. Nothing in the project has ever looked at
WHICH ones, with board context -- only aggregate accuracy. This runs the shipped
clone over the teacher's own logged decisions and categorises every disagreement
by what the teacher did vs what the clone would do, plus the board state.

Uses the pre-extracted dataset (data/imitation/yushinito_full.jsonl.gz) rather
than re-parsing replays, so it is fast; samples to keep runtime bounded.

READ-ONLY. Run:
  uv run --group dev python scratchpad/analyze_yushin_divergence.py --sample 18000
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

import diagnose_mlp as dm

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
WEIGHTS = "data/models/bc_alakazam_mlp.json"
POWERFUL_HAND = dm.POWERFUL_HAND


def _fmt(vals):
    if not vals:
        return "n=0"
    return f"n={len(vals)} mean={statistics.mean(vals):.2f} median={statistics.median(vals):.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=18000,
                     help="max MAIN decisions to score (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    deck = [int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()]
    pol = ImitationPolicy(WEIGHTS, deck=deck)

    print("loading dataset...", flush=True)
    rows = [r for r in read_decision_dataset(DATASET)
            if SelectContextKind(r.context) is SelectContextKind.MAIN]
    print(f"  {len(rows)} teacher MAIN decisions available", flush=True)
    if args.sample and len(rows) > args.sample:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
        print(f"  sampled {len(rows)}", flush=True)

    agree = disagree = skipped = 0
    # what the teacher chose -> what the clone chose, when they differ
    confusion: collections.Counter = collections.Counter()
    teacher_type_tot: collections.Counter = collections.Counter()
    teacher_type_agree: collections.Counter = collections.Counter()
    by_turn: dict[str, list[int]] = {"agree": [], "disagree": []}
    by_hand: dict[str, list[int]] = {"agree": [], "disagree": []}
    by_nopts: dict[str, list[int]] = {"agree": [], "disagree": []}
    by_won: dict[bool, list[int]] = {True: [0, 0], False: [0, 0]}  # [agree, total]
    # the specific case that drove M32: PH legal, what does each side do?
    ph_legal_tot = 0
    ph_teacher_fired = ph_clone_fired = ph_both = 0

    for i, r in enumerate(rows):
        if i % 2000 == 0 and i:
            print(f"  ...{i}/{len(rows)}", flush=True)
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or len(obs.select.option) < 2:
            skipped += 1
            continue
        gs = GameState.build(obs, cards)
        ctx = DecisionContext(raw=r.raw_observation, observation=obs, cards=cards)
        try:
            pick = pol.choose(ctx)
        except Exception:
            skipped += 1
            continue
        if not pick:
            skipped += 1
            continue

        t_idx = [a for a in r.action if 0 <= a < len(obs.select.option)]
        if not t_idx:
            skipped += 1
            continue
        t_opt = obs.select.option[t_idx[0]]
        c_opt = obs.select.option[pick[0]]
        same = set(pick) == set(t_idx)

        t_name = t_opt.type.name
        c_name = c_opt.type.name
        teacher_type_tot[t_name] += 1
        bucket = "agree" if same else "disagree"
        by_turn[bucket].append(obs.current.turn)
        by_hand[bucket].append(gs.hand_size)
        by_nopts[bucket].append(len(obs.select.option))
        by_won[r.won][1] += 1

        ph_idx = [j for j, o in enumerate(obs.select.option)
                  if o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND]
        if ph_idx:
            ph_legal_tot += 1
            t_ph = any(a in ph_idx for a in t_idx)
            c_ph = any(a in ph_idx for a in pick)
            ph_teacher_fired += t_ph
            ph_clone_fired += c_ph
            ph_both += (t_ph and c_ph)

        if same:
            agree += 1
            teacher_type_agree[t_name] += 1
            by_won[r.won][0] += 1
        else:
            disagree += 1
            confusion[(t_name, c_name)] += 1

    scored = agree + disagree
    print("\n" + "=" * 78)
    print("SECTION 5 — clone vs teacher, decision by decision")
    print("=" * 78)
    print(f"scored {scored} MAIN decisions ({skipped} skipped)")
    print(f"  agree    : {agree}  ({agree / max(scored,1):.1%})")
    print(f"  disagree : {disagree}  ({disagree / max(scored,1):.1%})")

    print(f"\n  AGREEMENT BY WHAT THE TEACHER CHOSE:")
    print(f"    {'teacher chose':<22}{'n':>8}{'agree':>9}{'rate':>9}")
    for t, n in teacher_type_tot.most_common():
        a = teacher_type_agree[t]
        print(f"    {t:<22}{n:>8}{a:>9}{a / max(n,1):>8.1%}")

    print(f"\n  TOP DISAGREEMENTS (teacher chose -> clone chose):")
    print(f"    {'teacher':<18}{'clone':<18}{'n':>8}{'% of all disagreements':>24}")
    for (t, c), n in confusion.most_common(15):
        print(f"    {t:<18}{c:<18}{n:>8}{n / max(disagree,1):>23.1%}")

    print(f"\n  CONTEXT of agreements vs disagreements:")
    for name, d in (("turn", by_turn), ("hand size", by_hand), ("n options", by_nopts)):
        print(f"    {name:<12} agree   : {_fmt([float(x) for x in d['agree']])}")
        print(f"    {name:<12} disagree: {_fmt([float(x) for x in d['disagree']])}")

    print(f"\n  agreement in games the teacher WON vs LOST:")
    for won in (True, False):
        a, n = by_won[won]
        print(f"    {'WON ' if won else 'LOST'}: {a}/{n} = {a / max(n,1):.1%}")

    print(f"\n  THE M32 CASE — decisions where Powerful Hand was legal ({ph_legal_tot}):")
    print(f"    teacher fired it : {ph_teacher_fired} ({ph_teacher_fired / max(ph_legal_tot,1):.1%})")
    print(f"    clone would fire : {ph_clone_fired} ({ph_clone_fired / max(ph_legal_tot,1):.1%})")
    print(f"    both fired       : {ph_both}")
    print(f"    => the clone is {'MORE' if ph_clone_fired > ph_teacher_fired else 'LESS'} "
          f"willing to fire PH than the teacher, on the teacher's own states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
