"""Offline: does BC-guided search improve or degrade BC's MAIN picks? (M10 Phase 2)

On the teacher's held-out MAIN decisions, compare the TEACHER's actual choice to
(a) plain BC and (b) search-BC. If search overturns BC and those overturns match
the teacher LESS than BC did, search is replacing good moves with bad ones — the
mechanism behind the arena regression. Runs offline (one search per decision, no
full games) so it's cheap and directly diagnostic.

Run:  uv run --group dev python scratchpad/audit_search_overturns.py [n_sample] [greedy_bias]
"""

from __future__ import annotations

import sys
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.search import SearchConfig
from ptcg_ai.decision.search_bc import ImitationSearchPolicy
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser

DECK_IDS = [int(x) for x in Path("decks/greengreenpurple.csv").read_text().split()]
W_V1 = "data/models/bc_650_v1.json"
V_MODEL = "data/models/v_650.json"


def main() -> int:
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    gb = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    cfg = load_config(profile="benchmark")
    sdk = load_sdk(cfg.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    parser = ObservationParser()

    bc = ImitationPolicy(W_V1, deck=DECK_IDS)
    search = ImitationSearchPolicy(W_V1, V_MODEL, deck=DECK_IDS, cards=cards, api=sdk.api,
                                   search_cfg=SearchConfig(greedy_bias=gb), rng_seed=0)

    rows = [r for r in read_decision_dataset(Path("data/imitation/greengreenpurple.jsonl.gz"))]
    _, val = split_by_game(rows, val_fraction=0.2, seed=0)
    main_rows = [r for r in val if SelectContextKind(r.context) is SelectContextKind.MAIN][:n_sample]

    bc_match = search_match = overturns = overturn_bc_right = overturn_search_right = 0
    considered = 0
    for r in main_rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or len(obs.select.option) < 2:
            continue
        ctx = DecisionContext(raw=r.raw_observation, observation=obs, cards=cards)
        teacher = set(r.action)
        try:
            bc_pick = set(bc.choose(ctx))
            search_pick = set(search.choose(ctx))
        except Exception:
            continue
        considered += 1
        bc_ok = bc_pick == teacher
        search_ok = search_pick == teacher
        bc_match += bc_ok
        search_match += search_ok
        if search_pick != bc_pick:
            overturns += 1
            overturn_bc_right += bc_ok
            overturn_search_right += search_ok

    print(f"MAIN decisions considered: {considered}  (greedy_bias={gb})")
    print(f"  BC     matches teacher: {bc_match} ({bc_match/max(considered,1):.1%})")
    print(f"  search matches teacher: {search_match} ({search_match/max(considered,1):.1%})")
    print(f"\noverturns (search != BC): {overturns} ({overturns/max(considered,1):.1%} of decisions)")
    if overturns:
        print(f"  on overturns, BC matched teacher:     {overturn_bc_right} ({overturn_bc_right/overturns:.1%})")
        print(f"  on overturns, search matched teacher: {overturn_search_right} ({overturn_search_right/overturns:.1%})")
        print(f"  => search trades {overturn_bc_right} teacher-correct BC picks for "
              f"{overturn_search_right} teacher-correct search picks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
