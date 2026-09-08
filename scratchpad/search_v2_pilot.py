"""M17 search pilot — does the SEARCH mechanism improve with the T3-corrected leaf V?

M10 found BC-guided determinized search (ImitationSearchPolicy, leaf V = v_650, 7 feats)
played WORSE than plain imitation-v1, and attributed it to a weak evaluator. M17 Fase 1
showed the leaf V's `threat` feature is blind to prose-scaling attacks (Rocket Rush) and
that adding T3 (prose-aware threat) is a real +0.014-0.017 offline-AUC fix. This pilot
tests whether that corrected V makes the search MECHANISM better — a directional read
before deciding to invest more, NOT a ship attempt.

Controlled A/B: SAME search plumbing, SAME BC seed/fallback/rollout (v1), SAME small n,
SAME opponent (imitation-v1, mirror deck). The ONLY thing that changes between the two
arenas is the leaf V:
  * arena 1 (M10 baseline): leaf V = v_650.json      (base-7, teacher-trained)
  * arena 2 (pilot):        leaf V = v_650_v2.json    (base-7 + T3, SAME dataset/fitter)
so any delta is attributable to T3 alone. The leaf features are computed from the FIXED
root seat (`GameState.build_for`), matching search_bc.value_features.

Run:  uv run python scratchpad/search_v2_pilot.py [n] [greedy_bias]
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision import search_bc
from ptcg_ai.decision.evaluator import _board_energy, _deckout_risk
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.decision.search import SearchConfig
from ptcg_ai.decision.search_bc import ImitationSearchPolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import ParsedObservation
from ptcg_ai.state.game_state import GameState

sys.path.insert(0, "scratchpad")
from value_features_v2 import _prose_max_threat  # noqa: E402

DECK = "decks/greengreenpurple.csv"
W_V1 = "data/models/bc_650_v1.json"
V_OLD = "data/models/v_650.json"       # M10 leaf V (base-7)
V_NEW = "data/models/v_650_v2.json"    # pilot leaf V (base-7 + T3)


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, x))


def value_features_v2_rooted(obs: ParsedObservation, cards, root_index):
    """base-7 + prose_threat, from the FIXED root seat. base-7 is identical to
    search_bc.value_features (so arena 1 vs arena 2 differ only by prose_threat)."""
    state = obs.current
    if state is None:
        return None
    gs = GameState.build_for(obs, cards, root_index)
    me, opp = gs.me, gs.opponent
    if me is None or opp is None:
        return None
    prize = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    reserve = _clip1((gs.reserve_attackers(me) - gs.reserve_attackers(opp)) / 5.0)
    survival = _clip1((gs.my_bench_count - len(opp.bench)) / 5.0)
    on_them_s = on_me_s = 0.0
    if gs.opp_active is not None and gs.opp_active.hp > 0:
        on_them_s = min(1.0, gs.max_threat(gs.my_active, gs.opp_active, extra_energy=1) / gs.opp_active.hp)
    if gs.my_active is not None and gs.my_active.hp > 0:
        on_me_s = min(1.0, gs.max_threat(gs.opp_active, gs.my_active, extra_energy=1) / gs.my_active.hp)
    threat = on_them_s - on_me_s
    energy = _clip1((_board_energy(me) - _board_energy(opp)) / 10.0)
    hand = _clip1((me.handCount - opp.handCount) / 10.0)
    deck_out = _deckout_risk(opp.deckCount) - _deckout_risk(me.deckCount)
    on_them_p = on_me_p = 0.0
    if gs.opp_active is not None and gs.opp_active.hp > 0:
        on_them_p = min(1.0, _prose_max_threat(gs, gs.my_active, gs.opp_active, me) / gs.opp_active.hp)
    if gs.my_active is not None and gs.my_active.hp > 0:
        on_me_p = min(1.0, _prose_max_threat(gs, gs.opp_active, gs.my_active, opp) / gs.my_active.hp)
    prose_threat = (on_them_p - on_me_p) - threat
    return [prize, threat, reserve, survival, energy, hand, deck_out, prose_threat]


class LearnedEvaluatorV2(search_bc.LearnedEvaluator):
    """Leaf V on base-7 + T3 (root-seat features). Same [-1,1] mapping + terminal
    short-circuit as the M10 evaluator; only the feature vector differs."""

    n_features = 8   # base-7 + prose_threat; the base class checks the payload matches

    def value(self, obs: ParsedObservation, root_index: int) -> float:
        state = obs.current
        if state is None:
            return 0.0
        if state.result != -1:
            if state.result == root_index:
                return 1.0
            if state.result == 1 - root_index:
                return -1.0
            return 0.0
        feats = value_features_v2_rooted(obs, self._cards, root_index)
        if feats is None:
            return 0.0
        z = self._b + sum(
            w * (f - m) / s for w, f, m, s in zip(self._w, feats, self._mean, self._std)
        )
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return 2.0 * p - 1.0


class ImitationSearchPolicyV2(ImitationSearchPolicy):
    """M10 search policy with the base-7+T3 leaf V swapped in (everything else identical)."""

    name = "imitation_search_v2"

    def __init__(self, weights, v_model, **kw):
        super().__init__(weights, v_model, **kw)
        model = v_model if isinstance(v_model, dict) else json.loads(Path(v_model).read_text(encoding="utf-8"))
        self._evaluator = LearnedEvaluatorV2(model, kw.get("cards"))


DECK_MIRROR = ("v1-BC(mirror)", DECK, W_V1)  # (label, deck, weights) for the mirror opponent

# Fase B opponents (archetype-diverse); v1 baselines from rl_selfplay.V1_BASELINE (n=300).
OPP_B = {
    "lucario":   ("decks/lucario800.csv", "data/models/bc_800_v2.json", 0.320),
    "kenn":      ("decks/kenn2439.csv",   "data/models/bc_940_v1.json", 0.783),
    "cinderace": ("decks/yoshiki.csv",    "data/models/bc_cinderace_probe.json", 0.367),
}


def _base_config():
    return load_config(profile="benchmark")


def _arena(label, leaf_v, gb, n, seed, opp, sdk, cards, base_cfg, use_v1_eval=False):
    """One arena: search(leaf_v, gb, seed) on the TR deck vs `opp`=(label, deck, weights).
    use_v1_eval=True runs the M10 base-7 evaluator (search-v1) for the controlled A/B."""
    _, opp_deck, opp_w = opp
    config = dataclasses.replace(base_cfg, paths=dataclasses.replace(
        base_cfg.paths, deck_path=Path(DECK), opponent_deck_path=Path(opp_deck)))
    da = load_deck(Path(DECK)); dids = da.as_list()
    db = load_deck(Path(opp_deck)); odids = db.as_list()
    scfg = SearchConfig(greedy_bias=gb)
    if use_v1_eval:
        search_inner = ImitationSearchPolicy(W_V1, V_OLD, deck=dids, cards=cards, api=sdk.api,
                                             search_cfg=scfg, rng_seed=seed)
    else:
        search_inner = ImitationSearchPolicyV2(W_V1, leaf_v, deck=dids, cards=cards, api=sdk.api,
                                               search_cfg=scfg, rng_seed=seed)
    pol_a = SafePolicy(search_inner, deck=dids, seed=1)
    pol_b = SafePolicy(ImitationPolicy(opp_w, deck=odids), deck=odids, seed=2)
    agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    t0 = time.perf_counter()
    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start(); pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(agent_a, agent_b, dids, odids)
        else:
            rec = runner.run(agent_b, agent_a, odids, dids)
        stats.add(rec.outcome.winner, a_played_as=a_side)
    wall = time.perf_counter() - t0
    lo, hi = stats.wilson_interval()
    inner = pol_a._inner
    print(f"\n=== {label} vs {opp[0]} (n={n}, gb={gb}, seed={seed}, swapped) ===")
    print(f"A: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}")
    print(f"  searched={inner.searched} fallbacks={inner.fallbacks}  wall={wall:.0f}s ({wall/n:.1f}s/game)")
    return stats.score_rate, lo, hi


def cmd_ab(n, gb):
    """Original controlled A/B (M17 Fase 3 repro): search-v1 vs search-v2, both vs v1-BC mirror."""
    sdk = load_sdk(_base_config().paths.sdk_dir); cards = CardDatabase.from_sdk(sdk)
    base = _base_config()
    r_old, lo_o, hi_o = _arena("search-v1(v_650)", V_OLD, gb, n, 1, DECK_MIRROR, sdk, cards, base, use_v1_eval=True)
    r_new, lo_n, hi_n = _arena("search-v2(v_650_v2)", V_NEW, gb, n, 1, DECK_MIRROR, sdk, cards, base)
    print(f"\n{'='*60}\nsearch-v1: {r_old:.3f} [{lo_o:.3f},{hi_o:.3f}]   "
          f"search-v2: {r_new:.3f} [{lo_n:.3f},{hi_n:.3f}]   T3 effect: {r_new-r_old:+.3f}")


def cmd_sweepA(n):
    """Fase A: matrix over (leaf V, greedy_bias), all vs the v1-BC mirror, one process.
    Winner = best mirror score (tiebreak: cheaper). Gate A: best >= 0.52."""
    sdk = load_sdk(_base_config().paths.sdk_dir); cards = CardDatabase.from_sdk(sdk)
    base = _base_config()
    V_RL = "data/models/v_rl_v2.json"
    matrix = [
        ("v_650_v2 @gb0.05", V_NEW, 0.05),   # M17 Fase 3 reference (~0.50)
        ("v_rl_v2  @gb0.05", V_RL, 0.05),    # free better-AUC V swap
        ("v_rl_v2  @gb0.02", V_RL, 0.02),    # more override of BC
        ("v_rl_v2  @gb0.10", V_RL, 0.10),    # less override of BC
    ]
    results = []
    for label, v, gb in matrix:
        r, lo, hi = _arena(label, v, gb, n, 1, DECK_MIRROR, sdk, cards, base)
        results.append((label, v, gb, r, lo, hi))
    print(f"\n{'='*68}\nFASE A sweep (mirror vs v1-BC, n={n}):")
    for label, v, gb, r, lo, hi in results:
        print(f"  {label:<20} {r:.3f}  [{lo:.3f},{hi:.3f}]")
    best = max(results, key=lambda x: x[3])
    print(f"WINNER: {best[0]}  ({best[3]:.3f})   GATE A (>=0.52): "
          f"{'PASS -> Fase B' if best[3] >= 0.52 else 'FAIL (K-A: search ties, does not beat BC)'}")
    print(f"  best config: leaf V={best[1]}  gb={best[2]}")


def cmd_oppB(n, leaf_v, gb):
    """Fase B: winning config vs the 3 archetype-diverse opponents; pooled delta vs V1_BASELINE."""
    sdk = load_sdk(_base_config().paths.sdk_dir); cards = CardDatabase.from_sdk(sdk)
    base = _base_config()
    rows, deltas = [], []
    for name, (deck, w, v1base) in OPP_B.items():
        r, lo, hi = _arena(f"search[{Path(leaf_v).stem}@{gb}]", leaf_v, gb, n, 1, (name, deck, w), sdk, cards, base)
        rows.append((name, r, lo, hi, v1base, r - v1base)); deltas.append(r - v1base)
    print(f"\n{'='*68}\nFASE B (search vs archetypes, n={n}, leaf V={leaf_v}, gb={gb}):")
    print(f"{'opponent':<12}{'search':>8}{'v1 base':>9}{'delta':>8}")
    for name, r, lo, hi, v1base, d in rows:
        print(f"{name:<12}{r:>8.3f}{v1base:>9.3f}{d:>+8.3f}")
    pooled = sum(deltas) / len(deltas)
    print(f"pooled delta vs v1: {pooled:+.3f}   GATE B (>=+0.05): "
          f"{'PASS -> Fase C (ship)' if pooled >= 0.05 else ('DECISION 0..+0.05' if pooled >= 0 else 'FAIL (K-B)')}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if mode == "ab":
        cmd_ab(int(sys.argv[2]) if len(sys.argv) > 2 else 40, float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
    elif mode == "sweepA":
        cmd_sweepA(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif mode == "oppB":
        cmd_oppB(int(sys.argv[2]) if len(sys.argv) > 2 else 60,
                 sys.argv[3] if len(sys.argv) > 3 else V_NEW,
                 float(sys.argv[4]) if len(sys.argv) > 4 else 0.05)
    else:
        raise SystemExit(f"unknown mode {mode!r} (ab|sweepA|oppB)")
