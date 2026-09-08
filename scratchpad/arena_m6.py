"""M6 arena: search pilot (rung 5) vs the shipped v5 config (greedy+sample).

Sequential, swapped sides, Wilson CI. Reports score rate + end-reason
breakdown + search diagnostics (mean/p95 decision time on the search side,
greedy-fallback rate inside SearchPolicy, decisions searched, SafePolicy
interventions). Ship gate G1: search(sample) vs greedy(sample) score >= 0.65
with the 95% Wilson lower bound clearing 0.5.

Modes:
  g1    search(sample) vs greedy(sample)          [the ship gate]
  g2    search(sample) vs meta decks (Lucario, Dragapult)   [no-regression]
  g3    search(deck)   vs search(sample)          [deck gauntlet under search]
  smoke tiny n, sanity only
"""

from __future__ import annotations

import collections
import dataclasses
import sys
import time
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.evaluator import EvalWeights
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.rule_based import RuleBasedPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.decision.search import SearchConfig, SearchPolicy

# Draw/setup abilities safe to always-fire for the Psychic-ex deck:
# Kangaskhan Run Errand (756), Fezandipiti Flip the Script (140),
# Meowth Last-Ditch Catch (1071), + Ogerpon Teal Dance (96).
PSY_ABILITIES = frozenset({756, 140, 1071, 96})
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.evaluation.metrics import MatchStats

REASON = {1: "prizes", 2: "deck-out", 3: "no-active", 4: "card-effect", None: "?"}
SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"
LUC = "decks/meta_lucario.csv"
DRAG = "decks/meta_dragapult.csv"
OGER = "decks/ogerpon_grass.csv"
V2 = "decks/greedy_water_v2.csv"
VIBE = "decks/vibechu.csv"      # the #1 player's actual Slowking engine deck
MAJK = "decks/majkel.csv"       # the #2 player's actual deck
PSY = "decks/psychic_ex.csv"    # v1 all-ex (failed: too slow)
PSYE = "decks/psychic_engine.csv"  # v2 cheap-attacker + Fezandipiti engine
SLAK = "decks/slaking_wall.csv"    # better-wall: Slaking ex (280/2col/2pz) vs sample's Mega Aboma (200/3/3pz)

_SDK = None
_CARDS = None


def _mk(kind, deck_ids, seed, cards, api, cfg, weights=None):
    if kind == "search":
        inner = SearchPolicy(deck=deck_ids, cards=cards, api=api, search_cfg=cfg,
                             weights=weights, rng_seed=seed)
    elif kind == "greedy":
        inner = GreedyPolicy(deck=deck_ids)
    elif kind == "rule":
        inner = RuleBasedPolicy(deck=deck_ids, ability_whitelist=PSY_ABILITIES)
    else:
        raise ValueError(kind)
    return SafePolicy(inner, deck=deck_ids, seed=seed)


def run(a_kind, deck_a, b_kind, deck_b, label, n=200, cfg=None, weights=None):
    global _SDK, _CARDS
    base = load_config(profile="benchmark")
    config = dataclasses.replace(base, paths=dataclasses.replace(
        base.paths, deck_path=Path(deck_a), opponent_deck_path=Path(deck_b)))
    if _SDK is None:
        _SDK = load_sdk(config.paths.sdk_dir)
        _CARDS = CardDatabase.from_sdk(_SDK)
    sdk, cards = _SDK, _CARDS
    cfg = cfg or SearchConfig()
    da, db = load_deck(Path(deck_a)), load_deck(Path(deck_b))
    pol_a = _mk(a_kind, da.as_list(), 1, cards, sdk.api, cfg, weights)
    pol_b = _mk(b_kind, db.as_list(), 2, cards, sdk.api, cfg)
    agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
    agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
    stats = MatchStats()
    wr, lr = collections.Counter(), collections.Counter()
    a_inner = pol_a._inner
    searched = fallbacks = 0
    t_start = time.perf_counter()

    for game in range(n):
        a_side = game % 2
        pol_a.on_battle_start()
        pol_b.on_battle_start()
        if a_side == 0:
            rec = runner.run(agent_a, agent_b, da.as_list(), db.as_list())
        else:
            rec = runner.run(agent_b, agent_a, db.as_list(), da.as_list())
        stats.add(rec.outcome.winner, a_played_as=a_side)
        if rec.outcome.winner == a_side:
            wr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
        elif rec.outcome.winner is not None:
            lr[REASON.get(rec.outcome.reason, rec.outcome.reason)] += 1
        if isinstance(a_inner, SearchPolicy):
            searched += a_inner.searched
            fallbacks += a_inner.fallbacks

    wall = time.perf_counter() - t_start
    lo, hi = stats.wilson_interval()
    print(f"\n=== {label} (n={n}, swapped) ===")
    print(f"A[{a_kind}]={Path(deck_a).stem}   B[{b_kind}]={Path(deck_b).stem}")
    print(f"A: {stats.summary()}   clears0.5? {'YES' if lo > 0.5 else 'no'}  "
          f">=0.65(lo>0.5)? {'SHIP' if (lo > 0.5 and stats.score_rate() >= 0.65) else 'no'}")
    print(f"  A wins:{dict(wr)}  A loses:{dict(lr)}")
    print(f"  interventions A={pol_a.interventions} B={pol_b.interventions}")
    if searched + fallbacks:
        print(f"  search: searched={searched} fallbacks={fallbacks} "
              f"fallback_rate={100*fallbacks/(searched+fallbacks):.1f}%  "
              f"wall={wall:.0f}s ({wall/n:.2f}s/game)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "smoke":
        run("search", SAMPLE, "greedy", SAMPLE, "search(sample) vs greedy(sample)", n=n or 20)
    elif mode == "g1":
        run("search", SAMPLE, "greedy", SAMPLE, "G1 search(sample) vs greedy(sample)=v5", n=n or 300)
    elif mode == "g2":
        run("search", SAMPLE, "greedy", LUC, "G2 search(sample) vs Lucario", n=n or 200)
        run("search", SAMPLE, "greedy", DRAG, "G2 search(sample) vs Dragapult", n=n or 200)
    elif mode == "g3":
        run("search", OGER, "search", SAMPLE, "G3 search(ogerpon) vs search(sample)", n=n or 200)
        run("search", V2, "search", SAMPLE, "G3 search(water_v2) vs search(sample)", n=n or 200)
        run("search", LUC, "search", SAMPLE, "G3 search(lucario) vs search(sample)", n=n or 200)
    elif mode == "tune":
        # Screen weight variants vs greedy(sample) at n=100 (Wilson +-0.10).
        nn = n or 100
        variants = {
            "default": EvalWeights(),
            "prize-heavy": EvalWeights(prize=0.55, threat=0.16, reserve=0.10,
                                       survival=0.10, energy=0.04, hand=0.03, deck_out=0.02),
            "survival-heavy": EvalWeights(prize=0.34, threat=0.22, reserve=0.16,
                                          survival=0.18, energy=0.05, hand=0.03, deck_out=0.02),
            "threat-heavy": EvalWeights(prize=0.34, threat=0.32, reserve=0.12,
                                        survival=0.12, energy=0.05, hand=0.03, deck_out=0.02),
        }
        for name_, w in variants.items():
            run("search", SAMPLE, "greedy", SAMPLE, f"tune[{name_}] vs greedy(sample)", n=nn, weights=w)
    elif mode == "bias":
        # Does trusting greedy unless clearly-better recover >=0.50?
        # bias=5.0 == always greedy (sanity floor, expect ~0.50).
        nn = n or 100
        for b in (5.0, 0.30, 0.15, 0.08):
            run("search", SAMPLE, "greedy", SAMPLE, f"bias[{b}] vs greedy(sample)",
                n=nn, cfg=SearchConfig(greedy_bias=b))
    elif mode == "slaksmoke":
        run("greedy", SLAK, "greedy", SAMPLE, "smoke greedy(slaking) vs greedy(sample)", n=n or 24)
    elif mode == "slaking":
        # Better-wall in the pilot's proven archetype. Bars: beat v5's meta
        # numbers (Lucario 0.60, Dragapult 0.94) and ideally greedy(sample).
        nn = n or 200
        run("greedy", SLAK, "greedy", SAMPLE, "greedy(slaking) vs greedy(sample)=v5", n=nn)
        run("greedy", SLAK, "greedy", LUC, "greedy(slaking) vs Lucario [v5 bar 0.60]", n=nn)
        run("greedy", SLAK, "greedy", DRAG, "greedy(slaking) vs Dragapult [v5 bar 0.94]", n=nn)
        run("rule", SLAK, "greedy", SAMPLE, "rule(slaking) vs greedy(sample) [pilot+]", n=nn)
    elif mode == "psysmoke":
        run("rule", PSYE, "greedy", SAMPLE, "smoke rule(psychic_engine) vs greedy(sample)", n=n or 20)
    elif mode == "psy2":
        # v2 cheap-attacker deck, judged vs the META (the unbiased gate).
        # Bar to beat = greedy(sample)'s meta numbers: Lucario 0.60, Dragapult 0.94.
        nn = n or 200
        run("rule", PSYE, "greedy", LUC, "rule(psy_engine) vs Lucario [v5 bar 0.60]", n=nn)
        run("rule", PSYE, "greedy", DRAG, "rule(psy_engine) vs Dragapult [v5 bar 0.94]", n=nn)
        run("rule", PSYE, "greedy", SAMPLE, "rule(psy_engine) vs greedy(sample) [ref]", n=nn)
        run("greedy", PSYE, "greedy", LUC, "greedy(psy_engine) vs Lucario [ability ablation]", n=nn)
    elif mode == "psychic":
        # THE user-chosen test: Basic-ex draw-engine deck + ability pilot vs v5.
        nn = n or 200
        run("rule", PSY, "greedy", SAMPLE, "rule(psychic_ex) vs greedy(sample)=v5", n=nn)
        run("greedy", PSY, "greedy", SAMPLE, "greedy(psychic_ex) vs greedy(sample) [ablation: no abilities]", n=nn)
        run("rule", PSY, "greedy", LUC, "rule(psychic_ex) vs Lucario", n=nn)
        run("rule", PSY, "greedy", DRAG, "rule(psychic_ex) vs Dragapult", n=nn)
    elif mode == "vibechu":
        # THE top-player-deck test. Does the #1 deck beat our v5 config, under
        # (a) our simple greedy pilot [engine won't run — abilities unused] and
        # (b) the search pilot [fires Slowking's ability by simulation]?
        nn = n or 150
        run("greedy", VIBE, "greedy", SAMPLE, "greedy(vibechu) vs greedy(sample)=v5", n=nn)
        run("search", VIBE, "greedy", SAMPLE, "search(vibechu,bias0) vs greedy(sample)=v5",
            n=nn, cfg=SearchConfig(greedy_bias=0.0))
        run("search", VIBE, "greedy", SAMPLE, "search(vibechu,bias.1) vs greedy(sample)=v5",
            n=nn, cfg=SearchConfig(greedy_bias=0.1))
    elif mode == "vibesmoke":
        run("search", VIBE, "greedy", SAMPLE, "smoke search(vibechu) vs greedy(sample)", n=n or 12)
    elif mode == "decisive":
        nn = n or 150
        # 1) sanity: always-greedy must floor at ~0.50 (rules out a harness bug).
        run("search", SAMPLE, "greedy", SAMPLE, "sanity always-greedy(bias5) vs greedy(sample)",
            n=100, cfg=SearchConfig(greedy_bias=5.0))
        # 2+3) the real bet: does search UNLOCK Ogerpon's ability engine to beat v5?
        run("search", OGER, "greedy", SAMPLE, "search(ogerpon,bias0) vs greedy(sample)=v5",
            n=nn, cfg=SearchConfig(greedy_bias=0.0))
        run("search", OGER, "greedy", SAMPLE, "search(ogerpon,bias.15) vs greedy(sample)=v5",
            n=nn, cfg=SearchConfig(greedy_bias=0.15))
