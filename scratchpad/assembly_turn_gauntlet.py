"""M34 Track D — the missing instrument: does a candidate ASSEMBLE FASTER, measured
in simulated games rather than on static states?

Every offline gate this project has trusted measured FIDELITY (accuracy on the
teacher's logged states) and it has now mispredicted the ladder seven times
(M11/M14/M31/M32). The head-to-head win-rate gate measures behaviour but, per
M14's calibration, cannot resolve a pilot difference below ~0.15 -- it read a
known +0.13 gap as -0.003, because a win/loss is one Bernoulli sample per game.

This measures the thing M33 actually identified as the defect, and it measures it
as a CONTINUOUS per-game quantity: the turn at which Alakazam reaches the Active
spot with a {P} attached (and the turn Powerful Hand first becomes legal). A mean
over n games of a low-variance count has far more statistical power than a win
rate at the same n, and unlike the 27 real Grimmsnarl replays we control n here.

Milestones mirror scratchpad/analyze_ph_availability.py::measure exactly, so the
numbers are directly comparable to the M33 table (teacher 4.42 / clone 5.16 for
"Alakazam in play" vs Grimmsnarl).

G-3 (pre-registered): candidate mean "Alakazam active with {P}" turn must be
<= champion - 0.25, with a paired bootstrap CI over field decks excluding 0.

READ-ONLY with respect to the repo (plays games in memory, writes nothing). Run:
  uv run --group dev python scratchpad/assembly_turn_gauntlet.py \
      --candidate data/models/bc_alakazam_fetch.json \
      --baseline  data/models/bc_alakazam_mlp.json  --games 6 --decks 20
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import statistics
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import Deck, load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.state.game_state import GameState

import field_gauntlet as fg

ALAKAZAM_ID = 743
POWERFUL_HAND = 1072

MILESTONES = ("zam_in_play", "zam_active", "zam_charged", "ph_legal")


class AssemblyRecorder:
    """Wraps a policy and records, per game, the first turn of each milestone.

    Delegates every Policy method through, so it drops into the existing
    field_gauntlet plumbing untouched. Recording happens on MAIN decisions only,
    which is where the agent sees a full board each turn.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name = f"rec({getattr(inner, 'name', '?')})"
        self.last_note = None
        self.games: list[dict[str, int]] = []
        self._cur: dict[str, int] = {}

    # -- Policy protocol ----------------------------------------------------
    def choose(self, ctx: DecisionContext) -> list[int]:
        try:
            self._observe(ctx)
        except Exception:
            pass  # instrumentation must never change the game being measured
        out = self._inner.choose(ctx)
        self.last_note = getattr(self._inner, "last_note", None)
        return out

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        return self._inner.choose_deck(ctx)

    def on_battle_start(self) -> None:
        self._cur = {}
        self._inner.on_battle_start()

    def on_battle_end(self, outcome) -> None:
        self._inner.on_battle_end(outcome)

    def flush(self) -> None:
        """Close the current game's record.

        BattleRunner never invokes on_battle_start/on_battle_end (field_gauntlet
        calls on_battle_start by hand for the same reason), so the harness must
        close each game itself rather than relying on the policy lifecycle.
        """
        self.games.append(dict(self._cur))
        self._cur = {}

    def __getattr__(self, item):  # interventions, etc.
        return getattr(self._inner, item)

    # -- instrumentation ----------------------------------------------------
    def _observe(self, ctx: DecisionContext) -> None:
        obs = ctx.observation
        if obs.select is None or obs.current is None:
            return
        if obs.select.context is not SelectContextKind.MAIN:
            return
        gs = GameState.build(obs, ctx.cards)
        me = gs.me
        if me is None:
            return
        turn = obs.current.turn
        in_play = [p for p in me.active if p is not None] + list(me.bench)
        if any(p.id == ALAKAZAM_ID for p in in_play):
            self._cur.setdefault("zam_in_play", turn)
        act = gs.my_active
        if act is not None and act.id == ALAKAZAM_ID:
            self._cur.setdefault("zam_active", turn)
            if any(e is EnergyKind.PSYCHIC for e in act.energies):
                self._cur.setdefault("zam_charged", turn)
        if any(o.type is OptionKind.ATTACK and o.attackId == POWERFUL_HAND
               for o in obs.select.option):
            self._cur.setdefault("ph_legal", turn)


def run_arm(label, deck_csv, weights, field, n, sdk, cards):
    """Play the Alakazam deck vs each greedy-piloted field deck, recording assembly."""
    base = load_config(profile="benchmark")
    da = load_deck(Path(deck_csv))
    per_deck: dict[str, dict[str, float]] = {}
    for dname, dids in field:
        db = Deck(card_ids=dids)
        config = dataclasses.replace(base, paths=dataclasses.replace(
            base.paths, deck_path=Path(deck_csv), opponent_deck_path=Path(deck_csv)))
        inner = SafePolicy(ImitationPolicy(weights, deck=da.as_list()),
                           deck=da.as_list(), seed=1)
        pol_a = AssemblyRecorder(inner)
        pol_b = SafePolicy(GreedyPolicy(deck=db.as_list()), deck=db.as_list(), seed=2)
        agent_a = PTCGAgent(pol_a, deck=da, cards=cards)
        agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
        env = BattleEnvironment(config, sdk=sdk)
        runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
        for game in range(n):
            pol_a.on_battle_start(); pol_b.on_battle_start()
            if game % 2 == 0:
                runner.run(agent_a, agent_b, da.as_list(), db.as_list())
            else:
                runner.run(agent_b, agent_a, db.as_list(), da.as_list())
            pol_a.flush()
        stats = {}
        for m in MILESTONES:
            vals = [g[m] for g in pol_a.games if m in g]
            stats[m] = statistics.mean(vals) if vals else float("nan")
            stats[m + "_n"] = len(vals)
            # keep the raw per-game turns: the per-deck mean is noisy at small
            # games/deck, so the pooled bootstrap below needs the samples.
            stats[m + "_raw"] = vals
        stats["games"] = len(pol_a.games)
        # games where Alakazam NEVER assembled — M32 saw ~15% of clone losses
        # never assemble vs the teacher's 4%, so this is a first-class outcome,
        # not missing data. Excluding them silently would flatter a slow model.
        stats["never_assembled"] = sum(1 for g in pol_a.games if "zam_charged" not in g)
        per_deck[dname] = stats
        print(f"  [{label}] {dname[:28]:<30} "
              + "  ".join(f"{m}={stats[m]:.2f}({stats[m+'_n']})" for m in MILESTONES),
              flush=True)
    return per_deck


def paired_bootstrap(a, b, key, seed=0):
    """Mean per-deck (candidate - baseline) on `key`, bootstrap 90% CI."""
    keys = [k for k in a if k in b
            and a[k][key] == a[k][key] and b[k][key] == b[k][key]]  # drop NaN
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0
    diffs = [a[k][key] - b[k][key] for k in keys]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
    return (sum(diffs) / len(diffs), boots[int(0.05 * len(boots))],
            boots[int(0.95 * len(boots))], len(keys))


def pooled_bootstrap(a_vals, b_vals, seed=0, n=2000):
    """Unpaired difference of means over ALL games, with a bootstrap 90% CI."""
    if not a_vals or not b_vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    obs = sum(a_vals) / len(a_vals) - sum(b_vals) / len(b_vals)
    boots = []
    for _ in range(n):
        sa = sum(rng.choice(a_vals) for _ in a_vals) / len(a_vals)
        sb = sum(rng.choice(b_vals) for _ in b_vals) / len(b_vals)
        boots.append(sa - sb)
    boots.sort()
    return obs, boots[int(0.05 * n)], boots[int(0.95 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="data/models/bc_alakazam_mlp.json")
    ap.add_argument("--deck", default="decks/yushinito.csv")
    ap.add_argument("--games", type=int, default=6, help="games per field deck")
    ap.add_argument("--decks", type=int, default=20, help="field decks to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marker", type=int, default=None,
                    help="restrict the field to decks containing this card id (e.g. 648 = "
                         "Marnie's Grimmsnarl). The cached field is only 5%% Grimmsnarl while "
                         "the defect M33 measured is Grimmsnarl-specific (58%% of the teacher's "
                         "field, 27%% of ours), so the unrestricted run tests the fix where the "
                         "gap barely exists.")
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    sdk = load_sdk(cfg.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    field = fg.extract_field(cards)
    if args.marker is not None:
        field = [(n, ids) for n, ids in field if args.marker in ids]
        print(f"field restricted to decks containing card {args.marker}: {len(field)} decks")
    if args.decks and len(field) > args.decks:
        random.Random(args.seed).shuffle(field)
        field = field[: args.decks]
    print(f"field: {len(field)} decks x {args.games} games\n")

    print("BASELINE:")
    b = run_arm("base", args.deck, args.baseline, field, args.games, sdk, cards)
    print("\nCANDIDATE:")
    a = run_arm("cand", args.deck, args.candidate, field, args.games, sdk, cards)

    print("\n" + "=" * 78)
    print("M34 TRACK D — assembly turn in SIMULATION (lower = faster)")
    print("=" * 78)
    print(f"{'milestone':<20}{'baseline':>10}{'candidate':>11}{'delta':>9}"
          f"{'90% CI (decks)':>20}{'90% CI (games)':>20}")
    verdict_delta = verdict_hi = None
    for m in MILESTONES:
        braw = [x for v in b.values() for x in v[m + "_raw"]]
        araw = [x for v in a.values() for x in v[m + "_raw"]]
        bm = statistics.mean(braw) if braw else float("nan")
        am = statistics.mean(araw) if araw else float("nan")
        d, lo, hi, nk = paired_bootstrap(a, b, m, args.seed)
        # The engine is not seedable from our side, so the two arms never see the
        # same shuffles; pairing over DECKS removes matchup variance but not
        # within-deck draw variance. The pooled game-level bootstrap uses every
        # game and is the tighter of the two whenever games/deck is small.
        gd, glo, ghi = pooled_bootstrap(araw, braw, args.seed)
        print(f"{m:<20}{bm:>10.2f}{am:>11.2f}{gd:>+9.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}{f'[{glo:+.2f}, {ghi:+.2f}]':>20}")
        if m == "zam_charged":
            verdict_delta, verdict_hi = gd, max(hi, ghi)

    nb = sum(v["never_assembled"] for v in b.values())
    na = sum(v["never_assembled"] for v in a.values())
    gb = sum(v["games"] for v in b.values())
    ga = sum(v["games"] for v in a.values())
    print(f"\nnever assembled Alakazam+{{P}}: baseline {nb}/{gb} ({nb/max(gb,1):.1%})  "
          f"candidate {na}/{ga} ({na/max(ga,1):.1%})")
    print("  (counted separately — these games have no assembly turn, so a model that")
    print("   simply fails more often would otherwise look faster on the games it wins)")

    print(f"\nG-3 (pre-registered): 'Alakazam active with {{P}}' delta <= -0.25 "
          f"AND the 90% CI must exclude 0.")
    ok = (verdict_delta is not None and verdict_delta <= -0.25
          and verdict_hi is not None and verdict_hi < 0)
    print(f"    delta {verdict_delta:+.2f}   CI upper {verdict_hi:+.2f}   "
          f"-> {'PASS' if ok else 'FAIL'}")
    print("\nNote: this is a BEHAVIOURAL gate, not a win-rate gate. It answers "
          "'does it assemble\nfaster', which is what M33 identified. The "
          "head_to_head veto still has to run separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
