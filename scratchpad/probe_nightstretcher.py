"""M27 de-risk probe for the Night Stretcher (1097) resilience swap. READ-ONLY.

Answers the two halves the head_to_head tie cannot:

  HALF 1 — OPPORTUNITY (from Kanga's real logs): how often is there actually a
    recoverable key Pokemon (Mega Kangaskhan 756 / Crustle 345 / Dwebble 344) sitting
    in OUR discard during a MAIN turn — i.e. would Night Stretcher have a target — split
    by won/lost games and by loss reason (bench-out is the mode M26 named). The original
    deck has no 1097, so the logs cannot show 1097 being PLAYED; they can show whether the
    OPPORTUNITY to recover exists. Bounds the ceiling of the tech.

  HALF 2 — BEHAVIOR (simulation with the tech clone): run the actual tech bundle
    (KANGASKHAN_1052 weights + the 1097 deck) vs a sample of the greedy field and count,
    with an instrumented policy, how often a 'play 1097' option is OFFERED and how often
    the learned MAIN scorer actually SELECTS it. This is the untunable generic-prior play
    rate from PASO 2, measured directly.

Run:  uv run --group dev python scratchpad/probe_nightstretcher.py [n_field_decks] [n_games]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import Deck, load_deck
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.evaluation.metrics import MatchStats
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

from field_gauntlet import extract_field

# The live Kanga replays (replays/54911514, 54893233) are gitignored and not on disk this
# session. The TEACHER's own dataset — the exact same Kangaskhan+Crustle deck, 461 games —
# is the right opportunity proxy: M26 showed the clone matches the teacher on every
# behavioral axis (bench depth, aggression), so its discard composition ~ the clone's.
TEACHER_DS = Path("data/imitation/懒惰的金枪鱼_screen.jsonl.gz")

NS = 1097                       # Night Stretcher
KEY_ATTACKERS = {756: "Kangaskhan", 345: "Crustle"}
RECOVERABLE_POKEMON = {756, 345, 344}
TECH_DECK = "decks/kanga_nightstretcher.csv"
WEIGHTS = "data/models/bc_kangaskhan_1052.json"


# =========================== HALF 1 — opportunity ============================

def opportunity() -> None:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    # per outcome (won/lost): MAIN turns, turns with any recoverable Pokemon in discard,
    # turns with a KEY attacker in discard; per game: did a key attacker ever hit discard.
    agg = collections.defaultdict(lambda: {
        "main": 0, "any_target": 0, "key_target": 0,
        "games": 0, "games_key_ever": 0,
    })
    # per-game accumulation so we can flag "a key attacker was in discard at some MAIN turn"
    game_seen: dict[str, dict] = {}

    for dec in read_decision_dataset(TEACHER_DS):
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        raw = getattr(dec, "raw_observation", None)
        if not raw:
            continue
        obs = parser.parse(raw)
        if obs.current is None:
            continue
        gs = GameState.build(obs, cards)
        me = gs.me
        if me is None:
            continue
        bucket = "WON" if dec.won else "LOST"
        disc = collections.Counter(c.id for c in me.discard)
        any_pkmn = any(disc.get(i, 0) for i in RECOVERABLE_POKEMON)
        key = any(disc.get(i, 0) for i in KEY_ATTACKERS)
        a = agg[bucket]
        a["main"] += 1
        a["any_target"] += int(any_pkmn)
        a["key_target"] += int(key)
        g = game_seen.setdefault(dec.game_id, {"bucket": bucket, "key_ever": False})
        g["key_ever"] = g["key_ever"] or key

    for gid, g in game_seen.items():
        agg[g["bucket"]]["games"] += 1
        agg[g["bucket"]]["games_key_ever"] += int(g["key_ever"])

    print("=" * 74)
    print("HALF 1 — RECOVERY OPPORTUNITY (teacher dataset, exact same deck, 461 games)")
    print("  a MAIN turn 'has a target' if a recoverable Pokemon is in OUR discard;")
    print("  'key target' = a Mega Kangaskhan or Crustle is in discard.\n")
    order = sorted(agg.keys())
    hdr = f"{'bucket':<20}{'games':>7}{'MAIN turns':>12}{'anyTarget%':>12}{'keyTarget%':>12}{'games w/key':>13}"
    print(hdr)
    for b in order:
        a = agg[b]
        print(f"{b:<20}{a['games']:>7}{a['main']:>12}"
              f"{a['any_target']/max(a['main'],1):>11.1%}"
              f"{a['key_target']/max(a['main'],1):>11.1%}"
              f"{a['games_key_ever']}/{a['games']:>1} ({a['games_key_ever']/max(a['games'],1):.0%})".rjust(13))


# =========================== HALF 2 — behavior ===============================

class InstrumentedImitation(ImitationPolicy):
    """Counts, per decision, whether a 'play 1097' option was offered and whether the
    learned scorer selected it; also per-game offered/played flags."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dec_offered = 0          # decisions where NS was a legal option
        self.dec_played = 0           # decisions where NS was in the chosen set
        self.main_offered = 0         # of those, in MAIN context
        self.main_played = 0
        self.games = 0
        self.games_offered = 0
        self.games_played = 0
        self._g_off = False
        self._g_play = False

    def on_battle_start(self) -> None:  # type: ignore[override]
        super().on_battle_start()
        self.games += 1
        self.games_offered += int(self._g_off)
        self.games_played += int(self._g_play)
        self._g_off = False
        self._g_play = False

    def choose(self, ctx: DecisionContext):
        chosen = super().choose(ctx)
        select = ctx.observation.select
        if select is not None and select.option:
            ns_idx = set()
            for i, opt in enumerate(select.option):
                try:
                    r = resolve_option(opt, ctx.observation)
                except Exception:
                    continue
                if r.card_id == NS and opt.type in (OptionKind.PLAY, OptionKind.CARD):
                    ns_idx.add(i)
            if ns_idx:
                self.dec_offered += 1
                self._g_off = True
                is_main = select.context is SelectContextKind.MAIN
                self.main_offered += int(is_main)
                if ns_idx & set(chosen):
                    self.dec_played += 1
                    self._g_play = True
                    self.main_played += int(is_main)
        return chosen


def behavior(n_field: int, n_games: int, weights: str = WEIGHTS) -> None:
    base = load_config(profile="benchmark")
    sdk = load_sdk(base.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    da = load_deck(Path(TECH_DECK))
    field = extract_field(cards)[:n_field]

    pol_a = InstrumentedImitation(weights, deck=da.as_list())
    print(f"  [weights={weights}  recover_rule={'ON' if pol_a._recover_id else 'off'}]")
    safe_a = SafePolicy(pol_a, deck=da.as_list(), seed=1)
    stats = MatchStats()

    import dataclasses
    for dname, dids in field:
        db = Deck(card_ids=dids)
        config = dataclasses.replace(base, paths=dataclasses.replace(
            base.paths, deck_path=Path(TECH_DECK), opponent_deck_path=Path(TECH_DECK)))
        pol_b = SafePolicy(GreedyPolicy(deck=db.as_list()), deck=db.as_list(), seed=2)
        agent_a = PTCGAgent(safe_a, deck=da, cards=cards)
        agent_b = PTCGAgent(pol_b, deck=db, cards=cards)
        env = BattleEnvironment(config, sdk=sdk)
        runner = BattleRunner(env, max_decisions=config.battle.max_decisions)
        for game in range(n_games):
            a_side = game % 2
            safe_a.on_battle_start(); pol_b.on_battle_start()
            if a_side == 0:
                rec = runner.run(agent_a, agent_b, da.as_list(), db.as_list())
            else:
                rec = runner.run(agent_b, agent_a, db.as_list(), da.as_list())
            stats.add(rec.outcome.winner, a_played_as=a_side)
    # flush the final game's flags
    pol_a.games_offered += int(pol_a._g_off)
    pol_a.games_played += int(pol_a._g_play)

    print("\n" + "=" * 74)
    print(f"HALF 2 — NS PLAY FREQUENCY (tech clone, {len(field)} field decks x {n_games} games "
          f"= {len(field)*n_games} games, WR {stats.score_rate:.3f}, interventions {safe_a.interventions})")
    print(f"  games where NS was ever OFFERED: {pol_a.games_offered}/{pol_a.games} "
          f"({pol_a.games_offered/max(pol_a.games,1):.0%})")
    print(f"  games where NS was ever PLAYED : {pol_a.games_played}/{pol_a.games} "
          f"({pol_a.games_played/max(pol_a.games,1):.0%})")
    print(f"  decisions NS offered: {pol_a.dec_offered}  |  NS selected: {pol_a.dec_played} "
          f"({pol_a.dec_played/max(pol_a.dec_offered,1):.0%} of offers taken)")
    print(f"  of which MAIN-context: offered {pol_a.main_offered}, selected {pol_a.main_played} "
          f"({pol_a.main_played/max(pol_a.main_offered,1):.0%})")
    print(f"  forced-recovery pre-empts fired: {pol_a._recover_used}")


def main(argv: list[str]) -> int:
    n_field = int(argv[0]) if len(argv) > 0 else 25
    n_games = int(argv[1]) if len(argv) > 1 else 6
    weights = argv[2] if len(argv) > 2 else WEIGHTS
    opportunity()
    behavior(n_field, n_games, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
