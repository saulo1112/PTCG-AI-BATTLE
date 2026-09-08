"""M28 diagnosis of imitation-kanga-nsr's REAL ladder play (replays/54988276). READ-ONLY.

nsr = the M27 forced-recovery bundle (Night Stretcher swap + the recover_rule pre-empt),
sitting ~900 on the ladder (+~37 over the plain kanga). This answers, from real games:
  1. record + loss shape — did the 55% bench-out rate (M26) drop?
  2. bundle parity — is the ladder agent the nsr bundle? (rules out a serving bug)
  3. DID THE RULE FIRE? — re-score every MAIN decision with the nsr policy and count real
     forced-recovery pre-empts, split won/lost + per game; NS play rate overall.
  4. loss archetypes — do Alakazam / Mega Lucario / Archaludon still dominate?

Reuses diagnose_kanga.record / loss_reason / ARCHETYPE_MARKERS and the probe's policy.
Run:  uv run --group dev python scratchpad/diagnose_nsr.py
"""

from __future__ import annotations

import collections
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.kaggle_replay import iter_player_decisions
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

import diagnose_kanga as dk
from probe_nightstretcher import InstrumentedImitation

NSR_FOLDER = Path("replays/54988276")
TECH_DECK = "decks/kanga_nightstretcher.csv"
NSR_WEIGHTS = "data/models/bc_kangaskhan_1052_ns.json"
OUR = dk.OUR


def _pct(a, b):
    return f"{a}/{b} = {a/max(b,1):.1%}"


def main() -> int:
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    print("=" * 74)
    print(f"FASE 0 — record & loss shape  [{NSR_FOLDER}]")
    r = dk.record(NSR_FOLDER)
    w, l = r["wins"], r["losses"]
    print(f"  {w}W-{l}L  (WR {w/max(w+l,1):.1%})  (+{r['selfmatch']} self)")
    print(f"  loss reasons: {dict(r['loss_reasons'])}")
    lr = r["loss_reasons"]
    benchout = lr.get("bench-out", 0)
    print(f"  >> bench-out share of losses: {_pct(benchout, l)}   (M26 Kanga baseline: 30/55 = 55%)")
    print(f"  prize margin (our prizes still needed at loss): {dict(sorted(r['prize_margin'].items()))}")
    print(f"  loss archetypes: {dict(r['loss_arch'])}")
    orl, orw = r["opp_rating_on_loss"], r["opp_rating_on_win"]
    if orl or orw:
        aw = sum(orw)/len(orw) if orw else float("nan")
        al = sum(orl)/len(orl) if orl else float("nan")
        print(f"  opp live rating: avg on WIN {aw:.0f} (n={len(orw)}) | on LOSS {al:.0f} (n={len(orl)})")

    print("\n" + "=" * 74)
    print("FASE 1 — bundle parity + DID THE RULE FIRE? (re-score with the nsr policy)")
    deck = [int(x) for x in Path(TECH_DECK).read_text(encoding="utf-8").split()]
    pol = InstrumentedImitation(NSR_WEIGHTS, deck=deck)
    learned = set(pol._weights.keys())
    print(f"  recover_rule: recover_id={pol._recover_id} clock_id={pol._clock_id}  "
          f"| learned contexts: {sorted(learned)}")

    parity_ok = parity_tot = 0
    fires_won = fires_lost = 0
    games_fired: set[str] = set()
    games_seen: set[str] = set()
    fire_when_active_dying = 0  # heuristic colour: was our active about to die when we recovered
    prev_recover = 0

    files = sorted(p for p in NSR_FOLDER.glob("*.json") if p.name != "metadata.json")
    for f in files:
        for dec in iter_player_decisions(f, OUR):
            games_seen.add(dec.game_id)
            ctxname = SelectContextKind(dec.context).name
            if ctxname not in learned:
                continue
            obs = parser.parse(dec.raw_observation)
            if obs.select is None or obs.current is None or len(obs.select.option) < 2:
                continue
            ctx = DecisionContext(raw=dec.raw_observation, observation=obs, cards=cards)
            try:
                pick = set(pol.choose(ctx))
            except Exception:
                continue
            parity_tot += 1
            if pick == set(dec.action):
                parity_ok += 1
            if pol._recover_used > prev_recover:   # the rule fired on THIS decision
                prev_recover = pol._recover_used
                games_fired.add(dec.game_id)
                if dec.won:
                    fires_won += 1
                else:
                    fires_lost += 1

    total_fires = pol._recover_used
    print(f"  MAIN-family parity (live action == nsr pick): {_pct(parity_ok, parity_tot)}")
    print(f"  bc_used={pol._bc_used}  bc_failures={pol._bc_failures}")
    print(f"\n  >> forced-recovery pre-empts FIRED: {total_fires}  in {len(games_fired)} distinct games "
          f"of {len(games_seen)}  ({len(games_fired)/max(len(games_seen),1):.0%} of games)")
    print(f"     fires in WON games: {fires_won}   fires in LOST games: {fires_lost}")
    print(f"  NS (1097) selected across all decisions: {pol.dec_played}  "
          f"(offered {pol.dec_offered}, {pol.dec_played/max(pol.dec_offered,1):.0%} of offers)")

    # win rate among games where the rule fired vs not (colour on whether recovery correlates with wins)
    print("\n" + "=" * 74)
    print("FASE 2 — did recovery correlate with winning? (per-game)")
    game_won: dict[str, bool] = {}
    for f in files:
        for dec in iter_player_decisions(f, OUR):
            game_won[dec.game_id] = dec.won
    fired_games = games_fired
    nonfired = set(game_won) - fired_games
    def wr(gs):
        wins = sum(1 for g in gs if game_won.get(g))
        return f"{wins}/{len(gs)} = {wins/max(len(gs),1):.0%}"
    print(f"  games where the rule fired:     WR {wr(fired_games)}")
    print(f"  games where it never fired:     WR {wr(nonfired)}")
    print("  (colour only, not causal — fires happen in the hard, clock-lost spots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
