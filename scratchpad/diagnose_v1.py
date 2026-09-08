"""Phase-0 diagnosis of imitation-v1's real ladder losses (M10).

Reads Logs/Submission imitation v1/ (86 games, our name "Saulo Quiñones Góngora")
and answers "where does v1 actually lose?" so we spend the improvement budget on
the right lever. Four cuts, weighted toward the reliable ones:

  1. Record + per-opponent (reliable: from `rewards`).
  2. Lethal-miss audit (RELIABLE, needs no terminal state): on each of our MAIN
     decisions, was a KO on the opponent's active available (structured damage via
     the TR_650 profile's damage_650, which models Rocket Rush's prose damage) and
     did we take it? Flags game-LOSING misses (a miss in a game we then lost).
  3. Loss-reason inference (DIRECTION ONLY — no clean terminal obs per
     docs/replay_analysis.md; uses the winner's last stored board as the proxy).
  4. Loss opponents' archetypes (extract each winner's deck, cluster by key cards).

Run:  uv run python scratchpad/diagnose_v1.py ["Logs/Submission imitation v1.2"]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.kaggle_replay import iter_player_decisions, player_seats
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

import extract_top_decks as et

LOG_DIR = Path("Logs/Submission imitation v1")
OUR_NAME = "Saulo Quiñones Góngora"
PROFILE = get_profile("TR_650")

# key cards to name loss archetypes (id -> label)
ARCHETYPE_MARKERS = {
    678: "Mega Lucario", 675: "Lunatone/Solrock", 269: "Bellibolt",
    190: "Archaludon", 743: "Alakazam", 400: "Team Rocket", 666: "Cinderace",
}


def _files() -> list[Path]:
    return sorted(p for p in LOG_DIR.glob("*.json") if p.name != "metadata.json")


def lethal_audit(parser: ObservationParser, cards: CardDatabase) -> dict:
    """TURN-level lethal miss (the correct metric — a turn has many MAIN decisions,
    the teacher develops then attacks last, so a per-decision count over-flags).

    For each (game, turn): lethal_available = some MAIN decision that turn offered a
    KO on the opp active; lethal_taken = we actually chose a lethal ATTACK that turn.
    A miss = available AND not taken. This matches M8.1's ~3% turn-level metric."""
    # (game, turn) -> [available, taken]
    turns: dict[tuple[str, int], list[bool]] = collections.defaultdict(lambda: [False, False])
    turn_won: dict[tuple[str, int], bool] = {}
    for f in _files():
        for dec in iter_player_decisions(f, OUR_NAME):
            if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
                continue
            obs = parser.parse(dec.raw_observation)
            if obs.select is None or obs.current is None:
                continue
            gs = GameState.build(obs, cards)
            if gs.opp_active is None or gs.opp_active.hp <= 0:
                continue
            key = (dec.game_id, obs.current.turn)
            turn_won[key] = dec.won
            lethal_idx = [
                i for i, opt in enumerate(obs.select.option)
                if opt.type is OptionKind.ATTACK
                and PROFILE.damage_fn(opt.attackId, gs, cards) >= gs.opp_active.hp
            ]
            if lethal_idx:
                turns[key][0] = True
                if any(a in lethal_idx for a in dec.action):
                    turns[key][1] = True
    avail = [k for k, (a, _) in turns.items() if a]
    taken = [k for k in avail if turns[k][1]]
    missed = [k for k in avail if not turns[k][1]]
    losing_miss_turns = [k for k in missed if not turn_won.get(k, True)]
    losing_miss_games = {g for g, _ in losing_miss_turns}
    return {
        "turns_analyzed": len(turns),
        "lethal_available": len(avail),
        "lethal_taken": len(taken),
        "lethal_missed": len(missed),
        "losing_miss_turns": len(losing_miss_turns),
        "losing_miss_games": len(losing_miss_games),
    }


def loss_reason(data: dict, our_seat: int) -> tuple[str, int]:
    """Direction-only inference from the winner's last stored board.

    Returns (reason, our_prizes_left_at_end). Winner's last observation is the
    most recent snapshot; players[yourIndex]=winner, [1-yourIndex]=us (loser)."""
    steps = data.get("steps", [])
    winner_seat = 1 - our_seat
    # find the winner's last cell that carries a current observation
    for step in reversed(steps):
        if winner_seat >= len(step):
            continue
        cur = (step[winner_seat].get("observation") or {}).get("current")
        if not cur or "players" not in cur:
            continue
        yidx = cur.get("yourIndex", winner_seat)
        us = cur["players"][1 - yidx] if len(cur["players"]) > 1 else None
        win = cur["players"][yidx]
        if us is None:
            break
        our_prizes = len(us.get("prize", []))
        our_deck = us.get("deckCount", 99)
        our_active = sum(1 for a in (us.get("active") or []) if a)
        our_bench = len(us.get("bench", []))
        win_prizes = len(win.get("prize", []))
        if our_deck <= 0:
            return "deck-out", our_prizes
        if our_active == 0 and our_bench == 0:
            return "bench-out", our_prizes
        # otherwise the winner took their last prize by KO -> prize race
        return "prize-race", our_prizes
    return "unknown", 6


def main() -> int:
    global LOG_DIR
    if len(sys.argv) > 1:
        LOG_DIR = Path(sys.argv[1])
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    files = _files()
    wins = losses = selfmatch = 0
    loss_reasons: collections.Counter = collections.Counter()
    prize_margin: collections.Counter = collections.Counter()  # our prizes left at loss
    loss_archetypes: collections.Counter = collections.Counter()

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        seats = player_seats(data, OUR_NAME)
        if len(seats) != 1:
            selfmatch += 1
            continue
        our = seats[0]
        rew = (data.get("rewards") or [0, 0])[our]
        if rew > 0:
            wins += 1
            continue
        losses += 1
        reason, our_prizes = loss_reason(data, our)
        loss_reasons[reason] += 1
        prize_margin[our_prizes] += 1  # prizes WE still needed when we lost (1-2 = close)
        # archetype of the winner
        deck = et.extract_deck(data, 1 - our)
        if deck:
            labels = [lab for cid, lab in ARCHETYPE_MARKERS.items() if cid in deck]
            loss_archetypes[labels[0] if labels else "other"] += 1

    print(f"=== ladder record ({LOG_DIR}) ===")
    print(f"  {wins}W - {losses}L  (+{selfmatch} self-match excluded)  of {len(files)} games")

    print("\n=== loss reasons (DIRECTION ONLY — inferred from winner's last board) ===")
    for r, n in loss_reasons.most_common():
        print(f"  {r:<12} {n}")

    print("\n=== how close were the losses (our prizes still needed at end) ===")
    for p in sorted(prize_margin):
        tag = "  <- close" if p <= 2 else ("  <- blowout" if p >= 5 else "")
        print(f"  needed {p} more prize(s): {prize_margin[p]} losses{tag}")

    print("\n=== loss opponents by archetype ===")
    for a, n in loss_archetypes.most_common():
        print(f"  {a:<18} {n}")

    print("\n=== lethal-miss audit (RELIABLE — TURN-level, our MAIN turns) ===")
    la = lethal_audit(parser, cards)
    avail = la["lethal_available"]
    print(f"  turns w/ KO available (that damage_650 recognizes): {avail}")
    if avail:
        print(f"  KO taken:  {la['lethal_taken']} ({la['lethal_taken']/avail:.1%})")
        print(f"  KO missed: {la['lethal_missed']} ({la['lethal_missed']/avail:.1%})")
    print(f"  missed-KO turns in LOST games: {la['losing_miss_turns']} "
          f"(across {la['losing_miss_games']} lost games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
