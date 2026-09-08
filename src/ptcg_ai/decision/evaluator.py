"""Leaf evaluator ``V(observation, root_index) -> [-1, 1]`` (ADR-0013 / 0010).

A linear sum of bounded, normalized, *perspective-safe* features — quantities
visible for BOTH players in any observation (board, zone counts, prize-pile
sizes), so a leaf owned by either player is scored identically from a fixed
root seat. Every feature is a (my − opponent) difference, which makes the whole
value antisymmetric: ``value(obs, 0) == -value(obs, 1)``. The feature set is
anchored to the measured loss modes (bench-out with an even prize race): prize
race, benched reserves, survival insurance, KO threat both ways, development.

SDK-free: consumes only parsed models + the card database.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.observation.models import ParsedObservation, ParsedPlayer
from ptcg_ai.state.game_state import GameState

#: Deck size below which deck-out risk starts to bite (cards remaining).
_DECKOUT_THRESHOLD = 8


@dataclass(frozen=True)
class EvalWeights:
    """Weights on the normalized feature differences. Defaults are the
    hand-tuned v0 prior (arena-tuned in M6-3); magnitudes sum to ~1 so a
    lopsided-but-not-terminal board maps near the ±1 rails."""

    prize: float = 0.42
    threat: float = 0.20
    reserve: float = 0.12
    survival: float = 0.12
    energy: float = 0.06
    hand: float = 0.04
    deck_out: float = 0.04


class Evaluator:
    """Scores a leaf observation for a fixed root player."""

    def __init__(self, weights: EvalWeights | None = None, cards: CardDatabase | None = None) -> None:
        self._w = weights or EvalWeights()
        self._cards = cards

    def value(self, obs: ParsedObservation, root_index: int) -> float:
        state = obs.current
        if state is None:
            return 0.0
        if state.result != -1:  # terminal short-circuit
            if state.result == root_index:
                return 1.0
            if state.result == 1 - root_index:
                return -1.0
            return 0.0  # draw (result == 2)

        gs = GameState.build_for(obs, self._cards, root_index)
        me, opp = gs.me, gs.opponent
        if me is None or opp is None:
            return 0.0
        w = self._w

        # Each term is a normalized (me − opp) difference in [-1, 1].
        prize = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
        reserve = _clip1((gs.reserve_attackers(me) - gs.reserve_attackers(opp)) / 5.0)
        survival = _clip1((gs.my_bench_count - len(opp.bench)) / 5.0)
        threat = self._threat_term(gs)
        energy = _clip1((_board_energy(me) - _board_energy(opp)) / 10.0)
        hand = _clip1((me.handCount - opp.handCount) / 10.0)
        deck_out = _deckout_risk(opp.deckCount) - _deckout_risk(me.deckCount)

        total = (
            w.prize * prize
            + w.threat * threat
            + w.reserve * reserve
            + w.survival * survival
            + w.energy * energy
            + w.hand * hand
            + w.deck_out * deck_out
        )
        return _clip1(total)

    def _threat_term(self, gs: GameState) -> float:
        """+ when I threaten to KO their Active next turn, − when they threaten
        mine (capped at a full KO each way)."""
        on_them = 0.0
        on_me = 0.0
        if gs.opp_active is not None and gs.opp_active.hp > 0:
            dmg = gs.max_threat(gs.my_active, gs.opp_active, extra_energy=1)
            on_them = min(1.0, dmg / gs.opp_active.hp)
        if gs.my_active is not None and gs.my_active.hp > 0:
            dmg = gs.max_threat(gs.opp_active, gs.my_active, extra_energy=1)
            on_me = min(1.0, dmg / gs.my_active.hp)
        return on_them - on_me


def _board_energy(player: ParsedPlayer) -> int:
    total = sum(len(p.energies) for p in player.active if p is not None)
    total += sum(len(p.energies) for p in player.bench)
    return total


def _deckout_risk(deck_count: int) -> float:
    """Rises from 0 to 1 as the deck empties past the threshold."""
    if deck_count >= _DECKOUT_THRESHOLD:
        return 0.0
    return (_DECKOUT_THRESHOLD - deck_count) / _DECKOUT_THRESHOLD


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, x))
