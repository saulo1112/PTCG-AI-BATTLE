"""M17 critic features v2 — domain-knowledge terms for the RL value baseline.

The M16 critic (`v_rl.json`) ranks won-vs-lost states with the SAME 7 antisymmetric
features the hand-tuned `decision/evaluator.py` uses. Three of those features are
demonstrably lossy for THIS game's decks (verified against the engine's card data,
docs/m17_findings.md):

  * `prize` is FLAT — a KO of a 2-prize ex and a 1-prize basic move it identically,
    even though `GameState` already computes per-Pokémon prize value (1/2/3) and
    leaves it dead (`game_state.py:57-58,269-280`).
  * `threat` uses STRUCTURED attack damage only (`game_state.attack_damage` line 139:
    "conditional +N effect text is prose and not modelled at rung 3"), so it badly
    underestimates the swarm/scaling attacks that decide our matchups — Rocket Rush
    (30×TR-in-play) and Voltaic Chain (20+20×board-{L}). Those exact prose formulas
    already exist in `deck_profiles.damage_fn`, but only for the AGENT's own attacks
    in the POLICY featurizer, never in the critic's threat estimate for either side.

This module recomputes the base 7 (internally consistent, refit fresh — NOT compared
to the stored 0.8249) and appends three ablatable domain terms:

  idx 7  pw_threat    (T1) prize-weighted threat: structured threat scaled by the
                           Active's prize value — trading into their 2-prize ex is
                           worth more than into a 1-prize basic.
  idx 8  board_prize  (T2) board prize pool diff: sum of prize value over each side's
                           in-play Pokémon — "their board exposes 2-3-prize targets /
                           mine exposes them" (prize mapping in state form).
  idx 9  prose_threat (T3) prose correction delta: (prose-aware threat) minus
                           (structured threat), isolating exactly what the rung-3
                           damage approximation misses for the scaling attacks.

T4 (post-Mega-Brave / Thunderous-Bolt "can't attack next turn" lock) was investigated
and DROPPED: `observation/models.py` exposes only special conditions
(poisoned/burned/asleep/paralyzed/confused), never an attack-lock flag — the state is
not observable, so it cannot be featurized (docs/m17_findings.md, Task C).

Dev-only (numpy). Read alongside `scratchpad/train_value.py` (the base-7 original) and
`scratchpad/critic_v2_gate.py` (the ablation gate that consumes this).
"""

from __future__ import annotations

from ptcg_ai.observation.models import EnergyKind, ParsedPlayer, ParsedPokemon
from ptcg_ai.state.game_state import GameState

# Column layout of the returned vector. The gate selects subsets by index.
FEATURE_NAMES = (
    "prize", "threat", "reserve", "survival", "energy", "hand", "deck_out",  # base 0-6
    "pw_threat",     # T1  idx 7
    "board_prize",   # T2  idx 8
    "prose_threat",  # T3  idx 9
)
BASE_COLS = tuple(range(7))
GROUP_COLS = {"t1": (7,), "t2": (8,), "t3": (9,)}

# -- prose-scaling attacks (engine ground truth; mirrors deck_profiles damage_fn) -----
_TR_POKEMON_IDS = frozenset({400, 401, 433, 434})  # Team Rocket's Tarountula/Spidops line
_ROCKET_RUSH = 560          # Spidops: 30 × Team Rocket's Pokémon in play
_ROCKET_RUSH_PER_TR = 30
_VOLTAIC_CHAIN = 363        # Iono's Voltorb: 20 + 20 × board {L} energy
_RESISTANCE_REDUCTION = 30  # game_state.RESISTANCE_REDUCTION


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _in_play(player: ParsedPlayer | None):
    if player is None:
        return []
    out = [p for p in player.active if p is not None]
    out.extend(player.bench)
    return out


def _tr_in_play(player: ParsedPlayer | None) -> int:
    return sum(1 for p in _in_play(player) if p.id in _TR_POKEMON_IDS)


def _l_energy_in_play(player: ParsedPlayer | None) -> int:
    return sum(1 for p in _in_play(player) for e in p.energies if e is EnergyKind.LIGHTNING)


def _board_prize_pool(player: ParsedPlayer | None, cards) -> int:
    """Total prizes the opponent would take for KO'ing this player's whole board."""
    return sum(GameState._prize_value(p, cards) for p in _in_play(player))


def _prose_base(attack_id: int, attacker_player: ParsedPlayer | None) -> int | None:
    """Prose damage base for a known scaling attack, computed from the ATTACKER's own
    board (so it is correct for either seat), or None to fall back to structured."""
    if attack_id == _ROCKET_RUSH:
        return _ROCKET_RUSH_PER_TR * _tr_in_play(attacker_player)
    if attack_id == _VOLTAIC_CHAIN:
        return 20 + 20 * _l_energy_in_play(attacker_player)
    return None


def _prose_max_threat(
    gs: GameState,
    attacker: ParsedPokemon | None,
    defender: ParsedPokemon | None,
    attacker_player: ParsedPlayer | None,
    extra_energy: int = 1,
) -> int:
    """Like `GameState.max_threat`, but substitutes the prose base for scaling attacks
    before applying weakness/resistance. Falls back to structured damage otherwise."""
    if attacker is None or defender is None or gs.cards is None:
        return 0
    atk_info = gs.cards.get_card(attacker.id)
    if atk_info is None:
        best = gs.best_attack(attacker)
        return best.damage if best else 0
    def_info = gs.cards.get_card(defender.id)
    attached = tuple(attacker.energies) + (atk_info.energyType,) * max(0, extra_energy)
    best_dmg = 0
    for attack_id in atk_info.attacks:
        attack = gs.cards.get_attack(attack_id)
        if attack is None:
            continue
        prose = _prose_base(attack_id, attacker_player)
        base = prose if prose is not None else attack.damage
        if base <= 0 or not gs.energy_pays(attached, attack.energies):
            continue
        dmg = base
        if def_info is not None:
            if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
                dmg *= 2
            if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
                dmg = max(0, dmg - _RESISTANCE_REDUCTION)
        best_dmg = max(best_dmg, dmg)
    return best_dmg


def features(obs, cards) -> list[float] | None:
    """The base-7 evaluator features plus T1/T2/T3, from the teacher/agent perspective.

    Base 7 recomputed inline to match `train_value.features` term-for-term (verified),
    so the gate's own base-7 baseline and the v2 vector share identical base columns."""
    if obs.current is None:
        return None
    gs = GameState.build(obs, cards)
    me, opp = gs.me, gs.opponent
    if me is None or opp is None:
        return None

    # -- base 7 (identical to train_value.features) --
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

    # -- T1 prize-weighted threat: structured threat scaled by Active prize value (1/2/3) --
    my_apv = gs.my_active_prize_value or 1
    opp_apv = gs.opp_active_prize_value or 1
    pw_threat = _clip1(on_them_s * (opp_apv / 2.0) - on_me_s * (my_apv / 2.0))

    # -- T2 board prize pool diff: their exposed prizes minus mine --
    board_prize = _clip1((_board_prize_pool(opp, cards) - _board_prize_pool(me, cards)) / 6.0)

    # -- T3 prose correction delta: prose-aware threat minus structured threat --
    on_them_p = on_me_p = 0.0
    if gs.opp_active is not None and gs.opp_active.hp > 0:
        on_them_p = min(1.0, _prose_max_threat(gs, gs.my_active, gs.opp_active, me) / gs.opp_active.hp)
    if gs.my_active is not None and gs.my_active.hp > 0:
        on_me_p = min(1.0, _prose_max_threat(gs, gs.opp_active, gs.my_active, opp) / gs.my_active.hp)
    prose_threat = (on_them_p - on_me_p) - threat

    return [prize, threat, reserve, survival, energy, hand, deck_out,
            pw_threat, board_prize, prose_threat]


# Reuse the evaluator's exact board-energy / deckout helpers (single source of truth).
from ptcg_ai.decision.evaluator import _board_energy, _deckout_risk  # noqa: E402
