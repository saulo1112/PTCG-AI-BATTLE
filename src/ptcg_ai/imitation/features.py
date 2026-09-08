"""Per-option feature vector for behavior cloning (SHIPPED, pure stdlib).

This is the single featurizer used both offline (training) and live (the
:class:`~ptcg_ai.imitation.policy.ImitationPolicy`) — training-serving skew is
prevented by there being exactly one code path. It scores ONE option of a
decision given a :class:`~ptcg_ai.imitation.deck_profiles.DeckProfile` (the
cloned player's deck-specific vocabularies + scalar extractors), the parsed
observation, the derived :class:`GameState`, and the card database.

Design facts it exploits:

- Each cloned player runs ONE fixed 60-card deck, so own-card identity is a small
  one-hot over the deck's ids (``profile.deck_ids``) rather than an embedding over
  the whole pool.
- In a softmax over a decision's options, features constant across options cancel;
  the state only bites through its *interactions* with option features (``A⊗S``,
  ``B⊗S``), which is where most of the capacity lives.
- Some attacks (Rocket Rush, Cosmic Beam) have text the engine can't express in
  structured damage, so ``profile.damage_fn`` hand-computes them.

Blocks: A option-type one-hot(12), B acting-card one-hot(deck+1), C attack
one-hot(attacks+1), D target block(area 4 + target one-hot + energy/damaged/wants
+ optional opponent-target features), E combat(6), F generic-opponent(5), S state
snapshot, then A⊗S and B⊗S_small interactions. Sizes come from the profile.

No numpy: the shipped agent runs on the stdlib-only Kaggle contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.imitation.deck_profiles import _frac
from ptcg_ai.observation.models import (
    AreaKind,
    OptionKind,
    ParsedObservation,
    ParsedSelect,
    SelectContextKind,
)
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

if TYPE_CHECKING:
    from ptcg_ai.imitation.deck_profiles import DeckProfile

# -- generic (deck-independent) block layout ----------------------------------

_OPTION_KIND_ORDER = (
    OptionKind.PLAY, OptionKind.ATTACH, OptionKind.EVOLVE, OptionKind.ABILITY,
    OptionKind.ATTACK, OptionKind.RETREAT, OptionKind.END, OptionKind.CARD,
    OptionKind.YES, OptionKind.NO, OptionKind.NUMBER,
)
_A = len(_OPTION_KIND_ORDER) + 1  # option-type one-hot (+other) = 12
_E = 6                            # combat interactions
_F = 5                            # generic opponent-active block
assert _A == 12  # kept in sync with deck_profiles._A (feature_dim)


@dataclass(frozen=True)
class DecisionState:
    """State scalars computed once per decision and reused across its options."""

    snapshot: tuple[float, ...]   # length profile.snapshot_len
    reduced: tuple[float, ...]    # length profile.reduced_len
    opp_active_hp: int
    opp_active_ex: bool
    active_dies_if_pass: bool


def feature_dim(profile: "DeckProfile") -> int:
    """Vector length for ``profile`` (mirror of ``profile.feature_dim``)."""
    return profile.feature_dim


def is_prize_pick(select: ParsedSelect) -> bool:
    """True when a TO_HAND select is choosing face-down prize cards.

    Prize picks are informationless (any legal choice is equivalent), so the
    policy must not try to learn them — it defers to the safe default.
    """
    if select.context is not SelectContextKind.TO_HAND:
        return False
    return any(o.area is AreaKind.PRIZE for o in select.option)


def effective_attack_damage(
    profile: "DeckProfile", attack_id: int | None, gs: GameState, cards: CardDatabase | None
) -> int:
    """Damage ``gs.my_active``'s attack deals to ``gs.opp_active`` (per profile)."""
    return profile.damage_fn(attack_id, gs, cards)


def decision_state(
    profile: "DeckProfile", obs: ParsedObservation, gs: GameState, cards: CardDatabase | None
) -> DecisionState:
    """Precompute the per-decision state scalars (shared across all options)."""
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    dies = False
    if my_active is not None and opp_active is not None:
        # "Do I die if I pass?" — best-effort; prose opponent attacks may read as 0.
        dies = gs.max_threat(opp_active, my_active, extra_energy=1) >= my_active.hp
    return DecisionState(
        snapshot=profile.snapshot_fn(obs, gs, cards),
        reduced=profile.reduced_fn(obs, gs, cards),
        opp_active_hp=opp_hp,
        opp_active_ex=opp_ex,
        active_dies_if_pass=dies,
    )


def featurize_option(
    profile: "DeckProfile",
    select: ParsedSelect,
    obs: ParsedObservation,
    gs: GameState,
    cards: CardDatabase | None,
    option_index: int,
    state: DecisionState | None = None,
) -> list[float]:
    """Feature vector (length ``profile.feature_dim``) scoring one option."""
    if state is None:
        state = decision_state(profile, obs, gs, cards)
    opt = select.option[option_index]
    resolved = resolve_option(opt, obs)

    # A. option-type one-hot
    a = [0.0] * _A
    try:
        a[_OPTION_KIND_ORDER.index(opt.type)] = 1.0
    except ValueError:
        a[_A - 1] = 1.0

    # B. acting-card one-hot (played/selected/ability-source card)
    b_len = profile.b_len
    b = [0.0] * b_len
    b[profile.deck_index.get(resolved.card_id, b_len - 1)] = 1.0

    # C. attack one-hot
    c_len = profile.c_len
    c = [0.0] * c_len
    if opt.type is OptionKind.ATTACK:
        c[profile.attack_index.get(opt.attackId, c_len - 1)] = 1.0

    # D. target block: area(4) + target one-hot(tlen+1) + energy/damaged/wants(3)
    #    + optional opponent-target features(opp_target_dim)
    tlen = len(profile.target_pokemon_ids)
    d = [0.0] * profile.d_len
    target_area = opt.inPlayArea if opt.inPlayArea is not None else opt.area
    area_slot = {AreaKind.ACTIVE: 0, AreaKind.BENCH: 1, AreaKind.STADIUM: 2}.get(target_area, 3)
    d[area_slot] = 1.0
    target_id = resolved.target_card_id if resolved.target_card_id is not None else resolved.card_id
    d[4 + profile.target_index.get(target_id, tlen)] = 1.0
    energy_i, damaged_i, wants_i = 4 + tlen + 1, 4 + tlen + 2, 4 + tlen + 3
    target_pkmn = _target_pokemon(opt, obs, target_area)
    if target_pkmn is not None:
        d[energy_i] = _frac(len(target_pkmn.energies), 4)
        d[damaged_i] = 1.0 if target_pkmn.hp < target_pkmn.maxHp else 0.0
        d[wants_i] = profile.wants_fn(target_pkmn, gs, cards)
    # Opponent-owned target (Boss's Orders / on-evolve gust) features.
    if profile.opp_target_dim > 0 and obs.current is not None:
        yidx = obs.current.yourIndex
        if opt.playerIndex is not None and opt.playerIndex != yidx:
            od = 4 + tlen + 4
            d[od] = 1.0  # target is an opponent Pokémon
            if target_pkmn is not None:
                d[od + 1] = _frac(target_pkmn.hp, 300)
                prize_val = 3 if _is_ex(target_pkmn, cards) else 1
                d[od + 2] = _frac(prize_val, 3)
                my_active = gs.my_active
                if my_active is not None:
                    d[od + 3] = 1.0 if gs.max_threat(my_active, target_pkmn) >= target_pkmn.hp else 0.0

    # E. combat interactions (only meaningful for ATTACK options)
    e = [0.0] * _E
    if opt.type is OptionKind.ATTACK:
        dmg = profile.damage_fn(opt.attackId, gs, cards)
        e[0] = _frac(dmg, 200)
        e[1] = 1.0 if (state.opp_active_hp > 0 and dmg >= state.opp_active_hp) else 0.0
        e[2] = _frac(dmg - state.opp_active_hp, 200) if dmg > state.opp_active_hp else 0.0
        e[3] = 1.0 if state.opp_active_ex else 0.0
        e[4] = 1.0 if opt.attackId == profile.flagged_attack_id else 0.0
        e[5] = 1.0 if state.active_dies_if_pass else 0.0

    # F. generic opponent-active block (card-agnostic, from CardDatabase)
    f = [0.0] * _F
    if gs.opp_active is not None and cards is not None:
        info = cards.get_card(gs.opp_active.id)
        f[0] = _frac(gs.opp_active.hp, 300)
        if info is not None:
            f[1] = 0.0 if info.basic else (0.5 if info.stage1 else 1.0)
            f[2] = 1.0 if (info.ex or info.megaEx) else 0.0
        f[3] = _frac(gs.opp_active_prize_value, 3)
        f[4] = _frac(len(gs.opp_active.energies), 4)

    s = list(state.snapshot)
    a_s = [av * sv for av in a for sv in state.snapshot]
    b_s = [bv * rv for bv in b for rv in state.reduced]

    return a + b + c + d + e + f + s + a_s + b_s


def _is_ex(pkmn, cards: CardDatabase | None) -> bool:
    if cards is None:
        return False
    info = cards.get_card(pkmn.id)
    return bool(info and (info.ex or info.megaEx))


def _target_pokemon(opt, obs: ParsedObservation, target_area: AreaKind | None):
    """The Pokémon an option acts on (attach target / promote / ability source /
    gust target). Honors ``playerIndex`` so opponent-owned targets resolve too."""
    state = obs.current
    if state is None or target_area not in (AreaKind.ACTIVE, AreaKind.BENCH):
        return None
    index = opt.inPlayIndex if opt.inPlayIndex is not None else opt.index
    player_index = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
    if not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    zone = player.active if target_area is AreaKind.ACTIVE else player.bench
    if index is None or not (0 <= index < len(zone)):
        return None
    return zone[index]


def featurize_decision(
    profile: "DeckProfile",
    select: ParsedSelect,
    obs: ParsedObservation,
    gs: GameState,
    cards: CardDatabase | None,
) -> list[list[float]]:
    """All option vectors for a decision (state computed once)."""
    state = decision_state(profile, obs, gs, cards)
    return [
        featurize_option(profile, select, obs, gs, cards, i, state)
        for i in range(len(select.option))
    ]
