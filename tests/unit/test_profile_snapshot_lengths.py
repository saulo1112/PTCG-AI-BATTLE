"""Every registered profile's snapshot/reduced fn must return exactly the declared length.

Nothing enforced this before. ``DeckProfile.feature_dim`` is computed from ``snapshot_len`` /
``reduced_len``, but the actual functions build tuples by hand — so a mismatch would silently
produce feature vectors of the wrong length, corrupting training AND making a shipped weights
file score garbage (``load_weights`` only checks the declared dim, not what the fn returns).

Also pins the two ALAKAZAM dims, since M31 Track E ablates by zeroing snapshot column ranges
and that arithmetic depends on the exact layout.
"""

from __future__ import annotations

import pytest

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.observation.models import CardKind, EnergyKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

PSYCHIC = int(EnergyKind.PSYCHIC)


def _card(cid, *, hp=120, basic=True):
    return {
        "cardId": cid, "name": f"c{cid}", "cardType": int(CardKind.POKEMON), "retreatCost": 1,
        "hp": hp, "weakness": None, "resistance": None, "energyType": PSYCHIC,
        "basic": basic, "stage1": False, "stage2": False, "ex": False, "megaEx": False,
        "tera": False, "aceSpec": False, "evolvesFrom": None, "skills": [], "attacks": [],
    }


def _cards() -> CardDatabase:
    # a few real deck ids plus an opponent, enough for every profile's lookups to resolve
    ids = {1, 5, 11, 14, 18, 66, 140, 305, 343, 344, 345, 400, 741, 742, 743, 756, 9999}
    return CardDatabase.from_records({"cards": [_card(i) for i in sorted(ids)], "attacks": []})


def _pokemon(cid):
    return {
        "id": cid, "serial": cid * 10, "hp": 100, "maxHp": 120, "appearThisTurn": False,
        "energies": [PSYCHIC], "energyCards": [], "preEvolution": [], "tools": [],
    }


def _obs():
    player = {
        "active": [_pokemon(741)], "bench": [_pokemon(742)], "benchMax": 5, "deckCount": 30,
        "discard": [], "prize": [None] * 6, "handCount": 3,
        "hand": [{"id": 743, "serial": 1, "playerIndex": 0},
                 {"id": 1079, "serial": 2, "playerIndex": 0},
                 {"id": 5, "serial": 3, "playerIndex": 0}],
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    opp = {
        "active": [_pokemon(9999)], "bench": [], "benchMax": 5, "deckCount": 28,
        "discard": [], "prize": [None] * 6, "handCount": 4, "hand": None,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    return {
        "select": {"type": 0, "context": int(SelectContextKind.MAIN), "minCount": 1, "maxCount": 1,
                   "remainDamageCounter": 0, "remainEnergyCost": 0,
                   "option": [{"type": 14}], "deck": None, "contextCard": None, "effect": None},
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": 6, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [player, opp]},
    }


@pytest.mark.parametrize("name", sorted(DP.PROFILES))
def test_snapshot_and_reduced_lengths_match_declaration(name: str) -> None:
    profile = DP.PROFILES[name]
    cards = _cards()
    obs = ObservationParser().parse(_obs())
    gs = GameState.build(obs, cards)
    assert len(profile.snapshot_fn(obs, gs, cards)) == profile.snapshot_len, (
        f"{name}.snapshot_fn length != declared snapshot_len")
    assert len(profile.reduced_fn(obs, gs, cards)) == profile.reduced_len, (
        f"{name}.reduced_fn length != declared reduced_len")


def test_alakazam_dims_are_pinned() -> None:
    # Track E's masked ablation indexes snapshot columns directly; pin the layout.
    assert DP.ALAKAZAM.feature_dim == 658
    assert DP.ALAKAZAM_V2.feature_dim == 1269
    assert DP.ALAKAZAM_V2.snapshot_len == 76
    assert DP.ALAKAZAM_MATCHUP.feature_dim == 684
    assert DP.ALAKAZAM_MATCHUP.snapshot_len == 31
    assert DP.ALAKAZAM_SPECIALIST.feature_dim == 697
    assert DP.ALAKAZAM_SPECIALIST.snapshot_len == 32
    # M34: the only ALAKAZAM variant that grows `reduced` instead of `snapshot`
    # — for TO_HAND, B⊗R is the sole state-conditioned block (see the profile's
    # header). Same snapshot as V1 on purpose.
    assert DP.ALAKAZAM_FETCH.feature_dim == 750
    assert DP.ALAKAZAM_FETCH.snapshot_len == DP.ALAKAZAM.snapshot_len == 29
    assert DP.ALAKAZAM_FETCH.reduced_len == 13
    # M46: the same move for MAIN. For two MAIN options of the SAME OptionKind,
    # A⊗S cancels too, so B⊗R (9 scalars) is the ONLY thing separating them —
    # and PLAY→PLAY is the #1 disagreement with the teacher (M33: 29.2%).
    # Snapshot deliberately unchanged: M31's V2 grew it to 76 and lost the
    # head-to-head, because snapshot only reaches options at option-TYPE
    # granularity and can never separate two options of the same type.
    assert DP.ALAKAZAM_MAIN2.feature_dim == 796
    assert DP.ALAKAZAM_MAIN2.snapshot_len == DP.ALAKAZAM.snapshot_len == 29
    assert DP.ALAKAZAM_MAIN2.reduced_len == 15
    assert DP.ALAKAZAM_MAIN2.deck_ids == DP.ALAKAZAM.deck_ids
    assert DP.ALAKAZAM_MAIN2.damage_fn is DP.ALAKAZAM.damage_fn
    assert DP.ALAKAZAM_MAIN2.wants_fn is DP.ALAKAZAM.wants_fn
    assert DP.ALAKAZAM_MAIN2.snapshot_fn is DP.ALAKAZAM.snapshot_fn
    # groups must be contiguous, ordered, and cover exactly the new 29..76 range
    bounds = [DP.ALAKAZAM_V2_GROUPS[g] for g in ("G1", "G2", "G3", "G4")]
    assert bounds[0][0] == DP.ALAKAZAM.snapshot_len == 29
    assert bounds[-1][1] == DP.ALAKAZAM_V2.snapshot_len
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start, "ALAKAZAM_V2 groups must be contiguous"


def test_alakazam_mem_dims_are_pinned() -> None:
    # M37 arm E: the two memory channels ALAKAZAM never used — the discard pile (every
    # other profile reads it; this line did not) and turnActionCount (parsed forever,
    # read by nobody). Additive: same deck/attacks/damage/wants, only the state blocks
    # grow. Placement follows M34 — the three scalars that must condition on WHICH card
    # is being chosen go in `reduced` (crossed with the card-id one-hot), the rest in
    # `snapshot`.
    assert DP.ALAKAZAM_MEM.feature_dim == 818
    assert DP.ALAKAZAM_MEM.snapshot_len == 36
    assert DP.ALAKAZAM_MEM.reduced_len == 12
    assert DP.ALAKAZAM_MEM.deck_ids == DP.ALAKAZAM.deck_ids
    assert DP.ALAKAZAM_MEM.damage_fn is DP.ALAKAZAM.damage_fn
    assert DP.ALAKAZAM_MEM.wants_fn is DP.ALAKAZAM.wants_fn
    # the first 29 / 9 slots must still be exactly ALAKAZAM's, or the added slots are
    # silently shifting the meaning of the base ones
    cards = _cards()
    obs = ObservationParser().parse(_obs())
    gs = GameState.build(obs, cards)
    assert (DP.ALAKAZAM_MEM.snapshot_fn(obs, gs, cards)[:29]
            == DP.ALAKAZAM.snapshot_fn(obs, gs, cards))
    assert (DP.ALAKAZAM_MEM.reduced_fn(obs, gs, cards)[:9]
            == DP.ALAKAZAM.reduced_fn(obs, gs, cards))


def test_grimmsnarl_dims_are_pinned() -> None:
    # M35 clone of Luca (top-5). opp_target_dim=4 is load-bearing beyond Boss's
    # Orders: DAMAGE_COUNTER/DAMAGE target the OPPONENT's board and the opp-target
    # block is their only discriminating signal, so a drop to 0 would silently
    # blind 9.6% of this teacher's decisions.
    assert DP.GRIMMSNARL.feature_dim == 706
    assert DP.GRIMMSNARL.snapshot_len == 29
    assert DP.GRIMMSNARL.reduced_len == 13
    assert DP.GRIMMSNARL.opp_target_dim == 4
    assert DP.GRIMMSNARL.flagged_attack_id == 937  # Shadow Bullet
    assert DP.PROFILES["GRIMMSNARL"] is DP.GRIMMSNARL
