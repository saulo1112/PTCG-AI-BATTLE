"""Per-deck featurization profiles for behavior cloning (SHIPPED, pure stdlib).

The imitation featurizer (:mod:`~ptcg_ai.imitation.features`) is deck-specific:
own-card identity is a one-hot over the target player's fixed decklist, the
attack vocabulary is that deck's attacks, and some attacks need hand-computed
damage because the engine's structured value can't express their text (Rocket
Rush scales with board width; Cosmic Beam is conditional and ignores W/R). A
:class:`DeckProfile` bundles those deck-specific pieces so one featurizer serves
every cloned player. Each profile's ``*_fn`` are **module-level functions, never
lambdas or closures**, so the shipped bundle stays import-safe and picklable.

Two profiles ship today: ``TR_650`` (M7 Team Rocket swarm) and ``LUCARIO_800``
(M8 Mega Lucario ex Fighting). ``TR_650`` reproduces the M7 featurizer bit-for-
bit so ``bc_650_v1.json`` keeps working unchanged.

No numpy / no third-party imports: the shipped agent runs on the stdlib-only
Kaggle contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.observation.models import EnergyKind, ParsedObservation, ParsedPlayer, ParsedPokemon
from ptcg_ai.state.game_state import GameState

#: Generic (deck-independent) block sizes, mirrored in :mod:`features`.
_A = 12  # option-type one-hot
_E = 6   # combat interactions
_F = 5   # generic opponent-active block


def _frac(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    v = num / den
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _count_in_play(player: ParsedPlayer | None, card_id: int) -> int:
    if player is None:
        return 0
    n = 0
    for p in player.active:
        if p is not None and p.id == card_id:
            n += 1
    for p in player.bench:
        if p.id == card_id:
            n += 1
    return n


def _count_on_bench(player: ParsedPlayer | None, card_id: int) -> int:
    return 0 if player is None else sum(1 for p in player.bench if p.id == card_id)


def _benched_with_energy(player: ParsedPlayer | None) -> int:
    return 0 if player is None else sum(1 for p in player.bench if p.energies)


# =============================================================================
# TR_650 — Team Rocket swarm (M7). Bodies moved verbatim from the M7 featurizer.
# =============================================================================

_TR_POKEMON_IDS = frozenset({400, 401, 433, 434})
_ROCKET_RUSH = 560
_TAKE_DOWN = 559
_CHIMING = 611
_MAX_BELT_ID = 1158
_ROCKET_RUSH_PER_TR = 30


def _tr_in_play(player: ParsedPlayer | None) -> int:
    if player is None:
        return 0
    count = 0
    for p in player.active:
        if p is not None and p.id in _TR_POKEMON_IDS:
            count += 1
    for p in player.bench:
        if p.id in _TR_POKEMON_IDS:
            count += 1
    return count


def _basic_energy_in_discard(player: ParsedPlayer | None, cards: CardDatabase | None) -> int:
    if player is None or cards is None:
        return 0
    count = 0
    for card in player.discard:
        info = cards.get_card(card.id)
        if info is not None and info.is_basic_energy:
            count += 1
    return count


def damage_650(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Rocket Rush = 30×TR in play (+50 Maximum Belt vs ex, then W/R)."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _ROCKET_RUSH:
        base = _ROCKET_RUSH_PER_TR * _tr_in_play(gs.me)
    elif attack_id == _TAKE_DOWN:
        base = 30
    elif attack_id == _CHIMING:
        base = 0
    else:
        atk = cards.get_attack(attack_id) if cards is not None else None
        base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if def_info is not None and (def_info.ex or def_info.megaEx):
        if any(t.id == _MAX_BELT_ID for t in attacker.tools):
            base += 50  # Maximum Belt, before Weakness/Resistance
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_650(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    tr = _tr_in_play(me)
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    discard_energy = _basic_energy_in_discard(me, cards)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(tr, 6),
        _frac(discard_energy, 14),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 300),
        _frac(_ROCKET_RUSH_PER_TR * tr, 200),
    )


def reduced_650(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    tr = _tr_in_play(me)
    discard_energy = _basic_energy_in_discard(me, cards)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(tr, 6),
        _frac(discard_energy, 14),
    )


def wants_650(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """A Tarountula/Spidops still short of Rocket Rush energy wants the attach."""
    return 1.0 if pkmn.id in _TR_POKEMON_IDS and len(pkmn.energies) < 2 else 0.0


# =============================================================================
# LUCARIO_800 — Mega Lucario ex Fighting (M8).
# =============================================================================

_F_ENERGY_ID = 6            # Basic {F} Energy (Aura Jab discard fuel)
_LUNATONE_ID = 675
_SOLROCK_ID = 676
_GRAVITY_MOUNTAIN_ID = 1252
_HEROS_CAPE_ID = 1159
_COSMIC_BEAM = 980


def _f_energy_in_discard(player: ParsedPlayer | None) -> int:
    return 0 if player is None else sum(1 for c in player.discard if c.id == _F_ENERGY_ID)


def _my_active_has_cape(gs: GameState) -> bool:
    a = gs.my_active
    return a is not None and any(t.id == _HEROS_CAPE_ID for t in a.tools)


def damage_800(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured damage + W/R, except Cosmic Beam (0 without Lunatone on the
    bench; ignores Weakness/Resistance per its text). All other Lucario-deck
    attacks have honest structured damage the engine reports correctly."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _COSMIC_BEAM:
        return 70 if _count_on_bench(gs.me, _LUNATONE_ID) > 0 else 0  # ignores W/R
    atk = cards.get_attack(attack_id) if cards is not None else None
    base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_800(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    stadium_ids = {c.id for c in state.stadium} if state is not None else set()
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(_f_energy_in_discard(me), 6),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 300),
        1.0 if _count_in_play(me, _LUNATONE_ID) > 0 else 0.0,
        1.0 if _count_in_play(me, _SOLROCK_ID) > 0 else 0.0,
        1.0 if _GRAVITY_MOUNTAIN_ID in stadium_ids else 0.0,
        1.0 if _my_active_has_cape(gs) else 0.0,
        _frac(_benched_with_energy(me), 5),
    )


def reduced_800(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(_f_energy_in_discard(me), 6),
        _frac(_benched_with_energy(me), 5),
        _frac(opp_hp, 300),
    )


def wants_800(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """A Lucario-deck Pokémon that can't yet pay its best attack wants energy."""
    return 1.0 if gs.wants_energy(pkmn) else 0.0


# --- LUCARIO_800_V2: adds hand-composition + active-energy visibility (M8.1) ---
# v2.0 saw hand SIZE but not COMPOSITION, so the clone over-clicked Lunar Cycle
# (discard a hand F energy → draw 3) without seeing whether it could spare the
# energy, starving its big Fighting attacks. These features expose the hand's
# make-up and the active's energy so the B⊗S interactions can learn "click the
# draw engine only with spare energy" and "play Premium Power Pro near a KO".

_MEGA_LUCARIO_ID = 678
_DRAW_SUPPORTER_IDS = (1192, 1227)   # Carmine, Lillie's Determination
_BOSS_ID = 1182
_POWER_PRO_ID = 1141
_SEARCH_ITEM_IDS = (1102, 1142, 1152)  # Dusk Ball, Fighting Gong, Poké Pad
_BASIC_POKEMON_IDS = (673, 675, 676, 677)  # Makuhita, Lunatone, Solrock, Riolu
_EVOLUTION_IDS = (674, 678)          # Hariyama, Mega Lucario ex


def _hand_count(player: ParsedPlayer | None, ids) -> int:
    if player is None or player.hand is None:
        return 0
    idset = ids if isinstance(ids, (set, frozenset)) else set(ids)
    return sum(1 for c in player.hand if c.id in idset)


def snapshot_800v2(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me = gs.me
    my_active = gs.my_active
    base = snapshot_800(obs, gs, cards)  # first 21 slots unchanged
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_f_hand = sum(1 for c in hand if c.id == _F_ENERGY_ID)
    active_energy = len(my_active.energies) if my_active is not None else 0
    return base + (
        _frac(n_f_hand, 6),
        _frac(_hand_count(me, _DRAW_SUPPORTER_IDS), 4),
        _frac(_hand_count(me, (_BOSS_ID,)), 2),
        _frac(_hand_count(me, (_POWER_PRO_ID,)), 4),
        _frac(_hand_count(me, _SEARCH_ITEM_IDS), 6),
        _frac(_hand_count(me, _BASIC_POKEMON_IDS), 6),
        _frac(_hand_count(me, _EVOLUTION_IDS), 4),
        _frac(active_energy, 4),
        1.0 if (my_active is not None and my_active.id == _MEGA_LUCARIO_ID) else 0.0,
        _frac(_count_on_bench(me, _MEGA_LUCARIO_ID), 2),
    )


def reduced_800v2(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me = gs.me
    my_active = gs.my_active
    base = reduced_800(obs, gs, cards)  # 7 slots
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_f_hand = sum(1 for c in hand if c.id == _F_ENERGY_ID)
    active_energy = len(my_active.energies) if my_active is not None else 0
    return base + (_frac(n_f_hand, 6), _frac(active_energy, 4))


# =============================================================================
# BELLIBOLT_940 — Iono's Bellibolt ex Lightning engine/aggro (M9).
# =============================================================================
# kenN2439's ~940-elo deck. Two Stage-1 lines (Tadbulb→Bellibolt ex the engine,
# Wattrel→Kilowattrel the draw engine), Voltorb a standalone attacker. The key
# engine is Bellibolt's Electric Streamer (unlimited per-turn {L} attach), which
# fuels Voltaic Chain's board-wide-energy scaling. No gust trainers (opp_target_
# dim=0). The M8.1 lesson is baked in from the start: the snapshot exposes hand
# COMPOSITION (basic-{L} count, draw supporters) and board energy — not just hand
# size — so the clone can learn "click Flashing Draw only with spare energy"
# rather than over-clicking the draw engine and starving its attacks (v2.0's bug).

_L_ENERGY_ID = 4              # Basic {L} Energy (Electric Streamer fuel / Voltaic Chain scaler)
_VOLTORB_ID = 265
_TADBULB_ID = 268
_BELLIBOLT_ID = 269          # Stage-1 ex, main attacker + Electric Streamer engine
_WATTREL_ID = 270
_KILOWATTREL_ID = 271        # Stage-1, Flashing Draw engine
_LEVINCIA_ID = 1254          # Stadium: recover 2 basic {L} from discard each turn
_DRAW_SUPPORTER_IDS_940 = (1227, 1233)  # Lillie's Determination, Canari
_VOLTAIC_CHAIN = 363         # Voltorb: 20 + 20 per {L} on all own Iono's Pokémon (prose)
_TINY_CHARGE = 367           # Tadbulb: 30 structured
_THUNDEROUS_BOLT = 368       # Bellibolt: 230, can't attack next turn (flagged)
_QUICK_ATTACK = 369          # Wattrel: 10 + coin-flip 20 (we model the guaranteed 10)
_MACH_BOLT = 370             # Kilowattrel: 70 structured


def _l_energy_on_board(player: ParsedPlayer | None) -> int:
    """Total Lightning energy attached across all of ``player``'s Pokémon.

    Every Pokémon in this deck is an "Iono's" Pokémon, so this is exactly the
    count Voltaic Chain scales on."""
    if player is None:
        return 0
    n = 0
    for p in player.active:
        if p is not None:
            n += sum(1 for e in p.energies if e is EnergyKind.LIGHTNING)
    for p in player.bench:
        n += sum(1 for e in p.energies if e is EnergyKind.LIGHTNING)
    return n


def _discard_resources_940(player: ParsedPlayer | None, cards: CardDatabase | None) -> int:
    """Pokémon + basic {L} energy in discard (recoverable by Night Stretcher /
    Max Rod / Energy Retrieval / Levincia — the deck's recycling loop)."""
    if player is None or cards is None:
        return 0
    n = 0
    for c in player.discard:
        if c.id == _L_ENERGY_ID:
            n += 1
        else:
            info = cards.get_card(c.id)
            if info is not None and info.cardType.name == "POKEMON":
                n += 1
    return n


def _kilowattrel_can_draw(player: ParsedPlayer | None) -> bool:
    """Flashing Draw needs a Kilowattrel in play carrying ≥1 {L} to discard."""
    if player is None:
        return False
    for p in list(player.active) + list(player.bench):
        if p is not None and p.id == _KILOWATTREL_ID:
            if any(e is EnergyKind.LIGHTNING for e in p.energies):
                return True
    return False


def damage_940(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured damage + W/R, except Voltaic Chain (board-{L} scaling prose)
    and Quick Attack (coin-flip: model the guaranteed 10, ignore the +20)."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _VOLTAIC_CHAIN:
        base = 20 + 20 * _l_energy_on_board(gs.me)
    elif attack_id == _QUICK_ATTACK:
        base = 10
    else:
        atk = cards.get_attack(attack_id) if cards is not None else None
        base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_940(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_l_hand = sum(1 for c in hand if c.id == _L_ENERGY_ID)
    n_draw_sup = sum(1 for c in hand if c.id in _DRAW_SUPPORTER_IDS_940)
    stadium_ids = {c.id for c in state.stadium} if state is not None else set()
    active_energy = len(my_active.energies) if my_active is not None else 0
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 300),
        _frac(n_l_hand, 6),                              # Electric Streamer fuel in hand
        _frac(_l_energy_on_board(me), 12),               # Voltaic Chain damage driver
        _frac(active_energy, 4),
        1.0 if _count_in_play(me, _BELLIBOLT_ID) > 0 else 0.0,   # engine online
        1.0 if _kilowattrel_can_draw(me) else 0.0,               # Flashing Draw available
        1.0 if _LEVINCIA_ID in stadium_ids else 0.0,
        _frac(_discard_resources_940(me, cards), 10),            # recycling loop depth
        _frac(n_draw_sup, 8),                                    # Lillie's + Canari in hand
        1.0 if (my_active is not None and my_active.id == _BELLIBOLT_ID) else 0.0,
    )


def reduced_940(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_l_hand = sum(1 for c in hand if c.id == _L_ENERGY_ID)
    my_active = gs.my_active
    active_energy = len(my_active.energies) if my_active is not None else 0
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(n_l_hand, 6),
        _frac(_l_energy_on_board(me), 12),
        _frac(active_energy, 4),
        _frac(_discard_resources_940(me, cards), 10),
        1.0 if _kilowattrel_can_draw(me) else 0.0,
    )


def wants_940(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """Iono's Pokémon that could use more energy. Voltorb/Bellibolt scale or need
    up to 4; Kilowattrel wants {L} for Flashing Draw + Mach Bolt; small basics
    want enough to attack."""
    n = len(pkmn.energies)
    if pkmn.id == _VOLTORB_ID:
        return 1.0 if n < 4 else 0.0        # Voltaic Chain rewards more energy
    if pkmn.id == _BELLIBOLT_ID:
        return 1.0 if n < 4 else 0.0        # Thunderous Bolt costs LLLC
    if pkmn.id == _KILOWATTREL_ID:
        return 1.0 if n < 3 else 0.0        # Mach Bolt LCC (+ Flashing Draw fuel)
    if pkmn.id in (_TADBULB_ID, _WATTREL_ID):
        return 1.0 if n < 2 else 0.0
    return 1.0 if gs.wants_energy(pkmn) else 0.0


# =============================================================================
# CINDERACE_METAL — Yoshiki Okayama's Archaludon ex / Cinderace Metal deck (M9 probe).
# =============================================================================
# A CLONABILITY PROBE only (no ship intent): built to measure this complex deck's
# offline MAIN accuracy before deciding whether a full clone attempt is worth it.
# Mechanically heavy: Boss's Orders gust (opp_target_dim=4), Duraludon's Raging
# Hammer (self-damage scaling prose), Archaludon's on-evolve energy accel, Cinderace
# Stage-2 played via its Explosiveness setup ability, Relicanth's Memory Dive.

_M_ENERGY_ID = 8
_RELICANTH_ID = 57
_DURALUDON_ID = 169
_ARCHALUDON_ID = 190
_CINDERACE_ID = 666
_BOSS_ID_C = 1182
_FULL_METAL_LAB_ID = 1244
_RAGING_HAMMER = 224          # Duraludon: 80 + 10 per damage counter on itself (prose)
_METAL_DEFENDER = 253         # Archaludon ex: 220, no-weakness-next-turn (flagged)


def _m_energy_on_board(player: ParsedPlayer | None) -> int:
    if player is None:
        return 0
    n = 0
    for p in player.active:
        if p is not None:
            n += sum(1 for e in p.energies if e is EnergyKind.METAL)
    for p in player.bench:
        n += sum(1 for e in p.energies if e is EnergyKind.METAL)
    return n


def _discard_resources_c(player: ParsedPlayer | None, cards: CardDatabase | None) -> int:
    """Basic {M} energy + Pokémon in discard (Assemble Alloy / Night Stretcher fuel)."""
    if player is None or cards is None:
        return 0
    n = 0
    for c in player.discard:
        if c.id == _M_ENERGY_ID:
            n += 1
        else:
            info = cards.get_card(c.id)
            if info is not None and info.cardType.name == "POKEMON":
                n += 1
    return n


def damage_cinderace(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured + W/R, except Raging Hammer (80 + 10 per damage counter on self)."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _RAGING_HAMMER:
        base = 80 + max(0, attacker.maxHp - attacker.hp)  # +10 per counter (=+HP lost)
    else:
        atk = cards.get_attack(attack_id) if cards is not None else None
        base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_cinderace(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_m_hand = sum(1 for c in hand if c.id == _M_ENERGY_ID)
    n_boss_hand = sum(1 for c in hand if c.id == _BOSS_ID_C)
    stadium_ids = {c.id for c in state.stadium} if state is not None else set()
    active_energy = len(my_active.energies) if my_active is not None else 0
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 300),
        _frac(n_m_hand, 6),
        _frac(_m_energy_on_board(me), 12),
        _frac(active_energy, 4),
        1.0 if _count_in_play(me, _ARCHALUDON_ID) > 0 else 0.0,
        1.0 if _count_in_play(me, _CINDERACE_ID) > 0 else 0.0,
        1.0 if _FULL_METAL_LAB_ID in stadium_ids else 0.0,
        _frac(n_boss_hand, 2),
        _frac(_discard_resources_c(me, cards), 10),
    )


def reduced_cinderace(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_m_hand = sum(1 for c in hand if c.id == _M_ENERGY_ID)
    n_boss_hand = sum(1 for c in hand if c.id == _BOSS_ID_C)
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(n_m_hand, 6),
        _frac(_m_energy_on_board(me), 12),
        _frac(_discard_resources_c(me, cards), 10),
        _frac(n_boss_hand, 2),
    )


def wants_cinderace(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    n = len(pkmn.energies)
    if pkmn.id == _ARCHALUDON_ID:
        return 1.0 if n < 3 else 0.0     # Metal Defender MMM
    if pkmn.id == _DURALUDON_ID:
        return 1.0 if n < 3 else 0.0     # Raging Hammer MMC
    if pkmn.id in (_CINDERACE_ID, _RELICANTH_ID):
        return 1.0 if n < 2 else 0.0
    return 1.0 if gs.wants_energy(pkmn) else 0.0


# =============================================================================
# KANGASKHAN_1052 — M20 clone of 懒惰的金枪鱼's Mega Kangaskhan ex / Crustle deck
# (Kaggle rank #32, ladder ~1052). Two attackers with very different roles:
#   * Mega Kangaskhan ex (756): 300 HP Basic, Rapid-Fire Combo ●●● 200 base, then
#     "flip until tails, +50 per heads". megaEx -> worth THREE prizes when KO'd.
#   * Crustle (345): 150 HP wall whose Ability prevents ALL damage from the
#     opponent's *ex* attackers, and Superb Scissors {G}●● 120 that ignores effects
#     on the defender. Crustle is the answer to ex decks; Kangaskhan is the clock.
# Energy is 12 special + 1 basic {G}: Mist/Spiky provide {C} (so Kangaskhan's ●●●
# is payable by anything), while only Grow Grass + the single basic {G} pay
# Superb Scissors' {G} — a real scarcity the featurizer should see.
# =============================================================================

_KANGASKHAN_ID = 756
_CRUSTLE_ID = 345
_DWEBBLE_ID = 344
_RAPID_FIRE = 1092           # Kangaskhan: 200 + coin-flip 50s (we model the guaranteed 200)
_SUPERB_SCISSORS = 479       # Crustle: 120, ignores effects on the defender
_ASCENSION = 478             # Dwebble: 0 dmg, searches its evolution
_GROW_GRASS_ID = 18          # special {G}, +20 HP to the {G} Pokémon it is on
_MIST_ENERGY_ID = 11         # special {C}, blocks attack EFFECTS (not damage)
_SPIKY_ENERGY_ID = 14        # special {C}, 2 counters back on the attacker
_G_BASIC_ID = 1
_HEROS_CAPE_ID = 1159        # +100 HP -> a 400 HP Kangaskhan
_BATTLE_CAGE_ID = 1264
_JUMBO_ICE_CREAM_ID = 1147   # heal 80 if the Active has 3+ energy
_BOSS_ID_K = 1182


def _g_energy_on(pkmn: ParsedPokemon | None) -> int:
    """{G}-providing energy attached (Grow Grass or basic {G}) — pays Superb Scissors."""
    if pkmn is None:
        return 0
    return sum(1 for e in pkmn.energies if e is EnergyKind.GRASS)


def _g_sources_in_hand(player: ParsedPlayer | None) -> int:
    if player is None or player.hand is None:
        return 0
    return sum(1 for c in player.hand if c.id in (_GROW_GRASS_ID, _G_BASIC_ID))


def _energy_in_hand_k(player: ParsedPlayer | None) -> int:
    if player is None or player.hand is None:
        return 0
    ids = (_GROW_GRASS_ID, _G_BASIC_ID, _MIST_ENERGY_ID, _SPIKY_ENERGY_ID)
    return sum(1 for c in player.hand if c.id in ids)


def _count_hand_k(player: ParsedPlayer | None, card_id: int) -> int:
    if player is None or player.hand is None:
        return 0
    return sum(1 for c in player.hand if c.id == card_id)


def damage_kangaskhan(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured damage + W/R, with Rapid-Fire Combo modelled at its GUARANTEED 200.

    Rapid-Fire Combo reads "flip a coin until you get tails; +50 damage for each
    heads" — expected value 250, but the guaranteed floor is 200. We keep the floor,
    matching how ``damage_940`` treats Quick Attack's coin flip: the featurizer's
    lethal/overkill slots must never claim a KO the attack cannot actually guarantee.
    Superb Scissors ignores *effects* on the defender, which this model never applied
    anyway, so it stays plain structured damage."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _RAPID_FIRE:
        base = 200
    elif attack_id == _ASCENSION:
        base = 0
    else:
        atk = cards.get_attack(attack_id) if cards is not None else None
        base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_kangaskhan(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    active_is_kanga = bool(my_active is not None and my_active.id == _KANGASKHAN_ID)
    active_is_crustle = bool(my_active is not None and my_active.id == _CRUSTLE_ID)
    active_energy = len(my_active.energies) if my_active is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    stadium_ids = {c.id for c in state.stadium} if state is not None else set()
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(gs.hand_size, 10),
        _frac(opp.handCount if opp else 0, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        # the matchup axis: Crustle blanks ex attackers, so "is the threat an ex?"
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 340),
        # who is holding the Active spot, and is it charged?
        1.0 if active_is_kanga else 0.0,
        1.0 if active_is_crustle else 0.0,
        _frac(active_energy, 3),
        _frac(_g_energy_on(my_active), 2),
        # board presence
        _frac(_count_in_play(me, _KANGASKHAN_ID), 3),
        _frac(_count_in_play(me, _CRUSTLE_ID), 3),
        _frac(_count_in_play(me, _DWEBBLE_ID), 3),
        # resources: {G} scarcity gates Superb Scissors, total energy gates Kangaskhan
        _frac(_g_sources_in_hand(me), 4),
        _frac(_energy_in_hand_k(me), 6),
        _frac(_count_hand_k(me, _BOSS_ID_K), 2),
        _frac(_count_hand_k(me, _JUMBO_ICE_CREAM_ID), 4),
        1.0 if _BATTLE_CAGE_ID in stadium_ids else 0.0,
        1.0 if any(t.id == _HEROS_CAPE_ID for t in (my_active.tools if my_active else ())) else 0.0,
    )


def reduced_kangaskhan(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    my_active = gs.my_active
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(gs.hand_size, 10),
        _frac(opp.handCount if opp else 0, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(len(my_active.energies) if my_active else 0, 3),
        _frac(_g_energy_on(my_active), 2),
        _frac(_g_sources_in_hand(me), 4),
        _frac(_energy_in_hand_k(me), 6),
        _frac(_count_in_play(me, _CRUSTLE_ID), 3),
    )


def wants_kangaskhan(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """Energy demand per attacker, from the real costs.

    Kangaskhan's ●●● is colour-blind (any 3 energy), so it simply wants a 3rd.
    Crustle's {G}●● needs a {G} specifically: it still 'wants' energy while it is
    short of 3 OR still missing its {G}, because a Crustle with 3 colourless cannot
    attack at all. Dwebble only ever needs the single ● for Ascension (0 damage, so
    the generic wants_energy would say no — but that attach is how the line evolves)."""
    n = len(pkmn.energies)
    if pkmn.id == _KANGASKHAN_ID:
        return 1.0 if n < 3 else 0.0
    if pkmn.id == _CRUSTLE_ID:
        return 1.0 if (n < 3 or _g_energy_on(pkmn) < 1) else 0.0
    if pkmn.id == _DWEBBLE_ID:
        return 1.0 if n < 1 else 0.0
    return 1.0 if gs.wants_energy(pkmn) else 0.0


# =============================================================================
# Profile registry
# =============================================================================

_SnapshotFn = Callable[[ParsedObservation, GameState, "CardDatabase | None"], tuple]
_DamageFn = Callable[["int | None", GameState, "CardDatabase | None"], int]
_WantsFn = Callable[[ParsedPokemon, GameState, "CardDatabase | None"], float]


@dataclass(frozen=True)
class DeckProfile:
    """Deck-specific vocabularies + scalar extractors for the featurizer."""

    name: str
    deck_ids: tuple[int, ...]
    attack_ids: tuple[int, ...]
    target_pokemon_ids: tuple[int, ...]
    snapshot_len: int
    reduced_len: int
    opp_target_dim: int
    flagged_attack_id: int | None
    snapshot_fn: _SnapshotFn
    reduced_fn: _SnapshotFn
    damage_fn: _DamageFn
    wants_fn: _WantsFn

    @cached_property
    def deck_index(self) -> dict[int, int]:
        return {cid: i for i, cid in enumerate(self.deck_ids)}

    @cached_property
    def attack_index(self) -> dict[int, int]:
        return {aid: i for i, aid in enumerate(self.attack_ids)}

    @cached_property
    def target_index(self) -> dict[int, int]:
        return {cid: i for i, cid in enumerate(self.target_pokemon_ids)}

    @property
    def b_len(self) -> int:
        return len(self.deck_ids) + 1

    @property
    def c_len(self) -> int:
        return len(self.attack_ids) + 1

    @property
    def d_len(self) -> int:
        # area(4) + target one-hot(len+1) + energy/damaged/wants(3) + opp-target
        return 8 + len(self.target_pokemon_ids) + self.opp_target_dim

    @property
    def feature_dim(self) -> int:
        return (
            _A + self.b_len + self.c_len + self.d_len + _E + _F + self.snapshot_len
            + _A * self.snapshot_len + self.b_len * self.reduced_len
        )


TR_650 = DeckProfile(
    name="TR_650",
    deck_ids=(1, 400, 401, 433, 434, 1094, 1097, 1121, 1134, 1152, 1158, 1213, 1216, 1220, 1257),
    attack_ids=(559, 560, 611, 612),
    target_pokemon_ids=(400, 401, 433, 434),
    snapshot_len=18,
    reduced_len=6,
    opp_target_dim=0,
    flagged_attack_id=611,
    snapshot_fn=snapshot_650,
    reduced_fn=reduced_650,
    damage_fn=damage_650,
    wants_fn=wants_650,
)

LUCARIO_800 = DeckProfile(
    name="LUCARIO_800",
    deck_ids=(6, 673, 674, 675, 676, 677, 678, 1102, 1123, 1141, 1142, 1152, 1159, 1182, 1192, 1227, 1252),
    attack_ids=(976, 977, 978, 979, 980, 981, 982, 983),
    target_pokemon_ids=(673, 674, 675, 676, 677, 678),
    snapshot_len=21,
    reduced_len=7,
    opp_target_dim=4,
    flagged_attack_id=983,  # Mega Brave (can't-reuse-next-turn), mirrors 650's Chiming slot
    snapshot_fn=snapshot_800,
    reduced_fn=reduced_800,
    damage_fn=damage_800,
    wants_fn=wants_800,
)

#: LUCARIO_800 + hand-composition & active-energy features (M8.1). Same vocab,
#: bigger snapshot/reduced (dim 633). A separate profile so bc_800_v1.json still
#: loads reproducibly; the new weights pin "LUCARIO_800_V2".
LUCARIO_800_V2 = DeckProfile(
    name="LUCARIO_800_V2",
    deck_ids=LUCARIO_800.deck_ids,
    attack_ids=LUCARIO_800.attack_ids,
    target_pokemon_ids=LUCARIO_800.target_pokemon_ids,
    snapshot_len=31,
    reduced_len=9,
    opp_target_dim=4,
    flagged_attack_id=983,
    snapshot_fn=snapshot_800v2,
    reduced_fn=reduced_800v2,
    damage_fn=damage_800,
    wants_fn=wants_800,
)

#: BELLIBOLT_940 — M9 clone of kenN2439's Iono's Bellibolt ex Lightning deck.
#: Hand-composition + board-energy features from the start (M8.1 lesson). dim 514.
BELLIBOLT_940 = DeckProfile(
    name="BELLIBOLT_940",
    deck_ids=(4, 265, 268, 269, 270, 271, 1086, 1097, 1110, 1118, 1121, 1152, 1227, 1233, 1254),
    attack_ids=(363, 367, 368, 369, 370),
    target_pokemon_ids=(265, 268, 269, 270, 271),
    snapshot_len=24,
    reduced_len=9,
    opp_target_dim=0,
    flagged_attack_id=368,  # Thunderous Bolt: can't attack next turn
    snapshot_fn=snapshot_940,
    reduced_fn=reduced_940,
    damage_fn=damage_940,
    wants_fn=wants_940,
)

#: CINDERACE_METAL — M9 clonability PROBE of Yoshiki's Archaludon/Cinderace deck.
#: opp_target_dim=4 (Boss's Orders gust). Built to measure offline MAIN accuracy
#: before committing to a full clone; not intended to ship as-is.
CINDERACE_METAL = DeckProfile(
    name="CINDERACE_METAL",
    deck_ids=(8, 57, 169, 190, 666, 1097, 1121, 1122, 1147, 1152, 1159, 1182, 1185, 1227, 1244),
    attack_ids=(61, 223, 224, 253, 965),
    target_pokemon_ids=(57, 169, 190, 666),
    snapshot_len=23,
    reduced_len=8,
    opp_target_dim=4,
    flagged_attack_id=253,  # Metal Defender
    snapshot_fn=snapshot_cinderace,
    reduced_fn=reduced_cinderace,
    damage_fn=damage_cinderace,
    wants_fn=wants_cinderace,
)

#: KANGASKHAN_1052 — M20 clone of 懒惰的金枪鱼 (rank #32, ~1052), the first
#: hand-authored profile for a >1000-elo teacher. opp_target_dim=4 (Boss's Orders).
#: flagged attack = Rapid-Fire Combo, the coin-flip attack whose modelled damage is
#: deliberately its guaranteed floor (see damage_kangaskhan).
KANGASKHAN_1052 = DeckProfile(
    name="KANGASKHAN_1052",
    deck_ids=(1, 11, 14, 18, 344, 345, 756, 1086, 1087, 1122, 1123, 1147, 1159,
              1182, 1197, 1225, 1227, 1264),
    attack_ids=(478, 479, 1092),
    target_pokemon_ids=(344, 345, 756),
    snapshot_len=28,
    reduced_len=9,
    opp_target_dim=4,
    flagged_attack_id=1092,
    snapshot_fn=snapshot_kangaskhan,
    reduced_fn=reduced_kangaskhan,
    damage_fn=damage_kangaskhan,
    wants_fn=wants_kangaskhan,
)

# =============================================================================
# ITF_ESYS_KASU — M22 clone of ITF_Esys_Kasu's Mega Starmie ex / Cinderace deck.
# =============================================================================
# The FIRST candidate whose generic-profile clone ever beat imitation-v1 on the field
# gauntlet (+0.036/+0.043 vs v1, +0.037 vs the KANGASKHAN_1052 clone; m22_findings.md),
# despite the LOWEST clonability of its screening round (0.530 on only 63 replay games).
# Mega Starmie ex (330 HP megaEx -> 3 prizes; Jetting Blow {W} 120 + 50 splash to a
# benched opponent Pokemon; Nebula Beam {C}{C}{C} 210, VERIFIED by engine skill text to
# ignore Weakness/Resistance and opponent-Active effects) is the finisher, backed by a
# Staryu line, 4x Cinderace (played face-down via its Explosiveness ability, so it needs
# no pre-evolution in the deck; Turbo Flare {C} 50 also searches 3 basic Energy onto the
# bench), 4x Crushing Hammer (coin-flip energy denial), 1x Boss's Orders (gust), and
# Ignition Energy (VERIFIED: provides {C}, or {C}{C}{C} if attached to an Evolution
# Pokemon — Mega Starmie ex and Cinderace both qualify, so a single Ignition attach can
# outright pay Nebula Beam's cost; discards itself at end of turn). All of this is read
# from the engine's own `CardInfo.skills[].text` (verified, not guessed — the M17
# lesson): earlier passes over `deck_profiles.py` only checked `.text` on POKEMON attacks
# and read `None` for items/tools/special-energy, which is the WRONG attribute — their
# real effect text lives in `.skills`, and it was there the whole time.

_STARMIE_ID = 1031
_ITF_CINDERACE_ID = 666
_ITF_STARYU_ID = 1030
_NEBULA_BEAM = 1488          # {C}{C}{C} 210, ignores W/R + opp-Active effects (VERIFIED)
_JETTING_BLOW = 1487         # {W} 120 (+50 to a benched opponent, W/R-exempt; unmodelled)
_TURBO_FLARE = 965           # {C} 50, also searches+attaches up to 3 basic Energy
_WATER_GUN = 1486            # {W} 20, Staryu
_ITF_WATER_ENERGY_ID = 3     # Basic {W} Energy
_IGNITION_ENERGY_ID = 17     # {C}, or {C}{C}{C} on an Evolution Pokemon (VERIFIED)
_ITF_BOSS_ID = 1182
_CRUSHING_HAMMER_ID = 1120
_ITF_HEROS_CAPE_ID = 1159


def damage_itf(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured damage + W/R, except Nebula Beam.

    Nebula Beam's own text (engine-verified) reads "This attack's damage isn't affected
    by Weakness or Resistance, or by any effects on your opponent's Active Pokemon" — so
    it is modelled at its flat 210, skipping the W/R adjustment every other attack in this
    deck gets. Jetting Blow's extra 50-to-bench splash and Turbo Flare's energy search are
    real effects, but neither changes Active-vs-Active KO math, which is this function's
    whole scope (matching every other profile's treatment of non-damage side effects)."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    atk = cards.get_attack(attack_id) if cards is not None else None
    base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    if attack_id == _NEBULA_BEAM:
        return base
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def _discard_resources_itf(player: ParsedPlayer | None, cards: CardDatabase | None) -> int:
    """Pokemon + basic {W} energy in discard — what Night Stretcher can recover."""
    if player is None or cards is None:
        return 0
    n = 0
    for c in player.discard:
        if c.id == _ITF_WATER_ENERGY_ID:
            n += 1
        else:
            info = cards.get_card(c.id)
            if info is not None and info.cardType.name == "POKEMON":
                n += 1
    return n


def snapshot_itf(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    active_energy = len(my_active.energies) if my_active is not None else 0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_water_hand = sum(1 for c in hand if c.id == _ITF_WATER_ENERGY_ID)
    n_ignition_hand = sum(1 for c in hand if c.id == _IGNITION_ENERGY_ID)
    n_boss_hand = sum(1 for c in hand if c.id == _ITF_BOSS_ID)
    n_hammer_hand = sum(1 for c in hand if c.id == _CRUSHING_HAMMER_ID)
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 330),
        # who is Active, and how charged
        1.0 if (my_active is not None and my_active.id == _STARMIE_ID) else 0.0,
        1.0 if (my_active is not None and my_active.id == _ITF_CINDERACE_ID) else 0.0,
        _frac(active_energy, 4),
        1.0 if _count_in_play(me, _STARMIE_ID) > 0 else 0.0,
        1.0 if any(t.id == _ITF_HEROS_CAPE_ID for t in (my_active.tools if my_active else ())) else 0.0,
        # fuel / disruption resources in hand
        _frac(n_water_hand, 6),
        _frac(n_ignition_hand, 4),
        _frac(n_boss_hand, 2),
        _frac(n_hammer_hand, 4),
        _frac(_discard_resources_itf(me, cards), 10),
    )


def reduced_itf(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    my_active = gs.my_active
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    active_energy = len(my_active.energies) if my_active is not None else 0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_boss_hand = sum(1 for c in hand if c.id == _ITF_BOSS_ID)
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(active_energy, 4),
        1.0 if _count_in_play(me, _STARMIE_ID) > 0 else 0.0,
        _frac(_discard_resources_itf(me, cards), 10),
        _frac(n_boss_hand, 2),
    )


def wants_itf(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """No prose-conditional costs in this deck (unlike Bellibolt/Kangaskhan) — every
    attack's payability is exactly what GameState.energy_pays already computes from the
    observation's own energies tuple (which already reflects Ignition Energy's engine-
    resolved {C}/{C}{C}{C} value), so the generic fallback is correct as-is."""
    return 1.0 if gs.wants_energy(pkmn) else 0.0


#: ITF_ESYS_KASU — M22 clone of ITF_Esys_Kasu (submission 54735267). opp_target_dim=4
#: (Boss's Orders). flagged attack = Nebula Beam, the only attack whose damage math
#: deviates from the standard weakness/resistance formula (see damage_itf).
ITF_ESYS_KASU = DeckProfile(
    name="ITF_ESYS_KASU",
    deck_ids=(3, 17, 666, 1030, 1031, 1086, 1097, 1120, 1121, 1122, 1145, 1159,
              1182, 1189, 1223, 1225, 1227, 1229),
    attack_ids=(965, 1486, 1487, 1488),
    target_pokemon_ids=(666, 1030, 1031),
    snapshot_len=25,
    reduced_len=8,
    opp_target_dim=4,
    flagged_attack_id=1488,
    snapshot_fn=snapshot_itf,
    reduced_fn=reduced_itf,
    damage_fn=damage_itf,
    wants_fn=wants_itf,
)


# =============================================================================
# THIRD_PTCG — M23 clone of THIRD PTCG Club's Team Rocket Rush + Mewtwo ex deck.
# =============================================================================
# THIRD PTCG Club is Kaggle rank #12 (ladder ~1117). Its deck is v1's exact Team
# Rocket Rush swarm (Tarountula 400 -> Spidops 401, Rocket Rush = 30 x TR-in-play)
# UPGRADED with two heavy hitters the basic swarm lacked: Team Rocket's Mewtwo ex
# (431; Erasure Ball {P}{P}{C} 160, +60 per Bench energy discarded up to 2 -> caps 280;
# Power Saver: can't attack unless 4+ TR Pokemon in play) and Articuno (414; Dark Frost
# {W}{C}{C} 60, +60 if Team Rocket's Energy attached). Every Pokemon here is a "Team
# Rocket's" Pokemon, so ALL of them count for Rocket Rush's 30x. This is the archetype
# v1 already clones best (686 ladder) — the damage core is `damage_650` verbatim, plus
# three engine-verified prose corrections read from `CardInfo.skills[]/attack.text`
# (the M22 lesson: effect text is NOT on the None-valued `.text` of items). opp_target_dim=4
# for Team Rocket's Giovanni (a gust: switches in an opponent's benched Pokemon).

_TR3_POKEMON_IDS = frozenset({400, 401, 414, 431, 432, 463})  # all count for Rocket Rush
_TR3_TAROUNTULA = 400
_TR3_SPIDOPS = 401
_TR3_ARTICUNO = 414
_TR3_MEWTWO_ID = 431
_TR3_WOBBUFFET = 432
_TR3_MURKROW = 463
_TR3_ROCKET_RUSH = 560     # 30 x TR-in-play (same as v1 TR_650)
_TR3_TAKE_DOWN = 559       # 30 flat (+10 self)
_TR3_DARK_FROST = 583      # 60, +60 if TR Energy attached
_TR3_ERASURE_BALL = 608    # 160 base, +60 per bench energy discarded (<=2) -> floor 160
_TR3_ROCKET_MIRROR = 609   # 0 (moves damage counters)
_TR3_HEADBUTT = 610        # 70 flat
_TR3_DECEIT = 652          # 0 (search)
_TR3_TORMENT = 653         # 30 flat
_TR3_ENERGY_ID = 15        # Team Rocket's Energy (special; provides {P}/{D}, gates Dark Frost)
_TR3_G_ENERGY_ID = 1
_TR3_P_ENERGY_ID = 5
_TR3_GIOVANNI_ID = 1218    # gust supporter
_TR3_HEROS_CAPE_ID = 1159
_TR3_BRAVE_BANGLE_ID = 1175  # +30 vs ex Active if holder is non-ex (before W/R)


def _tr3_in_play(player: ParsedPlayer | None) -> int:
    if player is None:
        return 0
    count = 0
    for p in player.active:
        if p is not None and p.id in _TR3_POKEMON_IDS:
            count += 1
    for p in player.bench:
        if p.id in _TR3_POKEMON_IDS:
            count += 1
    return count


def _tr3_has_rocket_energy(pkmn: ParsedPokemon | None) -> bool:
    """Team Rocket's Energy (id 15) attached — gates Dark Frost's +60. The engine
    surfaces it as an attached energy providing {P}/{D}; we detect it by the special
    energy's own card id when present in the attacker's energyCards, else fall back to
    a {P}/{D} energy presence heuristic."""
    if pkmn is None:
        return False
    for c in getattr(pkmn, "energyCards", ()) or ():
        if getattr(c, "id", None) == _TR3_ENERGY_ID:
            return True
    return False


def damage_third(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Rocket Rush 30x TR (v1's math) + engine-verified prose for Mewtwo/Articuno.

    Erasure Ball is modelled at its GUARANTEED 160 (the +60/discard is optional and
    self-inflicted, so — matching damage_940/damage_kangaskhan's coin-flip convention —
    the featurizer never claims a KO the attack cannot guarantee). Brave Bangle's +30 vs
    an ex Active and Team Rocket's Energy's +60 on Dark Frost are applied before W/R."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _TR3_ROCKET_RUSH:
        base = 30 * _tr3_in_play(gs.me)
    elif attack_id == _TR3_TAKE_DOWN:
        base = 30
    elif attack_id == _TR3_TORMENT:
        base = 30
    elif attack_id == _TR3_HEADBUTT:
        base = 70
    elif attack_id == _TR3_ERASURE_BALL:
        base = 160
    elif attack_id == _TR3_DARK_FROST:
        base = 60 + (60 if _tr3_has_rocket_energy(attacker) else 0)
    elif attack_id in (_TR3_ROCKET_MIRROR, _TR3_DECEIT):
        base = 0
    else:
        atk = cards.get_attack(attack_id) if cards is not None else None
        base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    # Brave Bangle: +30 to an opponent's ex Active if the holder is non-ex (before W/R).
    if (def_info is not None and (def_info.ex or def_info.megaEx)
            and atk_info is not None and not (atk_info.ex or atk_info.megaEx)
            and any(t.id == _TR3_BRAVE_BANGLE_ID for t in attacker.tools)):
        base += 30
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_third(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    tr = _tr3_in_play(me)
    active_energy = len(my_active.energies) if my_active is not None else 0
    hand = me.hand if (me is not None and me.hand is not None) else ()
    n_tr_energy_hand = sum(1 for c in hand if c.id == _TR3_ENERGY_ID)
    n_basic_energy_hand = sum(1 for c in hand if c.id in (_TR3_G_ENERGY_ID, _TR3_P_ENERGY_ID))
    n_giovanni_hand = sum(1 for c in hand if c.id == _TR3_GIOVANNI_ID)
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 300),
        # the Rocket Rush driver + its projected damage
        _frac(tr, 6),
        _frac(30 * tr, 200),
        # Mewtwo readiness: Power Saver needs 4+ TR Pokemon in play
        1.0 if tr >= 4 else 0.0,
        1.0 if _count_in_play(me, _TR3_MEWTWO_ID) > 0 else 0.0,
        1.0 if (my_active is not None and my_active.id == _TR3_MEWTWO_ID) else 0.0,
        1.0 if (my_active is not None and my_active.id == _TR3_ARTICUNO) else 0.0,
        # resources / fuel in hand
        _frac(active_energy, 4),
        _frac(n_tr_energy_hand, 4),
        _frac(n_basic_energy_hand, 6),
        _frac(n_giovanni_hand, 2),
    )


def reduced_third(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    tr = _tr3_in_play(me)
    my_active = gs.my_active
    active_energy = len(my_active.energies) if my_active is not None else 0
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(tr, 6),
        1.0 if tr >= 4 else 0.0,
        _frac(active_energy, 4),
    )


def wants_third(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """Energy demand per attacker. Rocket Rush (Spidops) costs {G}{C}; Mewtwo's Erasure
    Ball {P}{P}{C}; Articuno's Dark Frost {W}{C}{C}; small basics want enough to attack.
    Spidops keeps wanting energy to enable Rocket Rush swings; Mewtwo wants up to 3."""
    n = len(pkmn.energies)
    if pkmn.id == _TR3_MEWTWO_ID:
        return 1.0 if n < 3 else 0.0
    if pkmn.id == _TR3_ARTICUNO:
        return 1.0 if n < 3 else 0.0
    if pkmn.id == _TR3_SPIDOPS:
        return 1.0 if n < 2 else 0.0
    if pkmn.id in (_TR3_TAROUNTULA, _TR3_WOBBUFFET, _TR3_MURKROW):
        return 1.0 if n < 2 else 0.0
    return 1.0 if gs.wants_energy(pkmn) else 0.0


#: THIRD_PTCG — M23 clone of THIRD PTCG Club (submission 54840044, rank #12 ~1117).
#: v1's Team Rocket Rush core + Mewtwo ex / Articuno. flagged attack = Erasure Ball
#: (Mewtwo's optional-discard nuke, modelled at its guaranteed 160). opp_target_dim=4
#: for Team Rocket's Giovanni (gust).
THIRD_PTCG = DeckProfile(
    name="THIRD_PTCG",
    deck_ids=(1, 5, 15, 400, 401, 414, 431, 432, 463, 1094, 1097, 1119, 1134,
              1152, 1159, 1175, 1216, 1217, 1218, 1219, 1220, 1227, 1257),
    attack_ids=(559, 560, 583, 608, 609, 610, 652, 653),
    target_pokemon_ids=(400, 401, 414, 431, 432, 463),
    snapshot_len=25,
    reduced_len=7,
    opp_target_dim=4,
    flagged_attack_id=608,
    snapshot_fn=snapshot_third,
    reduced_fn=reduced_third,
    damage_fn=damage_third,
    wants_fn=wants_third,
)


# =============================================================================
# ALAKAZAM — M28 clone of Yushin Ito (#1) Stage-2 Alakazam combo/control deck.
# The win condition is Alakazam's "Powerful Hand" (1072, {P}): "place 2 damage
# counters on the opponent's Active for each card in your hand" = 20 * hand_size,
# PROSE damage the generic model reads as 0. Because it PLACES counters, Weakness/
# Resistance do NOT apply. Fezandipiti's "Cruel Arrow" (183, {C}{C}{C}) does a flat
# 100 to a chosen Pokémon (also prose-0 in the DB). Hand size is the deck's core
# state variable — it drives both the draw sequencing and the attack's damage — so
# the snapshot foregrounds it. Bodies: Abra 741 -> Kadabra 742 -> Alakazam 743
# (Rare Candy 1079 skips the middle), plus Dunsparce 305 / Dudunsparce 66 and
# Fezandipiti ex 140 as draw engines, Shaymin 343 for bench protection.
# =============================================================================

_ALAKAZAM_ID = 743
_KADABRA_ID = 742
_ABRA_ID = 741
_DUDUNSPARCE_ID = 66
_DUNSPARCE_ID = 305
_FEZANDIPITI_ID = 140
_SHAYMIN_ID = 343
_POWERFUL_HAND = 1072        # 20 * hand_size, placed counters (no W/R)
_CRUEL_ARROW = 183           # flat 100 to a target
_RARE_CANDY_A = 1079
_BOSS_ID_A = 1182
_DAWN_ID = 1231              # searches 3 cards; the deck's main tutor (M37 arm E)


def _p_energy_on(pkmn: ParsedPokemon | None) -> int:
    """{P}-providing energy attached (basic {P} or Telepath Psychic) — pays Powerful Hand."""
    if pkmn is None:
        return 0
    return sum(1 for e in pkmn.energies if e is EnergyKind.PSYCHIC)


def _count_hand_a(player: ParsedPlayer | None, card_id: int) -> int:
    if player is None or player.hand is None:
        return 0
    return sum(1 for c in player.hand if c.id == card_id)


def damage_alakazam(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured damage + the two prose attacks, with correct rule handling.

    Powerful Hand PLACES 2 damage counters (20 HP) per card in hand — placed counters
    ignore Weakness/Resistance, so it is a flat 20*hand_size. Cruel Arrow deals a flat
    100 to a chosen Pokémon (W/R not applied to Benched targets, and we do not know the
    target here, so keep it flat). Everything else is plain structured damage + W/R,
    matching the other profiles."""
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id == _POWERFUL_HAND:
        return max(0, 20 * gs.hand_size)
    if attack_id == _CRUEL_ARROW:
        return 100
    atk = cards.get_attack(attack_id) if cards is not None else None
    base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_alakazam(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    active_is_alakazam = bool(my_active is not None and my_active.id == _ALAKAZAM_ID)
    active_is_kadabra = bool(my_active is not None and my_active.id == _KADABRA_ID)
    active_energy = len(my_active.energies) if my_active is not None else 0
    ph_dmg = 20 * gs.hand_size
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    dspce = _count_in_play(me, _DUNSPARCE_ID) + _count_in_play(me, _DUDUNSPARCE_ID)
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        # hand size is the CORE variable: it powers both the draw plan and the attack
        _frac(gs.hand_size, 12),
        _frac(opp.handCount if opp else 0, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 340),
        # who holds the Active spot + is it charged to fire Powerful Hand ({P})?
        1.0 if active_is_alakazam else 0.0,
        1.0 if active_is_kadabra else 0.0,
        _frac(active_energy, 3),
        1.0 if _p_energy_on(my_active) >= 1 else 0.0,
        # board presence of the line + the engines
        _frac(_count_in_play(me, _ALAKAZAM_ID), 3),
        _frac(_count_in_play(me, _KADABRA_ID), 3),
        _frac(_count_in_play(me, _ABRA_ID), 3),
        1.0 if _count_in_play(me, _SHAYMIN_ID) >= 1 else 0.0,   # bench protection online
        1.0 if _count_in_play(me, _FEZANDIPITI_ID) >= 1 else 0.0,
        _frac(dspce, 3),
        # resources in hand
        _frac(_count_hand_a(me, _RARE_CANDY_A), 3),
        1.0 if _count_hand_a(me, _ALAKAZAM_ID) >= 1 else 0.0,
        _frac(_count_hand_a(me, _BOSS_ID_A), 2),
        # combo-lethal readiness: does Powerful Hand already KO the Active?
        1.0 if (opp_hp > 0 and ph_dmg >= opp_hp) else 0.0,
    )


def reduced_alakazam(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    my_active = gs.my_active
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(gs.hand_size, 12),
        _frac(opp.handCount if opp else 0, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(len(my_active.energies) if my_active else 0, 3),
        1.0 if _p_energy_on(my_active) >= 1 else 0.0,
        _frac(_count_in_play(me, _ALAKAZAM_ID), 3),
        _frac(_count_hand_a(me, _RARE_CANDY_A), 3),
        _frac(opp_hp, 340),
    )


def wants_alakazam(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """Energy demand per attacker from the real costs. The Alakazam line attacks for a
    single {P} (Powerful Hand / Super Psy Bolt / Teleportation Attack), so they want an
    energy only while they have no {P} attached. Dudunsparce/Fezandipiti need 3 {C};
    Dunsparce needs 2 for Ram. Shaymin is played for its Ability, not its attack."""
    n = len(pkmn.energies)
    if pkmn.id in (_ALAKAZAM_ID, _KADABRA_ID, _ABRA_ID):
        return 1.0 if _p_energy_on(pkmn) < 1 else 0.0
    if pkmn.id in (_DUDUNSPARCE_ID, _FEZANDIPITI_ID):
        return 1.0 if n < 3 else 0.0
    if pkmn.id == _DUNSPARCE_ID:
        return 1.0 if n < 2 else 0.0
    return 1.0 if gs.wants_energy(pkmn) else 0.0


#: The deck's 22 unique card ids, shared by ALAKAZAM and ALAKAZAM_V2 (V2's G1 group is a
#: per-card hand-composition vector over exactly these, so they must not drift apart).
_ALAKAZAM_DECK_IDS = (5, 13, 19, 66, 140, 305, 343, 741, 742, 743, 1079, 1081, 1086,
                      1097, 1129, 1152, 1182, 1184, 1197, 1225, 1231, 1266)

#: ALAKAZAM — M28 clone of Yushin Ito (#1). opp_target_dim=4 (Boss's Orders gust);
#: flagged attack = Powerful Hand, the prose-scaled win condition.
ALAKAZAM = DeckProfile(
    name="ALAKAZAM",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=29,
    reduced_len=9,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam,
    reduced_fn=reduced_alakazam,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


# =============================================================================
# ALAKAZAM_V2 — M31 Track E: feature enrichment (dim 1269).
# =============================================================================
# Data volume and model capacity were both measured SATURATED (1000->1284 episodes
# left held-out fidelity flat; h=48->h=96 moved it +0.006, inside the seed spread),
# so feature expressiveness is the remaining lever — and it is the #1 lesson of both
# external solution corpora (Lux AI S3 top-10, FIDE chess top-5).
#
# The concrete blindness this fixes: `hand_size` was a SINGLE SCALAR, so a 7-card hand
# of {Rare Candy, Alakazam, energy, ...} (combo ready) and a 7-card hand of loose energy
# (dead) were IDENTICAL to the model. A lone continuous scalar also only supports a
# MONOTONIC response, while the real strategy is non-monotonic: build the hand by
# drawing, then dump it into Powerful Hand (20 x hand_size).
#
# Each snapshot slot costs 13 dims (1 raw + 12 for the a(x)s option-type interaction), so
# the 47 new features cost 611. The interaction is the point: a binned hand_size gets
# option-type-conditioned copies for free (its effect when PLAYing differs from ATTACKing).
#
# Groups are contiguous so the offline trainer can ablate by zeroing column ranges
# (see ALAKAZAM_V2_GROUPS) instead of needing one DeckProfile per variant.
# =============================================================================

_HAND_BINS = (2, 4, 6, 8, 10)      # -> 6 buckets
_OPPHP_BINS = (60, 120, 180, 250)  # -> 5 buckets
_TURN_BINS = (2, 5, 9, 14)         # -> 5 buckets

#: Contiguous [start, end) slices of the V2 snapshot, for masked ablation.
ALAKAZAM_V2_GROUPS = {
    "G1": (29, 51),   # 22 — per-card hand composition
    "G2": (51, 67),   # 16 — binned (dual) encodings
    "G3": (67, 72),   #  5 — relative / differential
    "G4": (72, 76),   #  4 — combo / threat composites
}


def _in_play_v2(player: ParsedPlayer | None) -> list[ParsedPokemon]:
    if player is None:
        return []
    out = [p for p in player.active if p is not None]
    out.extend(player.bench)
    return out


def _hand_counts_v2(player: ParsedPlayer | None) -> dict[int, int]:
    out: dict[int, int] = {}
    if player is None or player.hand is None:
        return out
    for c in player.hand:
        out[c.id] = out.get(c.id, 0) + 1
    return out


def _signed(num: float, den: float) -> float:
    """Differential normalised to [-1, 1] (``_frac`` only covers [0, 1])."""
    if den <= 0:
        return 0.0
    v = num / den
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


def _bins_v2(value: float, edges: tuple[int, ...]) -> tuple[float, ...]:
    """One-hot over ``len(edges) + 1`` buckets: (<=e0), (<=e1), ..., (> last)."""
    out = [0.0] * (len(edges) + 1)
    for i, e in enumerate(edges):
        if value <= e:
            out[i] = 1.0
            return tuple(out)
    out[-1] = 1.0
    return tuple(out)


def snapshot_alakazam_v2(
    obs: ParsedObservation, gs: GameState, cards: CardDatabase | None
) -> tuple[float, ...]:
    base = snapshot_alakazam(obs, gs, cards)          # the 29 V1 features, unchanged
    me, opp = gs.me, gs.opponent
    my_active, opp_active = gs.my_active, gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_energy = len(opp_active.energies) if opp_active is not None else 0
    my_energy = len(my_active.energies) if my_active is not None else 0
    hand = _hand_counts_v2(me)

    # G1 — what is ACTUALLY in hand, per card (the "two 7-card hands" fix)
    g1 = tuple(_frac(hand.get(cid, 0), 4) for cid in _ALAKAZAM_DECK_IDS)

    # G2 — dual encoding: keep the continuous scalars in `base`, add one-hot bins so a
    # shallow model can express NON-MONOTONIC responses (Lux-1st: absolute-only features
    # "lacked generalization").
    g2 = (
        _bins_v2(gs.hand_size, _HAND_BINS)
        + _bins_v2(opp_hp, _OPPHP_BINS)
        + _bins_v2(obs.current.turn if obs.current is not None else 0, _TURN_BINS)
    )

    # G3 — relative/differential. The third one is the LETHAL MARGIN (how much Powerful
    # Hand over/under-kills), strictly richer than V1's binary "is it lethal" flag.
    g3 = (
        _signed(gs.hand_size - (opp.handCount if opp is not None else 0), 10),
        _signed(gs.my_bench_count - (len(opp.bench) if opp is not None else 0), 5),
        _signed(20 * gs.hand_size - opp_hp, 200),
        _signed(gs.my_deck_count - (opp.deckCount if opp is not None else 0), 40),
        _signed(my_energy - opp_energy, 3),
    )

    # G4 — composites a shallow net would struggle to assemble from parts.
    in_play = _in_play_v2(me)
    have_zam = hand.get(_ALAKAZAM_ID, 0) >= 1
    abra_ready = any(p.id == _ABRA_ID and not p.appearThisTurn for p in in_play)
    kadabra_ready = any(p.id == _KADABRA_ID and not p.appearThisTurn for p in in_play)
    can_evolve = bool(
        (abra_ready and have_zam and hand.get(_RARE_CANDY_A, 0) >= 1)  # Rare Candy skip
        or (kadabra_ready and have_zam)                                 # normal evolution
    )
    # `active_dies_if_pass` is computed in features.decision_state but only reaches ATTACK
    # options (block E); promoting it here exposes it to EVERY option type via a(x)s.
    dies = False
    if my_active is not None and opp_active is not None:
        dies = gs.max_threat(opp_active, my_active, extra_energy=1) >= my_active.hp
    g4 = (
        1.0 if can_evolve else 0.0,
        1.0 if dies else 0.0,
        _frac(opp_energy, 4),
        _frac(sum(1 for p in in_play if p.energies), 3),
    )

    return base + g1 + g2 + g3 + g4


#: ALAKAZAM_V2 — same deck/attacks/damage/wants as ALAKAZAM, richer snapshot (29 -> 76).
ALAKAZAM_V2 = DeckProfile(
    name="ALAKAZAM_V2",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=76,
    reduced_len=9,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam_v2,
    reduced_fn=reduced_alakazam,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


# =============================================================================
# ALAKAZAM_MATCHUP — M32 residual fix: a Grimmsnarl/Munkidori-conditioned feature.
#
# M32's teacher-comparison found the clone's win rate against Marnie's Grimmsnarl
# specifically (37%, n=27) is ~12 points below what the deck's own difficulty
# (teacher 52.9% here vs 57.6% overall) plus the clone's general ~4-point deficit
# would predict. A bootstrapped/controlled check (scratchpad/bootstrap_ph_turn.py)
# isolated a concrete, matchup-SPECIFIC behaviour: the clone fires Powerful Hand a
# median of 2 turns later against this deck (turn 6 vs the teacher's 4), while
# timing matches the teacher exactly in a matchup the clone wins (Mega Lucario,
# turn 4 vs 4.5) — so this is not a generic "clone is slower" trait (that lever was
# already closed generically in M31), it is triggered by facing this archetype.
#
# Mechanism (M32 section 1): Munkidori (112) removes the damage counters Powerful
# Hand places; Marnie's Impidimp/Morgrem/Grimmsnarl ex (646/647/648) is the line
# that runs it; Unfair Stamp (1080) / Team Rocket's Petrel (1219) / Spikemuth Gym
# (1259, a stadium) attack hand size, our damage dial. Unlike ALAKAZAM_V2 (which
# enriched self-state features broadly and lost the head_to_head veto), this adds
# ONLY an opponent-archetype signal so a shallow model can learn a DIFFERENT
# Powerful-Hand trigger threshold specifically when this pressure is detected —
# Lux AI S3 8th's "policy conditioning" idea, applied via observable board/discard
# markers instead of an explicit policy ID (which we don't have for opponents).
#
# Both markers are PUBLIC information (own board + own discard + shared stadium),
# so this needs no hidden-information assumption.
# =============================================================================

_MUNKIDORI_ID = 112
_MARNIE_IMPIDIMP_ID = 646
_MARNIE_MORGREM_ID = 647
_MARNIE_GRIMMSNARL_ID = 648
_UNFAIR_STAMP_ID = 1080
_TR_PETREL_ID = 1219
_SPIKEMUTH_GYM_ID = 1259

_DISRUPTION_LINE_IDS = frozenset(
    {_MUNKIDORI_ID, _MARNIE_IMPIDIMP_ID, _MARNIE_MORGREM_ID, _MARNIE_GRIMMSNARL_ID}
)
_HAND_ATTACK_IDS = frozenset({_UNFAIR_STAMP_ID, _TR_PETREL_ID})


def snapshot_alakazam_matchup(
    obs: ParsedObservation, gs: GameState, cards: CardDatabase | None
) -> tuple[float, ...]:
    base = snapshot_alakazam(obs, gs, cards)  # the 29 V1 features, unchanged
    opp = gs.opponent
    seen: set[int] = set()
    if opp is not None:
        seen.update(p.id for p in opp.active if p is not None)
        seen.update(p.id for p in opp.bench)
        seen.update(c.id for c in opp.discard)
    stadium_ids = {c.id for c in (obs.current.stadium if obs.current is not None else ())}

    facing_disruption = 1.0 if seen & _DISRUPTION_LINE_IDS else 0.0
    hand_attack_seen = 1.0 if (seen & _HAND_ATTACK_IDS) or (_SPIKEMUTH_GYM_ID in stadium_ids) else 0.0

    return base + (facing_disruption, hand_attack_seen)


#: ALAKAZAM_MATCHUP — same deck/attacks/damage/wants as ALAKAZAM, +2 opponent-
#: archetype-conditioning features (29 -> 31 raw snapshot dims).
ALAKAZAM_MATCHUP = DeckProfile(
    name="ALAKAZAM_MATCHUP",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=31,
    reduced_len=9,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam_matchup,
    reduced_fn=reduced_alakazam,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


# =============================================================================
# ALAKAZAM_SPECIALIST — M32 plan §Strategy B+C: features for the Grimmsnarl-
# routed specialist ensemble ONLY (see policy.py's "mlp_switch" scorer). Unlike
# ALAKAZAM_MATCHUP (Attempt #1, closed negative — two STICKY binary flags fed
# into the SAME shared model, diluted among 684 dims, no per-turn timing signal),
# this profile is used EXCLUSIVELY by a specialist ensemble trained only on
# Grimmsnarl games and reached only when the opponent's Munkidori/Grimmsnarl
# line is on board right now — so there is no risk of contaminating the
# general model's play against the other 69% of the field, and every feature
# below is TRANSIENT (can turn back off) rather than a permanent "ever seen".
#
# Grounded in Munkidori's actual card text (Adrena-Brain, EN_Card_Data.csv):
# "Once during your turn, if this Pokémon has any {D} Energy attached, you may
# move up to 3 damage counters from 1 of your Pokémon to 1 of your opponent's
# Pokémon." — it is only live while energized, and it both (a) heals up to 30
# HP off whichever of the opponent's Pokémon we hit with Powerful Hand and (b)
# deals that same 30 HP to one of ours. The urgency is proportional to how
# close our current Powerful Hand overkill margin is to that 30 HP window.
# =============================================================================


def snapshot_alakazam_specialist(
    obs: ParsedObservation, gs: GameState, cards: CardDatabase | None
) -> tuple[float, ...]:
    base = snapshot_alakazam(obs, gs, cards)  # the 29 V1 features, unchanged
    opp = gs.opponent
    opp_active = gs.opp_active
    turn = obs.current.turn if obs.current is not None else 0

    munkidori_energized = False
    disruption_on_board = False
    if opp is not None:
        for p in list(opp.active) + list(opp.bench):
            if p is None:
                continue
            if p.id in _DISRUPTION_LINE_IDS:
                disruption_on_board = True
            if p.id == _MUNKIDORI_ID and any(e is EnergyKind.DARKNESS for e in p.energies):
                munkidori_energized = True

    # how much of our current lethal margin a single Munkidori activation (up to
    # 30 HP moved) could erase; 1.0 = borderline/no lethal yet, 0.0 = safely
    # overkilling by 30+. Zero when Munkidori isn't live -- no threat, no urgency.
    overkill_margin_at_risk = 0.0
    if munkidori_energized:
        opp_hp = opp_active.hp if opp_active is not None else 0
        margin = 20 * gs.hand_size - opp_hp
        clamped = max(0.0, min(30.0, margin))
        overkill_margin_at_risk = (30.0 - clamped) / 30.0

    # a bounded ramp instead of a sticky flag -- ZERO once the line leaves the
    # board, and gives real per-turn gradient while it's up (Attempt #1's bug).
    disruption_pressure_turns = min(1.0, max(0.0, (turn - 1) / 10.0)) if disruption_on_board else 0.0

    return base + (
        1.0 if munkidori_energized else 0.0,
        overkill_margin_at_risk,
        disruption_pressure_turns,
    )


#: ALAKAZAM_SPECIALIST — same deck/attacks/damage/wants as ALAKAZAM, +3 transient
#: Munkidori-pressure features (29 -> 32 raw snapshot dims). Used ONLY by the
#: Grimmsnarl-routed specialist ensemble (policy.py "mlp_switch"), never the
#: general model.
ALAKAZAM_SPECIALIST = DeckProfile(
    name="ALAKAZAM_SPECIALIST",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=32,
    reduced_len=9,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam_specialist,
    reduced_fn=reduced_alakazam,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


# =============================================================================
# ALAKAZAM_FETCH — M34 Track B. The first profile change aimed at the REDUCED
# block rather than the snapshot, and the reason is mechanical.
#
# In a TO_HAND decision (tutor resolution: Dawn/Hilda/Poké Pad/Poffin) every
# option is OptionKind.CARD from the same area, so blocks A, C, E, F, the raw
# snapshot S and the A⊗S interaction are IDENTICAL across the options of that
# decision and cancel in the softmax. The only state-conditioned discrimination
# left is B⊗R — the card-id one-hot crossed with `reduced`. So for fetch
# decisions, `reduced` is the entire state channel, and enriching `snapshot`
# (what ALAKAZAM_V2 did, reduced_len untouched at 9) provably cannot help.
#
# Measured deficit (scratchpad/analyze_tohand_divergence.py, 10,393 teacher
# decisions re-scored with the shipped clone): on turns 1-5 the teacher fetches
# a combo piece 59.9% of the time and the clone 41.9%; the single largest
# disagreement is "teacher fetched Kadabra, clone would fetch Dudunsparce"
# (914×). On the 3,506-decision subset where a combo piece and a non-combo
# option were both available, teacher 49.4% vs clone 21.0% (McNemar p≈6e-141).
#
# The mechanism probe named the missing variable: the teacher's fetch rate
# swings 59.1% → 30.6% → 12.1% as Kadabra and then Abra are ALREADY IN HAND
# (don't fetch a piece you're holding), while neither `snapshot_alakazam` nor
# `reduced_alakazam` carries an Abra/Kadabra hand count — reduced holds only
# hand_size, opp hand, turn, prize_diff, active energy, has-{P}, Alakazam in
# play, Rare Candy in hand, opp HP. The model is blind to the governing
# variable, so it fits a proxy that inverts in the decisive cell. This is an
# observability gap, not a capacity gap: more MLP width over features that omit
# the variable cannot learn the rule.
#
# Additive and minimal: same deck/attacks/snapshot/damage/wants as ALAKAZAM,
# only `reduced` grows 9 -> 13 (dim 658 -> 750). A separate profile so the
# shipped bc_alakazam_mlp.json (pinned to 658) keeps loading unchanged.
# =============================================================================


def reduced_alakazam_fetch(
    obs: ParsedObservation, gs: GameState, cards: CardDatabase | None
) -> tuple[float, ...]:
    base = reduced_alakazam(obs, gs, cards)  # the 9 V1 scalars, unchanged
    me = gs.me
    return base + (
        # "do I already hold this piece?" — the variable the teacher conditions on
        _frac(_count_hand_a(me, _ABRA_ID), 2),
        _frac(_count_hand_a(me, _KADABRA_ID), 2),
        _frac(_count_hand_a(me, _ALAKAZAM_ID), 2),
        # a Kadabra is only worth fetching if there is an Abra on board to evolve
        _frac(_count_in_play(me, _ABRA_ID), 3),
    )


# =============================================================================
# ALAKAZAM_MAIN2 — M46: the same fix M34 applied to TO_HAND, applied to MAIN.
#
# WHY. In a softmax over a decision's options, anything CONSTANT across options
# cancels exactly (features.py:14-17, 208). For two MAIN options of the SAME
# OptionKind — "play Poffin" vs "play Dawn", "attach to active" vs "attach to
# bench" — block A (option type) is identical too, so A(x)S cancels as well and
# ALL state-conditioned discrimination collapses onto B(x)R: the acting card id
# crossed with `reduced`. That is 9 scalars, and it is the entire channel for
# "play card X BECAUSE of situation R".
#
# M33 measured that PLAY->PLAY is the #1 disagreement with the teacher (29.2% of
# all disagreements) — exactly the decisions squeezed through those 9 scalars.
# M34 widened this same channel for TO_HAND (9 -> 13) and TO_HAND fidelity moved
# +0.114. Nobody has ever widened it for MAIN.
#
# NOT the same as M31's ALAKAZAM_V2, which enriched `snapshot` (29 -> 76). That
# was offline-positive but LOST the head-to-head (-0.033, 90%CI [-0.056,-0.011]).
# Snapshot only reaches options through A(x)S, i.e. at option-TYPE granularity;
# it can never separate two options of the same type. This profile leaves
# snapshot untouched and widens only the block that can.
#
# The six additions, each a documented hole:
#   R9  active_dies_if_pass — computed once per decision (features.py:108-110) but
#       emitted ONLY at e[5], which features.py:184 gates on OptionKind.ATTACK. So
#       "my active dies next turn" cannot influence RETREAT, ATTACH, EVOLVE, PLAY
#       or END today. deck_profiles.py:1699-1701 names this bug; only V2 fixed it,
#       buried among 46 other scalars and never isolated.
#   R10/R11 Abra/Kadabra in hand — the M34 variables, absent from MAIN entirely.
#   R12/R13 my/opponent prizes left, ABSOLUTE. `reduced` carries only the signed
#       difference (R3), so "the opponent is on 1 prize, close NOW" is inexpressible.
#   R14 turnActionCount — MAIN has no within-turn clock at all. Parsed since forever
#       (observation/models.py:262), read by no profile in the ALAKAZAM line.
#
# Cost: +6 reduced scalars = +6*23 = +138 columns -> dim 796. Additive: every
# existing profile and payload is untouched.
# =============================================================================


def reduced_alakazam_main2(
    obs: ParsedObservation, gs: GameState, cards: CardDatabase | None
) -> tuple[float, ...]:
    base = reduced_alakazam(obs, gs, cards)  # the 9 V1 scalars, unchanged
    me = gs.me
    state = obs.current

    # Mirrors features.decision_state:107-110 exactly, so the scalar means the
    # same thing here as in block E — but reaches EVERY option type, not just ATTACK.
    dies = False
    if gs.my_active is not None and gs.opp_active is not None:
        dies = gs.max_threat(gs.opp_active, gs.my_active, extra_energy=1) >= gs.my_active.hp

    return base + (
        1.0 if dies else 0.0,
        _frac(_count_hand_a(me, _ABRA_ID), 2),
        _frac(_count_hand_a(me, _KADABRA_ID), 2),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        _frac(getattr(state, "turnActionCount", 0) if state else 0, 28),
    )


#: ALAKAZAM_MAIN2 — ALAKAZAM + 6 scalars in `reduced` (9 -> 15), the only block
#: that separates two MAIN options of the same OptionKind. dim 796.
ALAKAZAM_MAIN2 = DeckProfile(
    name="ALAKAZAM_MAIN2",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=29,
    reduced_len=15,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam,
    reduced_fn=reduced_alakazam_main2,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


#: ALAKAZAM_FETCH — ALAKAZAM + 4 evolution-line hand/board scalars in `reduced`
#: (9 -> 13), the only block that conditions a fetch on the situation. dim 750.
ALAKAZAM_FETCH = DeckProfile(
    name="ALAKAZAM_FETCH",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=29,
    reduced_len=13,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam,
    reduced_fn=reduced_alakazam_fetch,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


# =============================================================================
# ALAKAZAM_MEM — M37 arm E: the two free memory channels the ALAKAZAM line never used.
#
# The profile is STRICTLY MARKOVIAN today: of its 29 snapshot slots, only three
# ("energy_attached", "supporter_played", "retreated") say anything about what already
# happened, and they reset every turn. Two channels were sitting unused:
#
#   * THE DISCARD PILE. Every other profile in this file reads it (_discard_resources_940,
#     _basic_energy_in_discard, ...); ALAKAZAM is the only one that does not. Measured on
#     6,000 real MAIN decisions: mean 13.8 cards, max 52, empty in only 3.8%. And the most
#     discarded cards are exactly the engine — Poke Pad, Poffin, Dawn, Hilda, Rare Candy,
#     Telepath Energy. "How many Rare Candy are already burned" is real, free, cross-turn
#     memory about how much of the combo is still reachable.
#   * turnActionCount. Parsed since forever (observation/models.py:262), present in every
#     stored row, read by NO profile. Measured range 1..28 — a within-turn sequence index,
#     i.e. "how deep into this turn am I", which is exactly the axis a per-decision policy
#     is blind to.
#
# Placement follows M34: the discriminating scalars go in `reduced` (crossed with the
# card-id one-hot, the only block that can express "play/fetch card X in situation R");
# global counters go in `snapshot`. Costs 23 dims per reduced slot vs 13 per snapshot slot,
# so `reduced` gets only the three that must condition on card identity.
# dim: 74 + 36 + 12*36 + 23*12 = 818.
# =============================================================================

_ALAKAZAM_ENERGY_IDS = (5, 13, 19)          # Basic {P}, Enriching, Telepath Psychic
_ALAKAZAM_LINE_IDS = (741, 742, 743)        # Abra / Kadabra / Alakazam


def _discard_count_a(player: ParsedPlayer | None, ids) -> int:
    if player is None or player.discard is None:
        return 0
    idset = ids if isinstance(ids, (set, frozenset)) else set(ids)
    return sum(1 for c in player.discard if c.id in idset)


def snapshot_alakazam_mem(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    base = snapshot_alakazam(obs, gs, cards)          # first 29 slots unchanged
    me, state = gs.me, obs.current
    discard = me.discard if (me is not None and me.discard is not None) else ()
    return base + (
        _frac(getattr(state, "turnActionCount", 0) if state else 0, 20),
        _frac(len(discard), 60),
        _frac(_discard_count_a(me, (_RARE_CANDY_A,)), 3),
        _frac(_discard_count_a(me, (_DAWN_ID,)), 4),
        _frac(_discard_count_a(me, _ALAKAZAM_ENERGY_IDS), 7),
        _frac(_discard_count_a(me, _ALAKAZAM_LINE_IDS), 6),
        _frac(_discard_count_a(me, (_BOSS_ID_A,)), 3),
    )


def reduced_alakazam_mem(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    """The 9 ALAKAZAM scalars + 3 that must cross with WHICH card is being chosen.

    Energy is the scarcest resource in the deck (7 of 60), so "how much is already in
    the discard" changes which card is worth playing — not just how good the turn is.
    Same for Rare Candy (3 copies) and for how deep into the turn we are.
    """
    base = reduced_alakazam(obs, gs, cards)
    me, state = gs.me, obs.current
    return base + (
        _frac(getattr(state, "turnActionCount", 0) if state else 0, 20),
        _frac(7 - _discard_count_a(me, _ALAKAZAM_ENERGY_IDS), 7),   # energy still reachable
        _frac(3 - _discard_count_a(me, (_RARE_CANDY_A,)), 3),       # Rare Candy still reachable
    )


#: ALAKAZAM_MEM — M37 arm E. Additive: same deck/attacks/damage/wants as ALAKAZAM,
#: only the state blocks grow (snapshot 29->36, reduced 9->12). dim 818.
ALAKAZAM_MEM = DeckProfile(
    name="ALAKAZAM_MEM",
    deck_ids=_ALAKAZAM_DECK_IDS,
    attack_ids=(76, 183, 423, 424, 477, 1070, 1071, 1072),
    target_pokemon_ids=(66, 140, 305, 343, 741, 742, 743),
    snapshot_len=36,
    reduced_len=12,
    opp_target_dim=4,
    flagged_attack_id=_POWERFUL_HAND,
    snapshot_fn=snapshot_alakazam_mem,
    reduced_fn=reduced_alakazam_mem,
    damage_fn=damage_alakazam,
    wants_fn=wants_alakazam,
)


# =============================================================================
# GRIMMSNARL — M35 clone of "Luca" (Kaggle top-5, ~1194), Marnie's Grimmsnarl ex.
# Chosen over a higher-elo teacher because this deck is STRUCTURALLY clonable in a
# way Yushin's Alakazam is not: Powerful Hand deals 20 x hand_size, so every card
# Yushin plays costs him 20 damage and couples every decision in the turn through
# one global variable a per-decision reactive policy cannot represent. Shadow
# Bullet is a flat 180 — decisions are separable, which is what this architecture
# does well. Measured: generic-profile MAIN 0.554 here vs 0.470 for Yushin at the
# same data volume (148 vs 150 games).
#
#   * Marnie's Grimmsnarl ex (648): 320 HP Stage-2, Shadow Bullet {D}{D} 180 + a
#     separate 30 to a Benched Pokemon (that is its own DAMAGE sub-select, NOT part
#     of damage_fn's Active-vs-Active scope — same treatment as Jetting Blow in
#     damage_itf). Ability Punk Up: on evolving from hand, search up to 5 basic {D}
#     and attach them freely. ex -> 2 prizes.
#   * Munkidori (112): Ability Adrena-Brain moves up to 3 damage counters from one
#     of my Pokemon to one of the opponent's, gated on having ANY {D} attached — so
#     it wants exactly ONE energy, never more.
#   * Froslass (104): Freezing Shroud puts a counter on every Pokemon with an
#     Ability during Checkup. Played for the Ability; wants ZERO energy.
# Three attacks are unpayable in this deck (Frost Smash {W}, Mind Bend {P}, Chilly
# {W} — the list runs only basic {D}), so damage_fn returns 0 for them rather than
# letting features.py claim a phantom lethal.
#
# opp_target_dim=4 is LOAD-BEARING here, not just for Boss's Orders: DAMAGE_COUNTER
# and DAMAGE (1,429 rows, 9.6% of decisions) target the OPPONENT's board, and the
# opp-target block (hp / prize value / lethal flag) is measurably their only
# discriminating signal — opp_hp varies in 889 of 894 such decisions.
# =============================================================================

_MARNIE_IMPIDIMP = 646
_MARNIE_MORGREM = 647
_MARNIE_GRIMMSNARL = 648
_MUNKIDORI = 112
_FROSLASS = 104
_SNORUNT = 860
_D_BASIC_ID = 7              # the only energy in the deck: 10x basic {D}
_SHADOW_BULLET = 937         # 180 + 30 to a bench target (the bench hit is a sub-select)
_FROST_SMASH = 131           # {W}{C} — unpayable here
_MIND_BEND = 141             # {P}{C} — unpayable here
_CHILLY = 1239               # {W}    — unpayable here
_RARE_CANDY_G = 1079
_BOSS_ID_G = 1182
_SPIKEMUTH_GYM = 1259
_UNFAIR_STAMP = 1080
_G_MARNIE_LINE = (_MARNIE_IMPIDIMP, _MARNIE_MORGREM, _MARNIE_GRIMMSNARL)
_G_UNPAYABLE = (_FROST_SMASH, _MIND_BEND, _CHILLY)


def _d_energy_on(pkmn: ParsedPokemon | None) -> int:
    """Basic {D} attached — pays Shadow Bullet and switches Adrena-Brain on."""
    if pkmn is None:
        return 0
    return sum(1 for e in pkmn.energies if e is EnergyKind.DARKNESS)


def _d_energy_on_board(player: ParsedPlayer | None) -> int:
    if player is None:
        return 0
    total = 0
    for p in player.active:
        if p is not None:
            total += _d_energy_on(p)
    for p in player.bench:
        total += _d_energy_on(p)
    return total


def _my_damage_counters(player: ParsedPlayer | None) -> int:
    """Damage sitting on my own board — the fuel Adrena-Brain relocates."""
    if player is None:
        return 0
    total = 0
    for p in player.active:
        if p is not None:
            total += max(0, p.maxHp - p.hp)
    for p in player.bench:
        total += max(0, p.maxHp - p.hp)
    return total


def damage_grimmsnarl(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    """Structured damage + W/R. Shadow Bullet is honest 180 to the Active.

    The +30 to a Benched Pokemon is resolved by a separate DAMAGE select, so it is
    outside this function's Active-vs-Active scope (cf. damage_itf's Jetting Blow).
    The three off-colour attacks cost {W}/{P} and this list runs only basic {D}, so
    they can never be paid — return 0 so the lethal/overkill slots in features.py
    never claim a KO the deck cannot deliver.
    """
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None:
        return 0
    if attack_id in _G_UNPAYABLE:
        return 0
    atk = cards.get_attack(attack_id) if cards is not None else None
    base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None or cards is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def snapshot_grimmsnarl(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    me, opp, state = gs.me, gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    active_is_grimm = bool(my_active is not None and my_active.id == _MARNIE_GRIMMSNARL)
    active_is_morgrem = bool(my_active is not None and my_active.id == _MARNIE_MORGREM)
    active_energy = len(my_active.energies) if my_active is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    stadium_ids = {c.id for c in state.stadium} if state is not None else set()
    munkidori_online = False
    for p in list(me.active if me else ()) + list(me.bench if me else ()):
        if p is not None and p.id == _MUNKIDORI and _d_energy_on(p) >= 1:
            munkidori_online = True
            break
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(gs.hand_size, 10),
        _frac(opp.handCount if opp else 0, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 340),
        # who holds the Active spot and can it actually fire Shadow Bullet ({D}{D})
        1.0 if active_is_grimm else 0.0,
        1.0 if active_is_morgrem else 0.0,
        _frac(active_energy, 3),
        1.0 if _d_energy_on(my_active) >= 2 else 0.0,
        # the evolution line on board
        _frac(_count_in_play(me, _MARNIE_GRIMMSNARL), 3),
        _frac(_count_in_play(me, _MARNIE_MORGREM), 3),
        _frac(_count_in_play(me, _MARNIE_IMPIDIMP), 4),
        # the two Ability engines
        1.0 if _count_in_play(me, _MUNKIDORI) >= 1 else 0.0,
        1.0 if munkidori_online else 0.0,          # Adrena-Brain actually usable
        1.0 if _count_in_play(me, _FROSLASS) >= 1 else 0.0,
        # resources in hand
        _frac(_hand_count(me, (_RARE_CANDY_G,)), 3),
        _frac(_hand_count(me, (_BOSS_ID_G,)), 2),
        1.0 if _SPIKEMUTH_GYM in stadium_ids else 0.0,
        # fuel for Adrena-Brain: damage already sitting on my own board
        _frac(_my_damage_counters(me), 200),
    )


def reduced_grimmsnarl(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    """The 13 scalars crossed with the card-id one-hot (B(x)R).

    Designed FIRST, per M34: `reduced` is the ONLY state channel that can
    discriminate between the options of one decision (snapshot is constant across
    options and cancels in the softmax). ALAKAZAM_V2 enriched snapshot only and
    lost; ALAKAZAM_FETCH enriched reduced and moved TO_HAND fidelity +0.114. So the
    evolution-line hand/board composition goes HERE, where "fetch/play card X given
    situation R" can actually be expressed.
    """
    me, opp, state = gs.me, gs.opponent, obs.current
    my_active = gs.my_active
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(gs.hand_size, 10),
        _frac(opp.handCount if opp else 0, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
        _frac(_d_energy_on(my_active), 3),
        _frac(_d_energy_on_board(me), 6),
        _frac(_count_in_play(me, _MARNIE_GRIMMSNARL), 3),
        # what the line already holds — a Morgrem/Rare Candy is only worth having
        # if there is an Impidimp to evolve, and a 2nd copy is near-worthless
        _frac(_hand_count(me, (_MARNIE_IMPIDIMP,)), 3),
        _frac(_hand_count(me, (_MARNIE_MORGREM,)), 2),
        _frac(_hand_count(me, (_MARNIE_GRIMMSNARL,)), 2),
        _frac(_hand_count(me, (_RARE_CANDY_G,)), 3),
        _frac(_count_in_play(me, _MARNIE_IMPIDIMP), 4),
        _frac(_hand_count(me, (_D_BASIC_ID,)), 5),
    )


def wants_grimmsnarl(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    """Energy demand per Pokemon, from the real costs and Abilities.

    The generic fallback is ACTIVELY WRONG for half this deck: gs.best_attack picks
    the highest-damage attack and gs.energy_pays can never satisfy an off-colour
    cost with {D}, so Munkidori/Froslass/Snorunt all read a permanent 1.0. Measured
    at 0-5 attached: all three stuck ON. That poisons d[wants_i] in ATTACH_FROM
    (890 rows), the context where the slot matters most. Hence every Pokemon is
    enumerated explicitly here.
    """
    n = _d_energy_on(pkmn)
    if pkmn.id in (_MARNIE_GRIMMSNARL, _MARNIE_MORGREM):
        return 1.0 if n < 2 else 0.0        # Shadow Bullet / Corkscrew Punch = {D}{D}
    if pkmn.id == _MARNIE_IMPIDIMP:
        return 1.0 if n < 1 else 0.0
    if pkmn.id == _MUNKIDORI:
        return 1.0 if n < 1 else 0.0        # Adrena-Brain needs ANY {D}, never a 2nd
    if pkmn.id in (_FROSLASS, _SNORUNT):
        return 0.0                          # played for Freezing Shroud, never attacks
    return 1.0 if gs.wants_energy(pkmn) else 0.0


#: GRIMMSNARL — M35 hand-authored clone of Luca (top-5). 19 unique ids -> b_len 20,
#: 7 attacks -> c_len 8, 6 Pokemon + opp_target_dim 4 -> d_len 18. dim 706.
GRIMMSNARL = DeckProfile(
    name="GRIMMSNARL",
    deck_ids=(7, 104, 112, 646, 647, 648, 860, 1079, 1080, 1086,
              1097, 1122, 1137, 1152, 1182, 1219, 1227, 1231, 1259),
    attack_ids=(131, 141, 934, 935, 936, 937, 1239),
    target_pokemon_ids=(104, 112, 646, 647, 648, 860),
    snapshot_len=29,
    reduced_len=13,
    opp_target_dim=4,
    flagged_attack_id=_SHADOW_BULLET,
    snapshot_fn=snapshot_grimmsnarl,
    reduced_fn=reduced_grimmsnarl,
    damage_fn=damage_grimmsnarl,
    wants_fn=wants_grimmsnarl,
)


PROFILES: dict[str, DeckProfile] = {
    TR_650.name: TR_650,
    LUCARIO_800.name: LUCARIO_800,
    LUCARIO_800_V2.name: LUCARIO_800_V2,
    BELLIBOLT_940.name: BELLIBOLT_940,
    CINDERACE_METAL.name: CINDERACE_METAL,
    KANGASKHAN_1052.name: KANGASKHAN_1052,
    ITF_ESYS_KASU.name: ITF_ESYS_KASU,
    THIRD_PTCG.name: THIRD_PTCG,
    ALAKAZAM.name: ALAKAZAM,
    ALAKAZAM_V2.name: ALAKAZAM_V2,
    ALAKAZAM_MATCHUP.name: ALAKAZAM_MATCHUP,
    ALAKAZAM_SPECIALIST.name: ALAKAZAM_SPECIALIST,
    ALAKAZAM_FETCH.name: ALAKAZAM_FETCH,
    ALAKAZAM_MAIN2.name: ALAKAZAM_MAIN2,
    ALAKAZAM_MEM.name: ALAKAZAM_MEM,
    GRIMMSNARL.name: GRIMMSNARL,
}


# =============================================================================
# Generic auto-built profile — DEV-ONLY cheap screening, never registered/shipped
# by default. No hand-tuning: attacks use plain structured damage + weakness/
# resistance (no "prose" corrections), a Pokémon "wants" energy via the generic
# GameState.wants_energy fallback, and the snapshot is just the common core
# every hand-authored profile above already shares (turn/prizes/hands/bench/
# flags/opp-hp). Good enough for an offline MAIN-accuracy read to decide
# whether a candidate is worth a real hand-authored profile (see
# scratchpad/quick_screen.py) — NOT good enough to ship (prose-damage attacks,
# per-card energy priorities, etc. all silently fall back to generic behavior).
# =============================================================================


def generic_damage(attack_id: int | None, gs: GameState, cards: CardDatabase | None) -> int:
    attacker = gs.my_active
    defender = gs.opp_active
    if attacker is None or attack_id is None or cards is None:
        return 0
    atk = cards.get_attack(attack_id)
    base = atk.damage if atk is not None else 0
    if base <= 0 or defender is None:
        return max(0, base)
    atk_info = cards.get_card(attacker.id)
    def_info = cards.get_card(defender.id)
    if atk_info is not None and def_info is not None:
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            base *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            base = max(0, base - 30)
    return base


def generic_snapshot(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    opp, state = gs.opponent, obs.current
    opp_active = gs.opp_active
    opp_hp = opp_active.hp if opp_active is not None else 0
    opp_ex = False
    if opp_active is not None and cards is not None:
        info = cards.get_card(opp_active.id)
        opp_ex = bool(info and (info.ex or info.megaEx))
    my_active = gs.my_active
    active_damaged = bool(my_active is not None and my_active.hp < my_active.maxHp)
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    active_energy = len(my_active.energies) if my_active is not None else 0
    return (
        _frac(state.turn if state else 0, 20),
        _frac(gs.my_prizes_left, 6),
        _frac(gs.opp_prizes_left, 6),
        max(-1.0, min(1.0, prize_diff)),
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(gs.my_bench_count, 5),
        _frac(len(opp.bench) if opp else 0, 5),
        _frac(gs.my_deck_count, 60),
        1.0 if gs.energy_attached else 0.0,
        1.0 if gs.supporter_played else 0.0,
        1.0 if gs.retreated else 0.0,
        1.0 if opp_ex else 0.0,
        1.0 if active_damaged else 0.0,
        _frac(opp_hp, 300),
        _frac(active_energy, 4),
    )


def generic_reduced(obs: ParsedObservation, gs: GameState, cards: CardDatabase | None) -> tuple[float, ...]:
    opp, state = gs.opponent, obs.current
    my_hand = gs.hand_size
    opp_hand = opp.handCount if opp is not None else 0
    prize_diff = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    return (
        _frac(my_hand, 10),
        _frac(opp_hand, 10),
        _frac(state.turn if state else 0, 20),
        max(-1.0, min(1.0, prize_diff)),
    )


def generic_wants(pkmn: ParsedPokemon, gs: GameState, cards: CardDatabase | None) -> float:
    return 1.0 if gs.wants_energy(pkmn) else 0.0


def build_generic_profile(name: str, deck_ids: tuple[int, ...], cards: CardDatabase) -> DeckProfile:
    """Auto-build a screening-only DeckProfile from just a decklist + CardDatabase.

    No per-deck code to write — derives ``attack_ids``/``target_pokemon_ids``
    straight from the cards. Intended for :mod:`scratchpad.quick_screen`, never
    for a shipped submission (register manually + author real ``*_fn``s first).
    """
    attack_ids: list[int] = []
    target_ids: list[int] = []
    for cid in deck_ids:
        info = cards.get_card(cid)
        if info is None or info.cardType.name != "POKEMON":
            continue
        target_ids.append(cid)
        for aid in info.attacks:
            if aid not in attack_ids:
                attack_ids.append(aid)
    return DeckProfile(
        name=name,
        deck_ids=tuple(deck_ids),
        attack_ids=tuple(attack_ids),
        target_pokemon_ids=tuple(target_ids),
        snapshot_len=16,
        reduced_len=4,
        opp_target_dim=4,  # generic: keep the opponent-target block live (Boss's Orders etc.)
        flagged_attack_id=None,
        snapshot_fn=generic_snapshot,
        reduced_fn=generic_reduced,
        damage_fn=generic_damage,
        wants_fn=generic_wants,
    )


def get_profile(name: str) -> DeckProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown deck profile {name!r}; known: {sorted(PROFILES)}") from None
