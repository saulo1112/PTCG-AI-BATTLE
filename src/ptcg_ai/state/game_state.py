"""Perspective-normalized snapshot + derived tactical quantities (ADR-0013).

A :class:`GameState` is built once per decision from a
:class:`~ptcg_ai.observation.models.ParsedObservation` plus the static
:class:`~ptcg_ai.cards.database.CardDatabase`. It exposes the "me vs
opponent" view and the derived quantities the ladder's Scorers consult —
prize race, KO math (weakness ×2 / resistance −30), tempo flags — so no rule
re-derives them. Fields grow additively as rules need them (design:
docs/phase2_blueprint.md §4.1).

The class is SDK-free: it consumes only our own parsed models and card types,
so it is unit-testable by constructing a small :class:`CardDatabase` from
literal card metadata (no native engine required).
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.observation.models import (
    EnergyKind,
    ParsedObservation,
    ParsedPlayer,
    ParsedPokemon,
    ParsedState,
)

#: Rulebook resistance reduction (feature_inventory.md §9).
RESISTANCE_REDUCTION = 30

#: Energy types that pay for any cost symbol (wild). COLORLESS as a *cost*
#: symbol accepts anything; RAINBOW/TEAM_ROCKET as *attached* energy provide
#: any type (feature_inventory.md §3).
_WILD_PROVIDERS = frozenset({EnergyKind.RAINBOW, EnergyKind.TEAM_ROCKET})


@dataclass(frozen=True)
class GameState:
    """Derived, perspective-normalized view of one decision's board.

    ``None`` for ``current``/``cards`` degrades gracefully: every field has a
    safe default so a policy can still act (rung 3 without card knowledge
    still develops bench and attacks, just without KO math).
    """

    state: ParsedState | None
    cards: CardDatabase | None

    me: ParsedPlayer | None
    opponent: ParsedPlayer | None
    my_active: ParsedPokemon | None
    opp_active: ParsedPokemon | None

    my_prizes_left: int
    opp_prizes_left: int
    my_active_prize_value: int
    opp_active_prize_value: int

    my_bench_count: int
    bench_room: int
    has_bench_insurance: bool

    hand_size: int
    my_deck_count: int

    energy_attached: bool
    supporter_played: bool
    retreated: bool

    @classmethod
    def build(cls, observation: ParsedObservation, cards: CardDatabase | None) -> "GameState":
        """Build from the observation's own perspective (``yourIndex``)."""
        state = observation.current
        index = state.yourIndex if state is not None else 0
        return cls.build_for(observation, cards, index)

    @classmethod
    def build_for(
        cls, observation: ParsedObservation, cards: CardDatabase | None, player_index: int
    ) -> "GameState":
        """Build the "me vs opponent" view from ``player_index``'s seat.

        Rung-5 search evaluates leaf nodes that belong to either player, so the
        evaluator fixes the root player's index once and builds every leaf from
        that seat — never trusting the leaf's own ``yourIndex`` (which flips on
        opponent-owned nodes). ``build`` is the ``player_index == yourIndex``
        special case.
        """
        state = observation.current
        if state is None:
            return cls(
                state=None, cards=cards, me=None, opponent=None,
                my_active=None, opp_active=None,
                my_prizes_left=6, opp_prizes_left=6,
                my_active_prize_value=1, opp_active_prize_value=1,
                my_bench_count=0, bench_room=0, has_bench_insurance=False,
                hand_size=0, my_deck_count=0,
                energy_attached=False, supporter_played=False, retreated=False,
            )
        me = state.players[player_index]
        opp = state.players[1 - player_index]
        my_active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        bench_count = len(me.bench)
        return cls(
            state=state,
            cards=cards,
            me=me,
            opponent=opp,
            my_active=my_active,
            opp_active=opp_active,
            my_prizes_left=len(me.prize),
            opp_prizes_left=len(opp.prize),
            my_active_prize_value=cls._prize_value(my_active, cards),
            opp_active_prize_value=cls._prize_value(opp_active, cards),
            my_bench_count=bench_count,
            bench_room=max(0, me.benchMax - bench_count),
            has_bench_insurance=bench_count >= 1,
            hand_size=me.handCount,
            my_deck_count=me.deckCount,
            energy_attached=state.energyAttached,
            supporter_played=state.supporterPlayed,
            retreated=state.retreated,
        )

    # -- KO / combat math -------------------------------------------------

    def attack_damage(
        self, attack: AttackInfo, attacker: ParsedPokemon, defender: ParsedPokemon
    ) -> int:
        """Base attack damage adjusted for the defender's weakness/resistance.

        Weakness doubles; resistance subtracts ``RESISTANCE_REDUCTION``. Only
        the structured base damage is used — conditional "+N" effect text is
        prose (feature_inventory.md §9) and not modelled at rung 3. Returns 0
        for a 0-damage attack (status-only attacks).
        """
        damage = attack.damage
        if damage <= 0 or self.cards is None:
            return max(0, damage)
        atk_info = self.cards.get_card(attacker.id)
        def_info = self.cards.get_card(defender.id)
        if atk_info is None or def_info is None:
            return damage
        if def_info.weakness is not None and def_info.weakness == atk_info.energyType:
            damage *= 2
        if def_info.resistance is not None and def_info.resistance == atk_info.energyType:
            damage = max(0, damage - RESISTANCE_REDUCTION)
        return damage

    def is_lethal(
        self, attack: AttackInfo, attacker: ParsedPokemon, defender: ParsedPokemon
    ) -> bool:
        """True if this attack would KO the defender outright."""
        return self.attack_damage(attack, attacker, defender) >= defender.hp

    # -- energy / attacker readiness (rung 4) -----------------------------

    @staticmethod
    def energy_pays(attached: tuple[EnergyKind, ...], cost: tuple[EnergyKind, ...]) -> bool:
        """Whether ``attached`` energy satisfies an attack ``cost``.

        Specific-type symbols are matched first (a wild provider — RAINBOW /
        TEAM_ROCKET — can stand in); COLORLESS symbols are then paid by any
        leftover energy. This is the common case for our Water/Colorless decks;
        exotic special-energy text is not modelled at rung 4.
        """
        remaining = list(attached)
        colorless_needed = 0
        for symbol in cost:
            if symbol is EnergyKind.COLORLESS:
                colorless_needed += 1
                continue
            match = next(
                (e for e in remaining if e == symbol or e in _WILD_PROVIDERS), None
            )
            if match is None:
                return False
            remaining.remove(match)
        return len(remaining) >= colorless_needed

    def best_attack(self, pokemon: ParsedPokemon) -> AttackInfo | None:
        """The pokemon's highest-base-damage attack (``None`` if it has none)."""
        if self.cards is None:
            return None
        info = self.cards.get_card(pokemon.id)
        if info is None:
            return None
        best: AttackInfo | None = None
        for attack_id in info.attacks:
            attack = self.cards.get_attack(attack_id)
            if attack is None:
                continue
            if best is None or attack.damage > best.damage:
                best = attack
        return best

    def wants_energy(self, pokemon: ParsedPokemon) -> bool:
        """True if attaching more energy would bring the pokemon closer to a
        usable damaging attack — the signal for bench-first energy planning.

        A pokemon whose best damaging attack is already payable does NOT want
        more energy (charge the next attacker instead). Without card knowledge
        we conservatively say yes (the active still gets charged)."""
        if self.cards is None:
            return True
        best = self.best_attack(pokemon)
        if best is None or best.damage <= 0:
            return False
        return not self.energy_pays(pokemon.energies, best.energies)

    # -- board threat / reserves (rung 5 evaluator) -----------------------

    def reserve_attackers(self, player: ParsedPlayer | None, max_missing_energy: int = 2) -> int:
        """How many of ``player``'s benched Pokémon are (nearly) ready to attack.

        A benched Pokémon counts if it has a damaging attack that is already
        payable or within ``max_missing_energy`` attaches of being payable — the
        "insurance" against the bench-out loss mode (a charged replacement when
        the Active is KO'd). Without card knowledge, every bencher counts (we
        cannot assess readiness, so we do not punish having a bench)."""
        if player is None:
            return 0
        if self.cards is None:
            return len(player.bench)
        count = 0
        for pkmn in player.bench:
            best = self.best_attack(pkmn)
            if best is None or best.damage <= 0:
                continue
            need = len(best.energies)
            have = len(pkmn.energies)
            if have >= need or (need - have) <= max_missing_energy:
                count += 1
        return count

    def max_threat(
        self,
        attacker: ParsedPokemon | None,
        defender: ParsedPokemon | None,
        extra_energy: int = 0,
    ) -> int:
        """Max effective damage ``attacker``'s *payable* attacks deal to
        ``defender``. ``extra_energy`` optimistically pads the attacker with
        that many of its own energy type (model "after one more attach", the
        signal for 'can they KO me next turn?'). ``0`` if either Pokémon is
        missing or no card data. Falls back to base best-attack damage when the
        attacker's card is unknown."""
        if attacker is None or defender is None or self.cards is None:
            return 0
        atk_info = self.cards.get_card(attacker.id)
        if atk_info is None:
            best = self.best_attack(attacker)
            return best.damage if best else 0
        attached = tuple(attacker.energies) + (atk_info.energyType,) * max(0, extra_energy)
        best_dmg = 0
        for attack_id in atk_info.attacks:
            attack = self.cards.get_attack(attack_id)
            if attack is None or attack.damage <= 0:
                continue
            if self.energy_pays(attached, attack.energies):
                best_dmg = max(best_dmg, self.attack_damage(attack, attacker, defender))
        return best_dmg

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _prize_value(pokemon: ParsedPokemon | None, cards: CardDatabase | None) -> int:
        """Prizes the opponent takes for KO'ing this Pokémon (1 / 2 / 3)."""
        if pokemon is None or cards is None:
            return 1
        info: CardInfo | None = cards.get_card(pokemon.id)
        if info is None:
            return 1
        if info.megaEx:
            return 3
        if info.ex:
            return 2
        return 1
