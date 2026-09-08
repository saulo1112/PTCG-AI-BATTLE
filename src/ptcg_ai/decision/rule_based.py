"""Rule-based pilot — ladder rung 4 (ADR-0013, M5).

Extends the rung-3 :class:`~ptcg_ai.decision.greedy.GreedyPolicy` with the
subsystems the top-of-leaderboard replays use and greedy lacks
(docs/replay_analysis.md, M5 analysis): **bench-first energy planning** and
**retreat/promote**. It shares greedy's whole decision pipeline (routing,
Trainer whitelist, search/setup handlers) and only overrides *where the energy
goes* and *when to retreat* — the two levers the top-1/top-2 fingerprints show
account for the pilot gap.

Why bench-first matters (measured): greedy attaches 100% to the Active, so once
the Active can attack, further energy is "wasted" on it while the next attacker
sits empty — after a KO the promoted Pokémon starts from zero and the prize
race is lost. Every lean synergy deck we tried failed for exactly this reason
(deck v2 0.283, Lucario 0.530). Charging the *next* attacker on the bench once
the Active is ready is what makes those decks viable.
"""

from __future__ import annotations

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.observation.models import (
    AreaKind,
    OptionKind,
    ParsedObservation,
    ParsedSelect,
    SelectContextKind,
)
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

#: Whitelisted Pokémon **abilities** the pilot may click once per turn — the
#: top-of-leaderboard replays use ~0.9 ability-clicks/turn and it is their
#: engine (M5 analysis). Like the Trainer whitelist, effect text is prose so we
#: only auto-fire known-safe draw/accelerate abilities (they resolve with no
#: sub-select 70% of the time, else a TO_HAND / ATTACH we already handle):
#: Teal Mask Ogerpon ex (96) "Teal Dance" — attach a Grass energy from hand to
#: itself, then draw. Self-contained energy acceleration + card advantage.
_ABILITY_IDS = frozenset({96})


class RuleBasedPolicy(GreedyPolicy):
    """Rung 4: greedy pipeline + ability-clicks + bench-first energy + retreat."""

    name = "rule-based"

    def __init__(
        self,
        deck: "list[int] | None" = None,
        ability_whitelist: "frozenset[int] | None" = None,
        bench_first: bool = True,
        use_retreat: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(deck=deck, **kwargs)  # type: ignore[arg-type]
        self._ability_whitelist = (
            ability_whitelist if ability_whitelist is not None else _ABILITY_IDS
        )
        self._bench_first = bench_first
        self._use_retreat = use_retreat

    # -- bench-first energy planning --------------------------------------

    def _select_attach(
        self,
        options: tuple,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> int | None:
        """Charge the Active until its best attack is payable, then pre-charge
        the next benched attacker that still wants energy."""
        if not self._bench_first or cards is None:
            return super()._select_attach(options, obs, gs, cards)
        attach_opts = [
            (i, opt) for i, opt in enumerate(options) if opt.type is OptionKind.ATTACH
        ]
        if not attach_opts:
            return None

        # 1. Active still needs energy for its best attack -> charge it.
        if gs.my_active is not None and gs.wants_energy(gs.my_active):
            for i, opt in attach_opts:
                if opt.inPlayArea is AreaKind.ACTIVE:
                    self.last_note = "attach: charge the Active's attacker"
                    return i

        # 2. Active is ready -> pre-charge a benched attacker that wants energy.
        bench = gs.me.bench if gs.me is not None else ()
        for i, opt in attach_opts:
            if (
                opt.inPlayArea is AreaKind.BENCH
                and opt.inPlayIndex is not None
                and 0 <= opt.inPlayIndex < len(bench)
                and gs.wants_energy(bench[opt.inPlayIndex])
            ):
                self.last_note = "attach: pre-charge the next benched attacker"
                return i

        # 3. Fallback: the Active, else the first attach (never waste the drop).
        for i, opt in attach_opts:
            if opt.inPlayArea is AreaKind.ACTIVE:
                self.last_note = "attach: overflow onto the Active"
                return i
        self.last_note = "attach: first available target"
        return attach_opts[0][0]

    # -- active selection: front the fastest attacker, not the biggest body --

    def _pick_best_pokemon(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        cards: CardDatabase | None,
        want_max: int,
    ) -> list[int]:
        """Promote the Pokémon that can attack SOONEST (cheapest damaging
        attack), then hardest, then toughest.

        Greedy fronts the highest-HP body — correct for a single-attacker wall
        deck, but on a multi-attacker deck it strands cheap attackers on the
        bench behind an expensive tank that never swings (measured: a 3-energy
        ex clogging the Active while 2-energy hitters idle). Fronting the
        fastest attacker fixes that; for a wall deck whose tank *is* the main
        attacker the choice is unchanged."""
        if cards is None:
            return super()._pick_best_pokemon(select, obs, cards, want_max)
        scored: list[tuple[int, int, int, int]] = []
        for i, opt in enumerate(select.option):
            resolved = resolve_option(opt, obs)
            info = cards.get_card(resolved.card_id) if resolved.card_id is not None else None
            if info is None:
                scored.append((999, 0, 0, i))
                continue
            min_cost = 999
            best_dmg = 0
            for aid in info.attacks:
                atk = cards.get_attack(aid)
                if atk is None or atk.damage <= 0:
                    continue
                min_cost = min(min_cost, len(atk.energies))
                best_dmg = max(best_dmg, atk.damage)
            # sort key: cheapest attack, then hardest hit, then toughest body
            scored.append((min_cost, -best_dmg, -info.hp, i))
        if not scored:
            return self._safe_default(select)
        scored.sort()
        self.last_note = "promote the fastest available attacker"
        return [scored[0][3]]

    # -- retreat / promote -------------------------------------------------

    def _decide(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> list[int]:
        if select.type.name == "MAIN" and cards is not None:
            # Fire a whitelisted draw/accelerate ability first — it feeds the
            # rest of the turn (energy + cards) and never ends it, so it is safe
            # even before a lethal attack (which the next loop still takes).
            ability = self._use_ability(select, obs, cards)
            if ability is not None:
                return ability
            # Then consider retreating a stranded Active.
            if self._use_retreat:
                retreat = self._maybe_retreat(select, gs, cards)
                if retreat is not None:
                    return retreat
        return super()._decide(select, obs, gs, cards)

    def _use_ability(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        cards: CardDatabase,
    ) -> list[int] | None:
        """Click a whitelisted ability (the engine only offers each once per
        turn, so no loop guard is needed)."""
        for i, opt in enumerate(select.option):
            if opt.type is not OptionKind.ABILITY:
                continue
            resolved = resolve_option(opt, obs)
            if resolved.card_id is not None and resolved.card_id in self._ability_whitelist:
                self.last_note = "use a whitelisted ability (draw / accelerate)"
                return [i]
        return None

    def _maybe_retreat(
        self,
        select: ParsedSelect,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> list[int] | None:
        """Retreat when the Active cannot attack this turn but a benched
        Pokémon is ready to — swap tempo in rather than pass.

        Conservative on purpose: only when RETREAT is legal (the engine only
        offers it when affordable and not yet used), the Active has no usable
        damaging attack now, and some benched Pokémon does. Lethal is handled
        first by the normal flow, so this never skips a KO."""
        if not self._use_retreat:
            return None
        retreat_idx = next(
            (i for i, opt in enumerate(select.option) if opt.type is OptionKind.RETREAT),
            None,
        )
        if retreat_idx is None or gs.retreated or gs.my_active is None:
            return None
        # If the Active can already do damage now, don't retreat it.
        if self._can_attack_now(gs, gs.my_active):
            return None
        bench = gs.me.bench if gs.me is not None else ()
        if any(self._can_attack_now(gs, mon) for mon in bench):
            self.last_note = "retreat: swap a stranded Active for a ready bencher"
            return [retreat_idx]
        return None

    @staticmethod
    def _can_attack_now(gs: GameState, pokemon: "object") -> bool:
        """Whether ``pokemon`` has a damaging attack payable with its current
        energy (used to compare Active vs bench readiness)."""
        from ptcg_ai.observation.models import ParsedPokemon

        if not isinstance(pokemon, ParsedPokemon) or gs.cards is None:
            return False
        info = gs.cards.get_card(pokemon.id)
        if info is None:
            return False
        for attack_id in info.attacks:
            attack = gs.cards.get_attack(attack_id)
            if attack is None or attack.damage <= 0:
                continue
            if gs.energy_pays(pokemon.energies, attack.energies):
                return True
        return False
