"""Greedy one-step heuristics — ladder rung 3 (ADR-0013).

The Scorer here is a fixed priority order over the current legal options; no
lookahead. It targets exactly the failures the real ladder replays exposed
(docs/replay_analysis.md): an undeveloped bench, an unpowered attacker, and
ending the turn with a useful action still available. The ~38 contexts that
never occur under our deck (docs/game_analysis.md §1) fall through to a safe
deterministic default; ``SafePolicy`` is still the crash backstop.

Priority at MAIN (pick the single highest-priority legal action; the engine
loops back for the next one until we attack or END):

1. **Lethal attack** — if an attack KOs the opponent's Active, take it.
2. **Deck-search Item** — play a whitelisted search Item (finds Basics/energy
   without touching the hand): pure card advantage, no downside.
3. **Develop bench** — play a Basic Pokémon to an empty bench slot.
4. **Attach energy** — to the Active (the attacker) if we haven't this turn.
5. **Evolve** — if an evolution is available.
6. **Draw Supporter** — refresh a now-spent hand with a whitelisted draw
   Supporter (played late so it never shuffles away a card we could have used).
7. **Attack** — any legal (non-lethal) attack, rather than waste the turn.
8. **END** — only when nothing above applies.

Trainer play is gated to an explicit **card-ID whitelist** of pure
consistency cards (M3, docs/replay_analysis.md): the effect semantics of an
arbitrary Trainer are English prose (feature_inventory.md §9) and cannot be
read structurally, so we only auto-play cards whose effect is known-safe
(search/draw). Board-reading Trainers (Switch, Boss's Orders, tools, stadiums)
wait for the rung-4 evaluator (M4). The whitelist is deck-specific by design
and lives beside the deck it enables.
"""

from __future__ import annotations

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import BasePolicy, DecisionContext
from ptcg_ai.observation.models import (
    AreaKind,
    CardKind,
    OptionKind,
    ParsedObservation,
    ParsedSelect,
    SelectContextKind,
)
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

#: Deck-search Items that add cards to hand WITHOUT touching the current hand
#: (pure upside), played early to dig for what the deck needs:
#: Poké Pad (1152, any Pokémon), Dusk Ball (1102, Pokémon), Fighting Gong
#: (1142, {F} energy/Pokémon), Mega Signal (1145, a Mega ex — fetches the win
#: condition; empirically resolves with no sub-select, M4/W2). Items don't use
#: the once-per-turn Supporter slot, so they are always safe to fire.
_SEARCH_ITEM_IDS = frozenset({1152, 1102, 1142, 1145})

#: Fetch/accelerate Supporters, played BEFORE the draw Supporter (they set up
#: the board; the draw is the fallback that refreshes leftover chaff):
#: Cyrano (1205, search ex → hand), Waitress (1235, accelerate a Basic energy
#: from the top 6 — triggers an ATTACH_FROM sub-select we handle). Both are
#: pure-value (no hand discard). They share the one Supporter slot with the
#: draw Supporter, so the arena decides whether fetch-before-draw is net-good
#: per deck (M4/W2 ablation).
_FETCH_SUPPORTER_IDS = frozenset({1205, 1235})

#: Hand-refresh draw Supporters, low downside (cards return to the deck):
#: Lillie's Determination (1227) — shuffle hand into deck, draw 6. Played last
#: (only chaff remains). Carmine is deliberately excluded (it DISCARDS the hand
#: — higher downside; add only if the arena shows we need more draw).
_DRAW_SUPPORTER_IDS = frozenset({1227})


class GreedyPolicy(BasePolicy):
    """Rung 3: fixed-priority heuristics over the current options.

    The Trainer whitelists are constructor-injectable so the arena can ablate
    them (e.g. reproduce the M3 policy) and so a future deck can carry its own
    known-safe Trainer set without editing this class.
    """

    name = "greedy"

    def __init__(
        self,
        deck: list[int] | None = None,
        search_items: frozenset[int] | None = None,
        fetch_supporters: frozenset[int] | None = None,
        draw_supporters: frozenset[int] | None = None,
    ) -> None:
        super().__init__(deck=deck)
        self._search_items = search_items if search_items is not None else _SEARCH_ITEM_IDS
        self._fetch_supporters = (
            fetch_supporters if fetch_supporters is not None else _FETCH_SUPPORTER_IDS
        )
        self._draw_supporters = (
            draw_supporters if draw_supporters is not None else _DRAW_SUPPORTER_IDS
        )

    def choose(self, ctx: DecisionContext) -> list[int]:
        select = ctx.observation.select
        if select is None:
            raise ValueError("choose() called on a deck-submission observation")
        n = len(select.option)
        if n == 0:  # terminal/stale select (battle_flow.md) — nothing to pick
            self.last_note = "empty option list; returning nothing"
            return []

        gs = GameState.build(ctx.observation, ctx.cards)
        chosen = self._decide(select, ctx.observation, gs, ctx.cards)
        return self._legalize(chosen, select)

    # -- routing ----------------------------------------------------------

    def _decide(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> list[int]:
        context = select.context
        if select.type.name == "MAIN":
            return self._main(select, obs, gs, cards)
        if context is SelectContextKind.IS_FIRST:
            return self._is_first(select)
        if context in (
            SelectContextKind.SETUP_ACTIVE_POKEMON,
            SelectContextKind.TO_ACTIVE,
            SelectContextKind.SWITCH,
        ):
            return self._pick_best_pokemon(select, obs, cards, want_max=1)
        if context is SelectContextKind.SETUP_BENCH_POKEMON:
            # Develop: bench as many Basics as the engine allows at once.
            return list(range(min(select.maxCount, len(select.option))))
        if context is SelectContextKind.TO_HAND:
            # Deck-search results (Poké Pad / Dusk Ball / Fighting Gong /
            # Mega Signal / Cyrano / a draw effect): keep the most useful
            # cards — Basics, then energy, then other Pokémon.
            return self._pick_search_targets(select, obs, cards)
        if context in (SelectContextKind.ATTACH_FROM, SelectContextKind.ATTACH_TO):
            # Mid-effect energy attach (e.g. Waitress): put it on the Active
            # attacker if that Pokémon is a legal target, else the first.
            return self._pick_attach_target(select)
        if select.type.name == "COUNT":
            return self._max_number(select)
        return self._safe_default(select)

    # -- MAIN -------------------------------------------------------------

    def _main(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> list[int]:
        options = select.option

        # 1. Lethal attack.
        best_lethal: tuple[int, int] | None = None  # (damage, index)
        attack_indices: list[tuple[int, int]] = []  # (damage, index) for any attack
        if cards is not None and gs.my_active is not None and gs.opp_active is not None:
            for i, opt in enumerate(options):
                if opt.type is not OptionKind.ATTACK or opt.attackId is None:
                    continue
                attack = cards.get_attack(opt.attackId)
                if attack is None:
                    continue
                dmg = gs.attack_damage(attack, gs.my_active, gs.opp_active)
                attack_indices.append((dmg, i))
                if dmg >= gs.opp_active.hp:
                    if best_lethal is None or dmg > best_lethal[0]:
                        best_lethal = (dmg, i)
        if best_lethal is not None:
            self.last_note = f"lethal attack (dmg {best_lethal[0]} ≥ {gs.opp_active.hp} HP)"
            return [best_lethal[1]]

        # 2. Deck-search Item: dig for Basics/energy/Mega (no hand cost, no
        #    Supporter slot used).
        play = self._first_whitelisted_play(options, obs, cards, self._search_items)
        if play is not None:
            self.last_note = "play a deck-search Item to dig for resources"
            return [play]

        # 3. Develop bench: play a Basic Pokémon while there is room.
        if gs.bench_room > 0:
            play = self._first_basic_play(options, obs, cards)
            if play is not None:
                self.last_note = "develop: play a Basic Pokémon to the bench"
                return [play]

        # 4. Attach energy if not done this turn (target chosen by _select_attach:
        #    rung 3 = the Active; rung 4 overrides with bench-first planning).
        if not gs.energy_attached:
            attach = self._select_attach(options, obs, gs, cards)
            if attach is not None:
                self.last_note = "attach energy toward the active attacker"
                return [attach]

        # 5. Evolve if we can.
        for i, opt in enumerate(options):
            if opt.type is OptionKind.EVOLVE:
                self.last_note = "evolve to improve the board"
                return [i]

        # 6. Set up the board with a fetch/accelerate Supporter (before the
        #    generic draw): search an ex or accelerate energy.
        if not gs.supporter_played:
            play = self._first_whitelisted_play(options, obs, cards, self._fetch_supporters)
            if play is not None:
                self.last_note = "play a fetch/accelerate Supporter"
                return [play]

        # 7. Refresh a spent hand with a draw Supporter (only chaff remains).
        if not gs.supporter_played:
            play = self._first_whitelisted_play(options, obs, cards, self._draw_supporters)
            if play is not None:
                self.last_note = "play a draw Supporter to refresh the hand"
                return [play]

        # 8. Any legal attack rather than waste the turn.
        if attack_indices:
            dmg, idx = max(attack_indices, key=lambda t: t[0])
            self.last_note = f"attack for tempo (dmg {dmg})"
            return [idx]
        # No card knowledge but an attack is offered — take the first one.
        for i, opt in enumerate(options):
            if opt.type is OptionKind.ATTACK:
                self.last_note = "attack for tempo (no card data)"
                return [i]

        # 9. END.
        for i, opt in enumerate(options):
            if opt.type is OptionKind.END:
                self.last_note = "no useful action; end turn"
                return [i]
        return [0]

    # -- context helpers --------------------------------------------------

    def _first_basic_play(
        self,
        options: tuple,
        obs: ParsedObservation,
        cards: CardDatabase | None,
    ) -> int | None:
        """Index of a PLAY option that puts a Basic Pokémon into play."""
        fallback_play: int | None = None
        for i, opt in enumerate(options):
            if opt.type is not OptionKind.PLAY:
                continue
            if cards is None:
                continue
            resolved = resolve_option(opt, obs)
            if resolved.card_id is None:
                continue
            info = cards.get_card(resolved.card_id)
            if info is not None and info.is_basic_pokemon:
                return i
        return fallback_play

    def _first_whitelisted_play(
        self,
        options: tuple,
        obs: ParsedObservation,
        cards: CardDatabase | None,
        id_set: frozenset[int],
    ) -> int | None:
        """Index of a PLAY option whose card ID is in ``id_set`` (a known-safe
        Trainer). Needs card knowledge to resolve the played card's identity."""
        if cards is None:
            return None
        for i, opt in enumerate(options):
            if opt.type is not OptionKind.PLAY:
                continue
            resolved = resolve_option(opt, obs)
            if resolved.card_id is not None and resolved.card_id in id_set:
                return i
        return None

    def _pick_search_targets(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        cards: CardDatabase | None,
    ) -> list[int]:
        """Choose which searched cards to take to hand: Basics first (to
        develop the bench), then energy (to power attacks), then anything else.
        Takes up to ``maxCount``; ``choose`` legalizes toward ``minCount``."""
        ranked: list[tuple[int, int]] = []  # (rank, index); lower rank = better
        for i, opt in enumerate(select.option):
            rank = 3
            if cards is not None:
                resolved = resolve_option(opt, obs)
                info = (
                    cards.get_card(resolved.card_id)
                    if resolved.card_id is not None
                    else None
                )
                if info is not None:
                    if info.is_basic_pokemon:
                        rank = 0
                    elif info.is_basic_energy:
                        rank = 1
                    elif info.cardType is CardKind.POKEMON:
                        rank = 2
            ranked.append((rank, i))
        ranked.sort(key=lambda t: (t[0], t[1]))
        k = min(max(select.maxCount, 0), len(select.option))
        self.last_note = "search: take Basics, then energy"
        return [i for _, i in ranked[:k]]

    def _select_attach(
        self,
        options: tuple,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> int | None:
        """Which Pokémon gets the turn's one energy attachment. Rung 3 attaches
        to the Active; rung 4 (:class:`RuleBasedPolicy`) overrides this with
        bench-first planning."""
        return self._attach_to_active(options)

    def _attach_to_active(self, options: tuple) -> int | None:
        """Prefer attaching to the Active; else any attach; else None."""
        first_attach: int | None = None
        for i, opt in enumerate(options):
            if opt.type is not OptionKind.ATTACH:
                continue
            if first_attach is None:
                first_attach = i
            if opt.inPlayArea is AreaKind.ACTIVE:
                return i
        return first_attach

    def _pick_attach_target(self, select: ParsedSelect) -> list[int]:
        """Mid-effect attach (Waitress etc.): choose the target Pokémon option
        pointing at the Active (the attacker), else the first legal option.
        Options here are CARD selects carrying an ``area`` for the target."""
        for i, opt in enumerate(select.option):
            if opt.area is AreaKind.ACTIVE:
                self.last_note = "mid-effect attach: put energy on the Active"
                return [i]
        self.last_note = "mid-effect attach: no Active target, take the first"
        return self._safe_default(select)

    def _pick_best_pokemon(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        cards: CardDatabase | None,
        want_max: int,
    ) -> list[int]:
        """Choose the option whose Pokémon has the most HP (best body)."""
        best: tuple[int, int] | None = None  # (hp, index)
        for i, opt in enumerate(select.option):
            hp = 0
            if cards is not None:
                resolved = resolve_option(opt, obs)
                if resolved.card_id is not None:
                    info = cards.get_card(resolved.card_id)
                    if info is not None:
                        hp = info.hp
            if best is None or hp > best[0]:
                best = (hp, i)
        if best is None:
            return self._safe_default(select)
        self.last_note = "promote the highest-HP available Pokémon"
        return [best[1]]

    def _is_first(self, select: ParsedSelect) -> list[int]:
        """Go first by default (tempo for setup-hungry decks; open A/B, see
        docs/phase2_blueprint.md §4.3)."""
        for i, opt in enumerate(select.option):
            if opt.type is OptionKind.YES:
                self.last_note = "choose to go first"
                return [i]
        return self._safe_default(select)

    def _max_number(self, select: ParsedSelect) -> list[int]:
        """For COUNT selects (e.g. DRAW_COUNT), take the largest offered
        number — more cards is better, deck-out risk is negligible at rung 3."""
        best: tuple[int, int] | None = None  # (number, index)
        for i, opt in enumerate(select.option):
            value = opt.number if opt.number is not None else 0
            if best is None or value > best[0]:
                best = (value, i)
        if best is None:
            return self._safe_default(select)
        self.last_note = f"draw/count the maximum ({best[0]})"
        return [best[1]]

    def _safe_default(self, select: ParsedSelect) -> list[int]:
        """Deterministic legal answer for unhandled contexts: the first
        ``minCount`` options (never worse than random in expectation)."""
        n = len(select.option)
        k = min(max(select.minCount, 0), n)
        self.last_note = "unhandled context; safe default (first legal)"
        return list(range(k))

    # -- output safety ----------------------------------------------------

    @staticmethod
    def _legalize(chosen: list[int], select: ParsedSelect) -> list[int]:
        """Guarantee unique, in-range indices with length in [minCount, maxCount]."""
        n = len(select.option)
        seen: list[int] = []
        for idx in chosen:
            if 0 <= idx < n and idx not in seen:
                seen.append(idx)
        lo = min(max(select.minCount, 0), n)
        hi = min(select.maxCount, n)
        # Grow toward minCount with the lowest unused indices.
        if len(seen) < lo:
            for idx in range(n):
                if idx not in seen:
                    seen.append(idx)
                    if len(seen) >= lo:
                        break
        # Trim toward maxCount.
        if hi >= lo and len(seen) > hi:
            seen = seen[:hi]
        return seen
