"""Determinization: parsed observation -> concrete hidden-information worlds.

``search_begin`` requires the caller to supply concrete card-ID lists for every
hidden zone (ADR-0010). We split the problem by ownership:

* **My zones** (deck order + hidden prizes) are *exact multiset bookkeeping*:
  my 60-card deck minus everything I can see of mine (hand, board with all
  attachments and pre-evolutions, discard, my stadium, revealed prizes,
  cards I am currently looking at). The remaining multiset is shuffled and
  split into a draw-ordered deck and the hidden prize slots. The invariant
  ``|pool| == deckCount + hidden-prize-count`` is asserted; a mismatch raises
  :class:`DeterminizeError` so the caller can fall back to greedy for that
  decision rather than feed the engine a wrong count.

* **Opponent zones** are mechanically-valid *filler* weighted toward their
  observed cards (discard, board, revealed prizes), padded from my deck's ID
  vocabulary to reach the required counts. This is legitimate for a depth-1
  simulation of *my* turn, which almost never consumes the opponent's hidden
  information; belief-tracking their hand is deferred (ADR-0010, out of scope).

SDK-free: operates only on parsed models + the card database, so it is
unit-testable from literal fixtures.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.observation.models import (
    ParsedCard,
    ParsedObservation,
    ParsedPlayer,
    ParsedPokemon,
    ParsedState,
)


class DeterminizeError(RuntimeError):
    """The visible cards could not be reconciled with the deck list.

    Raised when the my-zone bookkeeping invariant fails (a seen card is not in
    the deck, or the pool size does not match the hidden-zone counts). The
    search policy treats this as "cannot determinize this decision" and falls
    back to greedy — never to a wrong-count ``search_begin`` call.
    """


@dataclass(frozen=True)
class HiddenInfo:
    """Concrete card-ID lists for one determinized world, in ``search_begin``
    argument order. List order is draw order for the decks."""

    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]

    def as_args(self) -> tuple[list[int], ...]:
        return (
            self.your_deck, self.your_prize, self.opponent_deck,
            self.opponent_prize, self.opponent_hand, self.opponent_active,
        )


class Determinizer:
    """Samples :class:`HiddenInfo` worlds from an observation.

    ``my_deck`` is the agent's own 60-card list (the ground truth for my
    hidden zones). ``cards`` supplies Basic-Pokémon knowledge for legal
    opponent guesses; without it, opponent filler still satisfies counts but
    cannot guarantee a Basic (tolerated — the engine errors are caught).
    """

    def __init__(self, my_deck: list[int], cards: CardDatabase | None, rng: random.Random) -> None:
        self._deck = Counter(my_deck)
        self._deck_ids = list(dict.fromkeys(my_deck))  # unique, stable order
        self._cards = cards
        self._rng = rng
        self._basics = self._basic_ids(my_deck, cards)

    def sample(self, obs: ParsedObservation, n: int) -> list[HiddenInfo]:
        """Return ``n`` independently-shuffled determinized worlds.

        Raises:
            DeterminizeError: if the observation has no board state, or the
                my-zone bookkeeping invariant fails.
        """
        state = obs.current
        if state is None:
            raise DeterminizeError("no board state to determinize")
        me = state.me
        opp = state.opponent

        pool, hidden_prizes, deck_hidden = self._my_unknown_pool(state, me, obs)
        opp_filler = self._opponent_filler(state, opp)
        return [
            self._one_world(pool, hidden_prizes, deck_hidden, me, opp_filler)
            for _ in range(n)
        ]

    # -- my zones (exact) -------------------------------------------------

    def _my_unknown_pool(
        self, state: ParsedState, me: ParsedPlayer, obs: ParsedObservation
    ) -> tuple[list[int], int, bool]:
        """Compute my hidden multiset. Returns (pool_ids, hidden_prize_count,
        deck_is_revealed). When my deck is revealed (``select.deck`` present),
        the deck contents are known to the engine and ``your_deck`` is ignored;
        the pool then holds only my hidden prizes.
        """
        # A face-down card in my OWN Active slot (initial simultaneous setup
        # reveal) is committed but its ID is hidden even from me, so it cannot
        # be reconciled with the deck list. These are rare setup draws where
        # greedy is adequate — fall back rather than mis-determinize.
        if any(slot is None for slot in me.active):
            raise DeterminizeError("own active is face-down (setup); fall back")

        pool = self._deck.copy()
        seen_serials: set[int] = set()

        def take(card_id: int, serial: int) -> None:
            pool[card_id] -= 1
            if pool[card_id] < 0:
                raise DeterminizeError(f"saw card {card_id} not accounted for in deck")
            seen_serials.add(serial)

        for card in me.hand or ():
            take(card.id, card.serial)
        for card in me.discard:
            take(card.id, card.serial)
        for pkmn in _iter_board(me):
            self._take_pokemon(pkmn, take)
        for card in state.stadium:
            if card.playerIndex == state.yourIndex:
                take(card.id, card.serial)
        for card in state.looking or ():
            if card is not None and card.playerIndex == state.yourIndex:
                take(card.id, card.serial)
        for card in me.prize:
            if card is not None:  # a revealed prize is a known, accounted card
                take(card.id, card.serial)

        # A card mid-effect-resolution (a played Item/Supporter) sits in limbo —
        # out of hand, not yet in discard. But ``effect``/``contextCard`` can
        # also point at a card already on the board (e.g. the Active paying an
        # attack cost); a serial already counted must not be subtracted again.
        select = obs.select
        if select is not None:
            for limbo in (select.effect, select.contextCard):
                if (limbo is not None and limbo.playerIndex == state.yourIndex
                        and limbo.serial not in seen_serials):
                    take(limbo.id, limbo.serial)

        deck_revealed = select is not None and select.deck is not None
        if deck_revealed:
            for card in select.deck or ():
                take(card.id, card.serial)

        hidden_prizes = sum(1 for p in me.prize if p is None)
        pool_ids = list(pool.elements())
        expected = (0 if deck_revealed else me.deckCount) + hidden_prizes
        if len(pool_ids) != expected:
            raise DeterminizeError(
                f"pool size {len(pool_ids)} != expected {expected} "
                f"(deckCount={me.deckCount}, hidden_prizes={hidden_prizes}, "
                f"deck_revealed={deck_revealed})"
            )
        return pool_ids, hidden_prizes, deck_revealed

    def _one_world(
        self,
        pool_ids: list[int],
        hidden_prizes: int,
        deck_revealed: bool,
        me: ParsedPlayer,
        opp_filler: tuple[list[int], list[int], list[int], list[int]],
    ) -> HiddenInfo:
        shuffled = pool_ids[:]
        self._rng.shuffle(shuffled)
        if deck_revealed:
            your_deck: list[int] = []
            prize_pool = shuffled  # entire pool is hidden prizes
        else:
            your_deck = shuffled[: me.deckCount]
            prize_pool = shuffled[me.deckCount:]
        your_prize = self._fill_prizes(me.prize, prize_pool)
        opp_deck, opp_prize, opp_hand, opp_active = opp_filler
        return HiddenInfo(
            your_deck=your_deck,
            your_prize=your_prize,
            opponent_deck=opp_deck,
            opponent_prize=opp_prize,
            opponent_hand=opp_hand,
            opponent_active=opp_active,
        )

    @staticmethod
    def _fill_prizes(prize: tuple[ParsedCard | None, ...], hidden_pool: list[int]) -> list[int]:
        """One ID per prize slot: revealed slots keep their card, hidden slots
        draw from the pool (consumed in order)."""
        out: list[int] = []
        it = iter(hidden_pool)
        for slot in prize:
            out.append(slot.id if slot is not None else next(it))
        return out

    def _take_pokemon(self, pkmn: ParsedPokemon, take) -> None:
        take(pkmn.id, pkmn.serial)
        for card in pkmn.energyCards:
            take(card.id, card.serial)
        for card in pkmn.tools:
            take(card.id, card.serial)
        for card in pkmn.preEvolution:
            take(card.id, card.serial)

    # -- opponent zones (filler) ------------------------------------------

    def _opponent_filler(
        self, state: ParsedState, opp: ParsedPlayer
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        vocab = self._opponent_vocab(state, opp)
        basic = self._first_basic(vocab)

        def fill(count: int, basic_first: bool) -> list[int]:
            if count <= 0:
                return []
            ids: list[int] = []
            if basic_first and basic is not None:
                ids.append(basic)
            i = 0
            while len(ids) < count:
                ids.append(vocab[i % len(vocab)])
                i += 1
            return ids[:count]

        opp_deck = fill(max(opp.deckCount, 0), basic_first=True)
        opp_prize = fill(len(opp.prize), basic_first=False)
        opp_hand = fill(opp.handCount, basic_first=False)
        opp_active: list[int] = []
        active_facedown = bool(opp.active) and opp.active[0] is None
        if active_facedown and basic is not None:
            opp_active = [basic]
        return opp_deck, opp_prize, opp_hand, opp_active

    def _opponent_vocab(self, state: ParsedState, opp: ParsedPlayer) -> list[int]:
        """Card-ID vocabulary for opponent filler: their observed cards first
        (real, valid IDs), padded with my deck's IDs to guarantee non-empty."""
        seen: list[int] = []
        for card in opp.discard:
            seen.append(card.id)
        for pkmn in _iter_board(opp):
            seen.append(pkmn.id)
            seen.extend(c.id for c in pkmn.energyCards)
            seen.extend(c.id for c in pkmn.tools)
        for card in opp.prize:
            if card is not None:
                seen.append(card.id)
        # de-duplicate preserving order, then pad with my deck vocabulary
        vocab = list(dict.fromkeys(seen)) + self._deck_ids
        return vocab or self._deck_ids or [1]

    def _first_basic(self, vocab: list[int]) -> int | None:
        if self._cards is None:
            return self._basics[0] if self._basics else None
        for cid in vocab:
            info = self._cards.get_card(cid)
            if info is not None and info.is_basic_pokemon:
                return cid
        return self._basics[0] if self._basics else None

    @staticmethod
    def _basic_ids(deck: list[int], cards: CardDatabase | None) -> list[int]:
        if cards is None:
            return []
        out: list[int] = []
        for cid in dict.fromkeys(deck):
            info = cards.get_card(cid)
            if info is not None and info.is_basic_pokemon:
                out.append(cid)
        return out


def _iter_board(player: ParsedPlayer):
    """Yield every Pokémon in play for a player (active then bench)."""
    for slot in player.active:
        if slot is not None:
            yield slot
    yield from player.bench
