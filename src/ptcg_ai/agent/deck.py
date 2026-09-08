"""Deck loading and validation.

Mirrors the engine's own deck checks (unknown card, copy limit, no Basic
Pokémon, ACE SPEC limit) so problems surface before a battle or submission
instead of as a ``BattleStartError``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config.schema import DECK_SIZE

#: Maximum copies of a same-name card (basic Energy exempt) — TCG rule.
MAX_COPIES = 4


@dataclass(frozen=True)
class Deck:
    """An ordered 60-card list (order is irrelevant; the engine shuffles)."""

    card_ids: tuple[int, ...]

    def as_list(self) -> list[int]:
        return list(self.card_ids)


def load_deck(path: Path) -> Deck:
    """Read a ``deck.csv`` (one card ID per line, 60 lines).

    Raises:
        ValueError: On wrong line count or non-integer content.
    """
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) != DECK_SIZE:
        raise ValueError(f"{path}: expected {DECK_SIZE} card lines, found {len(lines)}")
    try:
        ids = tuple(int(ln) for ln in lines)
    except ValueError as exc:
        raise ValueError(f"{path}: non-integer card ID: {exc}") from None
    return Deck(card_ids=ids)


def validate_deck(deck: Deck, cards: CardDatabase | None = None) -> list[str]:
    """Return a list of rule violations (empty list = valid).

    Structural checks always run; the card-knowledge checks (copy limits,
    Basic Pokémon presence, ACE SPEC) run only when a :class:`CardDatabase`
    is provided.
    """
    problems: list[str] = []
    if len(deck.card_ids) != DECK_SIZE:
        problems.append(f"deck has {len(deck.card_ids)} cards; must be {DECK_SIZE}")
    if any(cid <= 0 for cid in deck.card_ids):
        problems.append("card IDs must be positive integers")
    if cards is None:
        return problems

    unknown = sorted({cid for cid in deck.card_ids if cid not in cards})
    if unknown:
        problems.append(f"unknown card IDs: {unknown}")
        return problems  # remaining checks need valid cards

    name_counts: Counter[str] = Counter()
    ace_spec_count = 0
    has_basic_pokemon = False
    for cid in deck.card_ids:
        info = cards.get_card(cid)
        assert info is not None
        if not info.is_basic_energy:
            name_counts[info.name] += 1
        ace_spec_count += info.aceSpec
        has_basic_pokemon = has_basic_pokemon or info.is_basic_pokemon

    for name, count in sorted(name_counts.items()):
        if count > MAX_COPIES:
            problems.append(f"{count} copies of {name!r} (max {MAX_COPIES})")
    if ace_spec_count > 1:
        problems.append(f"{ace_spec_count} ACE SPEC cards (max 1)")
    if not has_basic_pokemon:
        problems.append("deck contains no Basic Pokémon")
    return problems
