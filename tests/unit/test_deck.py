"""Deck loading and structural validation."""

import pytest

from ptcg_ai.agent.deck import Deck, load_deck, validate_deck
from tests.conftest import DECKS_DIR


def test_load_valid_deck() -> None:
    deck = load_deck(DECKS_DIR / "valid_deck.csv")
    assert len(deck.card_ids) == 60
    assert all(isinstance(cid, int) and cid > 0 for cid in deck.card_ids)


def test_load_wrong_count_raises() -> None:
    with pytest.raises(ValueError, match="expected 60"):
        load_deck(DECKS_DIR / "invalid_59_cards.csv")


def test_load_non_integer_raises() -> None:
    with pytest.raises(ValueError, match="non-integer"):
        load_deck(DECKS_DIR / "invalid_non_int.csv")


def test_validate_structural_without_card_db() -> None:
    deck = load_deck(DECKS_DIR / "valid_deck.csv")
    assert validate_deck(deck) == []

    bad = Deck(card_ids=(0,) * 60)
    problems = validate_deck(bad)
    assert any("positive" in p for p in problems)
