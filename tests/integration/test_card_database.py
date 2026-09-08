"""Card database over the live engine tables."""

import pytest

from ptcg_ai.agent.deck import load_deck, validate_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import AppConfig
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.observation.models import CardKind

pytestmark = pytest.mark.sdk


@pytest.fixture(scope="module")
def cards() -> CardDatabase:
    from ptcg_ai.utils.paths import default_sdk_dir

    return CardDatabase.from_sdk(load_sdk(default_sdk_dir()))


def test_database_is_populated(cards: CardDatabase) -> None:
    assert len(cards) > 1000
    # Card 3 is Basic Water Energy in the competition pool.
    water = cards.get_card(3)
    assert water is not None
    assert water.cardType is CardKind.BASIC_ENERGY


def test_sample_deck_fully_resolvable(cards: CardDatabase, app_config: AppConfig) -> None:
    deck = load_deck(app_config.paths.deck_path)
    assert all(cid in cards for cid in deck.card_ids)
    assert validate_deck(deck, cards) == []


def test_attacks_linked_from_cards(cards: CardDatabase, app_config: AppConfig) -> None:
    deck = load_deck(app_config.paths.deck_path)
    pokemon = [
        info for cid in set(deck.card_ids)
        if (info := cards.get_card(cid)) and info.cardType is CardKind.POKEMON
    ]
    assert pokemon, "sample deck should contain Pokémon"
    for info in pokemon:
        for attack_id in info.attacks:
            assert cards.get_attack(attack_id) is not None
