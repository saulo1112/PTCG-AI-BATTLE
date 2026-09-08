"""Submission build + validation with a dummy SDK dir (no engine needed).

The entrypoint itself is stdlib-only and cg-independent, so the smoke test
runs for real even against the dummy cg/.
"""

from pathlib import Path

import dataclasses
import pytest

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.config import AppConfig
from ptcg_ai.config.schema import PathsConfig
from ptcg_ai.observation.models import CardKind, EnergyKind
from ptcg_ai.submission.builder import build_submission
from ptcg_ai.submission.validate import smoke_test_entrypoint, validate_submission
from tests.conftest import DECKS_DIR


def _tiny_cards() -> CardDatabase:
    """A minimal real card DB so builds are SDK-free (dummy cg can't load one)."""
    mk = lambda cid, **kw: CardInfo(  # noqa: E731
        cardId=cid, name=f"c{cid}", cardType=CardKind.POKEMON, retreatCost=1, hp=60,
        weakness=None, resistance=None, energyType=EnergyKind.WATER, basic=True,
        stage1=False, stage2=False, ex=False, megaEx=False, tera=False,
        aceSpec=False, evolvesFrom=None, skills=(), attacks=kw.get("attacks", ()),
    )
    return CardDatabase(
        {721: mk(721, attacks=(900,)), 722: mk(722)},
        {900: AttackInfo(attackId=900, name="Surf", text="", damage=30,
                         energies=(EnergyKind.WATER,))},
    )


@pytest.fixture()
def dummy_config(tmp_path: Path) -> AppConfig:
    sdk_dir = tmp_path / "sdk"
    cg = sdk_dir / "cg"
    cg.mkdir(parents=True)
    (cg / "__init__.py").write_text("", encoding="utf-8")
    (cg / "libcg.so").write_bytes(b"\x00dummy")
    (cg / "cg.dll").write_bytes(b"\x00dummy")
    return AppConfig(
        paths=PathsConfig(
            sdk_dir=sdk_dir,
            deck_path=DECKS_DIR / "valid_deck.csv",
            opponent_deck_path=DECKS_DIR / "valid_deck.csv",
            build_dir=tmp_path / "build",
            replay_dir=tmp_path / "replays",
        )
    )


def test_build_and_validate(dummy_config: AppConfig) -> None:
    tarball = build_submission(dummy_config, output_name="test-sub", cards=_tiny_cards())
    assert tarball.name == "test-sub.tar.gz"
    assert validate_submission(tarball) == []


def test_entrypoint_smoke_runs_greedy(dummy_config: AppConfig) -> None:
    tarball = build_submission(dummy_config, cards=_tiny_cards())
    # Empty problem list means: deck call works, greedy engaged, MAIN legal.
    assert smoke_test_entrypoint(tarball) == []


def test_size_limit_flagged(dummy_config: AppConfig) -> None:
    tarball = build_submission(dummy_config, cards=_tiny_cards())
    problems = validate_submission(tarball, size_limit_mib=0.000001)
    assert any("limit" in p for p in problems)


def test_missing_pieces_flagged(dummy_config: AppConfig, tmp_path: Path) -> None:
    import tarfile

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        marker = tmp_path / "unrelated.txt"
        marker.write_text("nothing", encoding="utf-8")
        tar.add(marker, arcname="unrelated.txt")
    problems = validate_submission(bad)
    assert any("main.py" in p for p in problems)
    assert any("deck.csv" in p for p in problems)
    assert any("libcg.so" in p for p in problems)


def test_missing_sdk_raises(dummy_config: AppConfig, tmp_path: Path) -> None:
    config = dataclasses.replace(
        dummy_config,
        paths=dataclasses.replace(dummy_config.paths, sdk_dir=tmp_path / "nope"),
    )
    with pytest.raises(FileNotFoundError):
        build_submission(config)
