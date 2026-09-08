"""Build the Kaggle submission tarball (ADR-0001, ADR-0014).

Bundle contents (all at the archive top level, as Kaggle requires):

- ``main.py``        — copied from ``submission/_entrypoint.py``; imports the
  bundled ``ptcg_ai/`` subtree and runs the greedy policy, falling back to
  inline safe-random on any failure.
- ``deck.csv``       — the configured deck.
- ``card_data.json`` — serialized static card knowledge (ADR-0014), so the
  agent needs no native engine to reason about cards.
- ``ptcg_ai/``       — a **pruned, stdlib-only** subtree with just the modules
  the decision path needs (no config/yaml, no SDK import). Copied verbatim
  from the research package with empty package ``__init__.py`` files, so the
  shipped code matches the tested code exactly (ADR-0014).
- ``cg/``            — the vendored SDK package with all native libraries
  (unused by the greedy entrypoint, bundled so search-based successors and
  the official layout match).
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config.schema import AppConfig

logger = logging.getLogger(__name__)

_ENTRYPOINT = Path(__file__).with_name("_entrypoint.py")

#: Root of the research package (``src/ptcg_ai``).
_PKG_ROOT = Path(__file__).parent.parent

#: The ONLY modules the greedy decision path imports at runtime. Kept in the
#: shipped subtree; everything else (config, registry, environment, analytics)
#: is deliberately excluded so the entrypoint stays stdlib-only (ADR-0014).
_AGENT_MODULES = (
    "observation/models.py",
    "observation/parser.py",
    "observation/resolve.py",
    "cards/database.py",
    "decision/base.py",
    "decision/greedy.py",
    "state/game_state.py",
    # Rung-6 imitation pilot (M7/M8). Pure-stdlib and inert without a bundled
    # ``bc_weights.json`` (the entrypoint falls back to greedy), so shipping
    # them always is safe: the greedy build behaves identically.
    "imitation/deck_profiles.py",
    "imitation/features.py",
    "imitation/policy.py",
    # M37: hand-written stdlib attention/layernorm for the set-transformer MAIN scorer.
    # Imported unconditionally by policy.py, inert unless a context declares
    # ``kind: setxf2_ensemble``.
    "imitation/setnet.py",
)

#: Packages that need an (empty) ``__init__.py`` in the shipped subtree.
_AGENT_PACKAGES = ("", "observation", "cards", "decision", "state", "imitation")


#: A BC payload is trained against ONE deck's card vocabulary. Below this much overlap
#: between the bundled deck and the profile's `deck_ids`, the scorer is not "slightly
#: off" -- most cards hash into the out-of-vocabulary bucket and the network sees the
#: same features no matter which card it looks at.
_MIN_DECK_PROFILE_OVERLAP = 0.80


def _assert_deck_matches_profile(weights_path: Path, card_ids) -> None:
    """Fail the build if the bundled deck is not the one the weights were trained on.

    THIS EXISTS BECAUSE IT HAPPENED (M37, submission 55203764). `build_submission` takes
    the deck from ``config.paths.deck_path``, which defaults to the SDK's sample deck. A
    bundle built without overriding it shipped ALAKAZAM-trained weights alongside a
    Snover/Mega Abomasnow deck: **0 of 9 distinct cards in the profile vocabulary**. The
    agent laddered to ~500 elo, losing at turn 4 with all six prizes untaken.

    Nothing caught it. `validate_submission` passes (the tarball is structurally fine),
    `smoke_test_entrypoint` passes with ``bc_failures: 0`` (an unknown card does not
    raise -- it scores as OOV), and the extracted-tarball check confirmed the right
    profile and feature_dim, because the PROFILE was right; the DECK was not.
    """
    import json as _json

    from ptcg_ai.imitation.deck_profiles import get_profile

    payload = _json.loads(Path(weights_path).read_text(encoding="utf-8"))
    profile = get_profile(payload.get("profile", "TR_650"))
    vocab = set(profile.deck_ids)
    if not vocab:
        return
    distinct = set(card_ids)
    overlap = len(distinct & vocab) / len(distinct)
    if overlap < _MIN_DECK_PROFILE_OVERLAP:
        raise ValueError(
            f"bundled deck does not match the weights' profile {profile.name!r}: only "
            f"{len(distinct & vocab)}/{len(distinct)} distinct cards ({overlap:.0%}) are "
            f"in its vocabulary, below the {_MIN_DECK_PROFILE_OVERLAP:.0%} floor. The "
            f"scorer would be blind (everything falls into the OOV bucket). Pass the "
            f"deck the payload was trained on via config.paths.deck_path."
        )


def build_submission(
    config: AppConfig,
    output_name: str | None = None,
    cards: CardDatabase | None = None,
    weights_path: Path | None = None,
) -> Path:
    """Stage and compress the submission; return the tarball path.

    Args:
        cards: card knowledge to serialize into ``card_data.json``. Defaults
            to loading it from the SDK at ``config.paths.sdk_dir`` (which the
            build machine has); tests may inject a small database instead.
        weights_path: optional BC weights JSON to bundle as ``bc_weights.json``.
            When present the entrypoint runs the imitation pilot; when omitted
            the bundle is a plain greedy agent (the imitation modules stay inert).

    Raises:
        FileNotFoundError: If the SDK ``cg/`` package, deck, or weights file is missing.
        ValueError: If the deck file is structurally invalid.
    """
    output_name = output_name if output_name is not None else config.submission.output_name
    if weights_path is not None and not Path(weights_path).is_file():
        raise FileNotFoundError(f"BC weights file not found: {weights_path}")
    cg_dir = config.paths.sdk_dir / "cg"
    if not cg_dir.is_dir():
        raise FileNotFoundError(f"Vendored SDK package not found: {cg_dir}")
    deck = load_deck(config.paths.deck_path)  # validates structure

    if cards is None:
        from ptcg_ai.environment.sdk import load_sdk

        cards = CardDatabase.from_sdk(load_sdk(config.paths.sdk_dir))

    build_dir = config.paths.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = build_dir / f"{output_name}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="ptcg-submission-") as tmp:
        stage = Path(tmp)
        shutil.copyfile(_ENTRYPOINT, stage / "main.py")
        (stage / "deck.csv").write_text(
            "\n".join(str(cid) for cid in deck.card_ids) + "\n", encoding="utf-8"
        )
        (stage / "card_data.json").write_text(
            json.dumps(cards.to_records(), separators=(",", ":")), encoding="utf-8"
        )
        if weights_path is not None:
            _assert_deck_matches_profile(weights_path, deck.card_ids)
            shutil.copyfile(weights_path, stage / "bc_weights.json")
        _stage_agent_package(stage / "ptcg_ai")
        shutil.copytree(
            cg_dir, stage / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )

        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add children individually so members sit at the archive top
            # level (`main.py`, not `stage/main.py`).
            for child in sorted(stage.iterdir()):
                tar.add(child, arcname=child.name)

    size_mib = tarball_path.stat().st_size / (1024 * 1024)
    logger.info("Built %s (%.1f MiB)", tarball_path, size_mib)
    return tarball_path


def _stage_agent_package(dest_root: Path) -> None:
    """Copy the curated ``ptcg_ai`` subtree with empty package inits."""
    for pkg in _AGENT_PACKAGES:
        pkg_dir = dest_root / pkg if pkg else dest_root
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    for rel in _AGENT_MODULES:
        src = _PKG_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"Agent module missing from package: {src}")
        shutil.copyfile(src, dest_root / rel)
