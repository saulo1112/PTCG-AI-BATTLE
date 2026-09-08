"""The bundled deck must be the one the BC weights were trained on.

REGRESSION TEST FOR A REAL, EXPENSIVE FAILURE (M37, submission 55203764). A bundle went
to the ladder carrying ALAKAZAM-trained weights next to the SDK's default sample deck --
Snover/Mega Abomasnow, **zero** cards in common with the profile's vocabulary. Every card
hashed into the out-of-vocabulary bucket, so the scorer saw identical features for every
option and the agent laddered to ~500 elo, losing at turn 4 with all six prizes untaken.

Every existing guard passed it:
  * `validate_submission` -- the tarball is structurally perfect;
  * `smoke_test_entrypoint` -- `bc_failures: 0`, because an unknown card does not raise,
    it just scores as OOV;
  * the extracted-tarball check -- profile and feature_dim were both CORRECT. The profile
    was never the problem; the deck was.

So the failure is invisible to every structural check by construction, and only a
deck-vs-vocabulary comparison catches it.
"""

from __future__ import annotations

import pytest

from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.submission.builder import (
    _MIN_DECK_PROFILE_OVERLAP,
    _assert_deck_matches_profile,
)


@pytest.fixture
def payload(tmp_path):
    import json

    p = tmp_path / "w.json"
    p.write_text(json.dumps({
        "profile": "ALAKAZAM",
        "feature_dim": DP.ALAKAZAM.feature_dim,
        "contexts": {},
    }), encoding="utf-8")
    return p


def test_matching_deck_passes(payload) -> None:
    _assert_deck_matches_profile(payload, list(DP.ALAKAZAM.deck_ids) * 3)


def test_the_exact_failure_that_shipped_is_rejected(payload) -> None:
    """The real cards from the sample deck that went out as submission 55203764."""
    wrong = [3] * 35 + [722] * 4 + [723] * 4 + [1145] * 4 + [1227] * 4 + \
            [1235] * 4 + [721] * 2 + [1205] * 2 + [1158]
    assert not (set(wrong) & set(DP.ALAKAZAM.deck_ids)), "fixture must have ZERO overlap"
    with pytest.raises(ValueError, match="does not match the weights' profile"):
        _assert_deck_matches_profile(payload, wrong)


def test_partial_overlap_below_the_floor_is_rejected(payload) -> None:
    """A deck sharing SOME cards is still rejected -- the failure is graded, not binary."""
    vocab = list(DP.ALAKAZAM.deck_ids)
    keep = max(1, int(len(vocab) * (_MIN_DECK_PROFILE_OVERLAP - 0.3)))
    mixed = vocab[:keep] + list(range(90001, 90001 + len(vocab) - keep))
    with pytest.raises(ValueError, match="OOV bucket"):
        _assert_deck_matches_profile(payload, mixed)


def test_greedy_bundles_are_unaffected() -> None:
    """No weights means no profile to match; the check must not fire on greedy builds.

    `build_submission` only calls the assertion when `weights_path is not None`, so this
    pins the contract rather than the implementation detail.
    """
    import inspect

    from ptcg_ai.submission import builder

    src = inspect.getsource(builder.build_submission)
    assert "if weights_path is not None:" in src
    idx = src.index("_assert_deck_matches_profile")
    assert src.rindex("if weights_path is not None:", 0, idx) > 0
