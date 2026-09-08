"""M11: the pure-stdlib MLP option scorer must match a numpy reference exactly
(no train/serve skew), and v1 linear payloads must keep loading unchanged."""

from __future__ import annotations

import random

import numpy as np
import pytest

from ptcg_ai.imitation.policy import _dot, _mlp_forward, _score, _validate_spec


def _random_mlp(dim: int, h: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "kind": "mlp",
        "h": h,
        "W1": (rng.standard_normal((dim, h)) * 0.3).tolist(),
        "b1": (rng.standard_normal(h) * 0.1).tolist(),
        "w2": (rng.standard_normal(h) * 0.5).tolist(),
        "b2": float(rng.standard_normal()),
    }


def _numpy_forward(spec: dict, x: list[float]) -> float:
    W1 = np.asarray(spec["W1"]); b1 = np.asarray(spec["b1"])
    w2 = np.asarray(spec["w2"]); b2 = float(spec["b2"])
    return float(w2 @ np.maximum(np.asarray(x) @ W1 + b1, 0.0) + b2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stdlib_mlp_matches_numpy(seed: int) -> None:
    dim, h = 386, 32
    spec = _random_mlp(dim, h, seed)
    rng = random.Random(seed)
    for _ in range(20):
        # sparse-ish vector like the real featurizer (many exact zeros)
        x = [0.0 if rng.random() < 0.6 else rng.uniform(-1, 1) for _ in range(dim)]
        assert _mlp_forward(spec, x) == pytest.approx(_numpy_forward(spec, x), abs=1e-9)


def test_ensemble_is_mean_of_members() -> None:
    dim, h = 386, 16
    members = [_random_mlp(dim, h, s) for s in range(3)]
    spec = {"kind": "mlp_ensemble", "members": members}
    x = [0.0 if i % 3 else 0.5 for i in range(dim)]
    expected = sum(_mlp_forward(m, x) for m in members) / 3
    assert _score(spec, x) == pytest.approx(expected)


def test_linear_spec_still_scores() -> None:
    w = [0.1 * i for i in range(386)]
    x = [1.0] * 386
    assert _score(w, x) == pytest.approx(_dot(w, x))


def test_validate_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        _validate_spec("MAIN", [1.0, 2.0], 386)          # wrong linear length
    with pytest.raises(ValueError):
        _validate_spec("MAIN", {"kind": "mlp", "W1": [[0.0]], "b1": [0.0, 0.0],
                                "w2": [0.0]}, 386)        # W1 rows != dim
    with pytest.raises(ValueError):
        _validate_spec("MAIN", {"kind": "wat"}, 386)      # unknown kind


# -- M34: per-context DeckProfile override ------------------------------------

def test_context_profile_override_validates_against_its_own_dim() -> None:
    """A context may declare its own profile; it is checked against THAT dim.

    This is what lets TO_HAND score under ALAKAZAM_FETCH (750) inside a payload
    whose other contexts — notably MAIN's MLP ensemble — stay on ALAKAZAM (658).
    """
    from ptcg_ai.imitation.deck_profiles import get_profile

    alakazam = get_profile("ALAKAZAM").feature_dim
    fetch = get_profile("ALAKAZAM_FETCH").feature_dim
    assert alakazam != fetch

    spec = {"kind": "linear", "profile": "ALAKAZAM_FETCH", "w": [0.0] * fetch}
    _validate_spec("TO_HAND", spec, alakazam)  # payload dim differs -> must not raise

    with pytest.raises(ValueError):
        _validate_spec("TO_HAND", {"kind": "linear", "profile": "ALAKAZAM_FETCH",
                                   "w": [0.0] * alakazam}, alakazam)


def test_linear_kind_spec_scores_like_a_flat_vector() -> None:
    w = [0.25 * i for i in range(64)]
    x = [1.0 if i % 2 else 0.0 for i in range(64)]
    assert _score({"kind": "linear", "w": w}, x) == pytest.approx(_dot(w, x))
