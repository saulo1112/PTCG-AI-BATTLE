"""M48 — a critic payload of the wrong width must fail loudly, not silently.

`LearnedEvaluator.value` scores with ``zip(weights, feats, mean, std)``, and `zip` stops
at the shortest input **without raising**. So a critic fitted on a different feature set
than the evaluator computes would not error: it would apply the surviving weights to the
wrong features and the search would run on a corrupted leaf value, invisibly.

The risk is not hypothetical. In this repo right now:

    value_features               -> 7  (base-7)
    value_features_v2_rooted     -> 8  (base-7 + prose_threat, M17)
    data/models/v_alakazam_v2.json  ->  10 weights (M47's vf2 refit)

Feeding the 10-weight payload to either evaluator would have used a prefix of the weights
against features they were never fitted for. This is the same failure SHAPE as submission
55203764, where the wrong deck shipped and every structural check passed (M37).
"""

from __future__ import annotations

import pytest

from ptcg_ai.decision.search_bc import LearnedEvaluator


def _model(n):
    return {"mean": [0.0] * n, "std": [1.0] * n, "weights": [0.1] * n, "bias": 0.0}


def test_a_matching_payload_loads():
    ev = LearnedEvaluator(_model(7))
    assert len(ev._w) == 7


def test_a_wider_payload_is_rejected_instead_of_being_truncated():
    with pytest.raises(ValueError, match="scores 7 features but the critic payload has 10"):
        LearnedEvaluator(_model(10))


def test_a_narrower_payload_is_rejected_too():
    with pytest.raises(ValueError, match="scores 7 features but the critic payload has 5"):
        LearnedEvaluator(_model(5))


def test_the_error_names_the_features_when_the_payload_records_them():
    model = _model(10)
    model["features"] = ["prize", "threat", "reserve", "survival", "energy", "hand",
                         "deck_out", "pw_threat", "board_prize", "prose_threat"]
    with pytest.raises(ValueError, match="prose_threat"):
        LearnedEvaluator(model)


def test_an_internally_inconsistent_payload_is_rejected():
    model = {"mean": [0.0] * 7, "std": [1.0] * 6, "weights": [0.1] * 7, "bias": 0.0}
    with pytest.raises(ValueError, match="inconsistent"):
        LearnedEvaluator(model)


def test_a_subclass_declares_its_own_width():
    """M17's evaluator computes 8 features, so an 8-wide payload must load THERE and a
    7-wide one must not -- the guard has to follow the subclass, not the base class."""
    class _Eight(LearnedEvaluator):
        n_features = 8

    assert len(_Eight(_model(8))._w) == 8
    with pytest.raises(ValueError, match="scores 8 features"):
        _Eight(_model(7))
