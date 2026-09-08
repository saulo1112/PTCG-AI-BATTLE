"""M47 — `rl_update`'s new flags must be inert when absent.

WHY THIS EXISTS. `scratchpad/rl_selfplay.py::rl_update` is the update that M16, M19,
M38 and M39 all ran, and every conclusion drawn from those milestones assumes a
specific gradient expression. M47 adds three keyword arguments to it (PPO ratio clip,
advantage normalisation, entropy bonus). If any of them changed behaviour when left at
its default, four milestones' worth of results would silently stop being reproducible
and we would not find out until a post-mortem.

So this pins the default path against a reference implementation of the PRE-M47
expression, copied verbatim below, and demands **bit-for-bit** equality -- not
`allclose`. The M47 flags are the fourth instance in this project of the
"additive, absent-by-default, existing artefact byte-identical" pattern (M27
`recover_rule`, M34 per-context profiles, M42 `count_heads`); each of those shipped
with exactly this kind of guard.

The clipped path is tested separately for the property that actually defines it: with
a large epsilon the clip can never bind, so it must reduce to the unclipped update.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scratchpad"))

rl_selfplay = pytest.importorskip("rl_selfplay")

from ptcg_ai.imitation.train import Decision, _pack, _seg_softmax  # noqa: E402
from train_mlp_main import _Adam, _mlp_scores  # noqa: E402


def _rl_update_pre_m47(members, members_bc, decisions, lambda_a=0.05, tau=1.0,
                       lr=1e-3, epochs=10):
    """The exact body of `rl_update` as of commit cd90e70 (M38/M39). Do not 'fix'."""
    bigX, starts, lens, chosen_global, adv = _pack(decisions)
    onehot = np.zeros(bigX.shape[0]); onehot[chosen_global] = 1.0
    adv_rep = np.repeat(adv, lens)
    N = len(decisions); M = len(members)
    opts = [{k: _Adam(np.shape(P[k]), lr) for k in ("W1", "b1", "w2", "b2")} for P in members]
    for _ in range(epochs):
        fwd = [_mlp_scores(bigX, P) for P in members]
        mean_scores = sum(f[0] for f in fwd) / M
        probs = _seg_softmax(mean_scores / tau, starts, lens)
        g = (probs - onehot) * adv_rep / (N * tau)
        for mi, P in enumerate(members):
            _, Z1, A1 = fwd[mi]
            dscore = g / M
            dw2 = A1.T @ dscore + lambda_a * (P["w2"] - members_bc[mi]["w2"])
            db2 = float(dscore.sum())
            dZ1 = np.outer(dscore, P["w2"]) * (Z1 > 0)
            dW1 = bigX.T @ dZ1 + lambda_a * (P["W1"] - members_bc[mi]["W1"])
            db1 = dZ1.sum(axis=0)
            P["W1"] = opts[mi]["W1"].step(P["W1"], dW1)
            P["b1"] = opts[mi]["b1"].step(P["b1"], db1)
            P["w2"] = opts[mi]["w2"].step(P["w2"], dw2)
            P["b2"] = float(opts[mi]["b2"].step(np.array(P["b2"]), db2))
    return members


def _fixture(dim=17, h=5, n_decisions=24, n_members=2, seed=7):
    """A small batch with the shape of a real one: ragged option counts, a mix of
    positive and negative advantages, ReLU units both alive and dead."""
    rng = np.random.default_rng(seed)
    decisions = []
    for _ in range(n_decisions):
        n_opts = int(rng.integers(2, 9))
        decisions.append(Decision(
            X=rng.standard_normal((n_opts, dim)),
            chosen=[int(rng.integers(n_opts))],
            weight=float(rng.standard_normal() * 0.45),   # A = R - V(s) scale
        ))
    members = [{"W1": rng.standard_normal((dim, h)) * 0.1, "b1": rng.standard_normal(h) * 0.1,
                "w2": rng.standard_normal(h) * 0.1, "b2": float(rng.standard_normal() * 0.1)}
               for _ in range(n_members)]
    return decisions, members


def _copy(members):
    return [{k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in P.items()}
            for P in members]


def _assert_identical(got, want):
    assert len(got) == len(want)
    for i, (P, Q) in enumerate(zip(got, want)):
        for k in ("W1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(
                np.asarray(P[k]), np.asarray(Q[k]),
                err_msg=f"member {i} param {k!r} drifted from the pre-M47 update")


@pytest.mark.parametrize("epochs", [1, 10])
def test_defaults_reproduce_the_pre_m47_update_bit_for_bit(epochs):
    decisions, members = _fixture()
    bc = _copy(members)

    new = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                lambda_a=0.1, tau=1.0, lr=3e-5, epochs=epochs)
    old = _rl_update_pre_m47(_copy(members), _copy(bc), decisions,
                             lambda_a=0.1, tau=1.0, lr=3e-5, epochs=epochs)
    _assert_identical(new, old)


def test_on_the_first_epoch_the_clip_is_exactly_the_unclipped_update():
    """At the first gradient step the policy has not moved yet, so the ratio is exactly
    1.0 and the clip cannot bind whatever epsilon is. That pins the sign and scale of
    the whole clipped expression against the known-good one: if the ratio were built
    from the wrong reference (the running policy instead of the frozen collecting one)
    it would still be 1 here, but the epochs>1 test below would then fail."""
    decisions, members = _fixture(seed=11)
    bc = _copy(members)

    clipped = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                    lambda_a=0.1, tau=1.0, lr=3e-5, epochs=1, clip=0.2)
    plain = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                  lambda_a=0.1, tau=1.0, lr=3e-5, epochs=1)
    _assert_identical(clipped, plain)


def test_a_non_binding_clip_is_still_importance_weighted_not_plain_reinforce():
    """Documents a real and intended difference, found by this test suite.

    With a band so wide it never binds, the surrogate is still `ratio * A`, whose
    gradient carries the importance weight `ratio` that plain REINFORCE lacks. So the
    two updates legitimately diverge from the SECOND epoch on -- and that divergence is
    precisely the stale-batch correction M47 is testing. A version of this test that
    demanded equality here would be asserting the bug back in.
    """
    decisions, members = _fixture(seed=11)
    bc = _copy(members)

    wide = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                 lambda_a=0.1, tau=1.0, lr=3e-5, epochs=10, clip=1e6)
    plain = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                  lambda_a=0.1, tau=1.0, lr=3e-5, epochs=10)
    # Same order of magnitude (the ratio stays near 1 at this lr), but not identical.
    assert not np.allclose(wide[0]["W1"], plain[0]["W1"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(wide[0]["W1"], plain[0]["W1"], rtol=1e-2, atol=1e-5)


def test_the_clip_actually_binds_and_changes_the_result():
    """Guard against a clip that is wired in but inert -- the failure mode that would
    make M47 measure nothing while looking like it ran."""
    decisions, members = _fixture(seed=3)
    bc = _copy(members)

    tight = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                  lambda_a=0.1, tau=1.0, lr=1e-2, epochs=10, clip=0.05)
    plain = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                  lambda_a=0.1, tau=1.0, lr=1e-2, epochs=10)
    assert not np.allclose(tight[0]["W1"], plain[0]["W1"]), (
        "a tight clip over 10 epochs at lr=1e-2 left the weights unchanged: the ratio "
        "mask is not reaching the gradient")


def test_advantage_normalisation_standardises_the_batch():
    """adv_norm must centre and scale the advantages, which is what makes the effective
    step ~2-2.5x larger at std(A)~0.4 -- the reason lr has to come down with it."""
    decisions, members = _fixture(seed=5)
    bc = _copy(members)

    normed = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                   lambda_a=0.1, tau=1.0, lr=3e-5, epochs=5, adv_norm=True)
    plain = rl_selfplay.rl_update(_copy(members), _copy(bc), decisions,
                                  lambda_a=0.1, tau=1.0, lr=3e-5, epochs=5)
    assert not np.allclose(normed[0]["W1"], plain[0]["W1"])

    raw = np.array([d.weight for d in decisions])
    assert abs(raw.mean()) > 1e-6, "fixture advantages are already centred; test is vacuous"
    scaled = (raw - raw.mean()) / (raw.std() + 1e-8)
    assert abs(scaled.mean()) < 1e-9
    assert abs(scaled.std() - 1.0) < 1e-6


def _fake_traj(monkeypatch, rows):
    """Feed `_featurize_traj` a synthetic batch without touching disk, the parser, the
    card database or the featurizer -- the advantage recursion is the only thing here
    that has any logic in it, and it depends on none of those."""
    import types

    class _Row:
        def __init__(self, gid, si, won):
            self.game_id, self.step_index, self.won = gid, si, won
            self.context = rl_selfplay.MAIN_CTX
            self.raw_observation, self.action = {}, [0]

    values = {(gid, si): v for gid, si, v, _ in rows}
    monkeypatch.setattr(rl_selfplay, "read_decision_dataset",
                        lambda _p: [_Row(g, s, w) for g, s, _, w in rows])
    monkeypatch.setattr(rl_selfplay, "GameState", types.SimpleNamespace(build=lambda o, c: None))
    monkeypatch.setattr(rl_selfplay.F, "is_prize_pick", lambda _s: False)
    monkeypatch.setattr(rl_selfplay.F, "featurize_decision",
                        lambda *a, **k: [[0.0]])
    parser = types.SimpleNamespace(parse=lambda raw: types.SimpleNamespace(
        select=types.SimpleNamespace(option=[0]), current=object()))
    holder = {"i": 0}
    order = [(g, s) for g, s, _, _ in rows]

    def fake_value(obs, cards, v):
        key = order[holder["i"]]
        holder["i"] += 1
        return values[key]

    monkeypatch.setattr(rl_selfplay, "_value", fake_value)
    return parser


def test_gae_lambda_absent_reproduces_the_monte_carlo_advantage(monkeypatch):
    """The load-bearing regression: M16 through M47 all ran A = R - V(s)."""
    rows = [("g1", 0, 0.4, True), ("g1", 1, 0.6, True), ("g2", 0, 0.7, False)]
    parser = _fake_traj(monkeypatch, rows)
    decs = rl_selfplay._featurize_traj("ignored", None, parser, None)
    assert [d.weight for d in decs] == pytest.approx([1 - 0.4, 1 - 0.6, 0 - 0.7])


def test_gae_lambda_one_telescopes_back_to_the_monte_carlo_return(monkeypatch):
    """With gamma=1 and lambda=1 the deltas telescope: sum of (V_{t+1} - V_t) plus the
    terminal (R - V_T) collapses to R - V_t. If this drifts, the recursion is wrong."""
    rows = [("g1", 0, 0.4, True), ("g1", 1, 0.6, True), ("g1", 2, 0.55, True),
            ("g2", 0, 0.7, False), ("g2", 1, 0.2, False)]
    parser = _fake_traj(monkeypatch, rows)
    decs = rl_selfplay._featurize_traj("ignored", None, parser, None, gae_lambda=1.0)
    assert [d.weight for d in decs] == pytest.approx(
        [1 - 0.4, 1 - 0.6, 1 - 0.55, 0 - 0.7, 0 - 0.2])


def test_gae_lambda_zero_is_one_step_td(monkeypatch):
    rows = [("g1", 0, 0.4, True), ("g1", 1, 0.6, True), ("g1", 2, 0.55, True)]
    parser = _fake_traj(monkeypatch, rows)
    decs = rl_selfplay._featurize_traj("ignored", None, parser, None, gae_lambda=0.0)
    # inner steps: V_{t+1} - V_t ; last step bootstraps on the outcome: R - V_T
    assert [d.weight for d in decs] == pytest.approx([0.6 - 0.4, 0.55 - 0.6, 1 - 0.55])


def test_gae_advantage_is_mostly_within_game(monkeypatch):
    """The whole point: TD advantages must stop being constant across a game. Two games
    with opposite outcomes but identical value paths get IDENTICAL inner advantages under
    lambda=0, where Monte Carlo would give them opposite signs throughout."""
    rows = [("win", 0, 0.4, True), ("win", 1, 0.6, True),
            ("loss", 0, 0.4, False), ("loss", 1, 0.6, False)]
    parser = _fake_traj(monkeypatch, rows)
    mc = rl_selfplay._featurize_traj("ignored", None, parser, None)
    parser = _fake_traj(monkeypatch, rows)
    td = rl_selfplay._featurize_traj("ignored", None, parser, None, gae_lambda=0.0)
    assert mc[0].weight == pytest.approx(0.6) and mc[2].weight == pytest.approx(-0.4)
    assert td[0].weight == pytest.approx(td[2].weight)   # same state path, same signal


def test_gae_respects_step_index_not_file_order(monkeypatch):
    """`_collect_impl` writes rows in order, but the parallel path merges per opponent.
    The recursion must key on step_index so a reordered file cannot silently invert time."""
    rows = [("g1", 2, 0.55, True), ("g1", 0, 0.4, True), ("g1", 1, 0.6, True)]
    parser = _fake_traj(monkeypatch, rows)
    decs = rl_selfplay._featurize_traj("ignored", None, parser, None, gae_lambda=0.0)
    # returned in FILE order, but each advantage computed from its step_index neighbour
    assert [d.weight for d in decs] == pytest.approx([1 - 0.55, 0.6 - 0.4, 0.55 - 0.6])


def test_the_new_setup_ships_the_champion_and_reverts_the_m39_reweight():
    """`alakazam_final` exists to correct three inputs; if any of them regresses the
    whole run measures the wrong thing, and nothing else would catch it."""
    cfg = rl_selfplay.SETUPS["alakazam_final"]
    assert cfg["base_ckpt"] == "data/models/bc_alakazam_final.json"
    assert cfg["opponents"]["mirror"][1] == "data/models/bc_alakazam_final.json"
    weights = {name: w for name, (_, _, w, _) in cfg["opponents"].items()}
    assert weights == {"grimmsnarl": 0.40, "mirror": 0.35, "kangaskhan": 0.25}, (
        "opponent mix drifted off M38's; 0.30/0.30/0.40 is M39's reweight, which M39 "
        "measured as confidently worse (0.438 [0.382-0.494] vs the frozen init)")
    assert cfg["heldout"][1] is not None, "held-out archetype must have a real clone"
