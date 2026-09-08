"""M37: the pure-stdlib set transformer must match torch exactly, and must keep the
duplicate-card symmetry that justifies having no positional encoding.

TWO LAYERS OF CHECK, on purpose:

* ``test_stdlib_matches_torch`` proves the kernel is CORRECT, against
  `nn.MultiheadAttention` — the module the model was actually trained with, not a numpy
  transcription of `setnet` (which would only prove the code agrees with itself). The
  reference runs in float64: in float32 the residual drift through 3 LayerNorms and a
  softmax lands near 1e-6 and a 1e-9 bar would be meaningless. It needs torch, which is
  NOT in the project venv, so it skips under `uv run`.
* ``test_golden_scores`` pins the exact outputs that the torch check validated, in plain
  stdlib. It runs everywhere. Without it a kernel regression would sail through CI,
  because the correctness test is the one that gets skipped.

Everything else here is torch-free by construction.
"""

from __future__ import annotations

import math
import random

import pytest

from ptcg_ai.imitation import setnet

DIM, OPT_END, ST_LO, ST_HI = 658, 69, 69, 103


def _random_spec(seed: int, d: int = 32, layers: int = 2, heads: int = 4) -> dict:
    """A structurally valid flat spec with reproducible pseudo-random weights."""
    rng = random.Random(seed)
    r = lambda n, s=0.3: [rng.gauss(0, s) for _ in range(n)]  # noqa: E731
    member = {
        "opt_w": r(d * OPT_END), "opt_b": r(d),
        "state0_w": r(d * (ST_HI - ST_LO)), "state0_b": r(d),
        "state2_w": r(d * d), "state2_b": r(d),
        "film_w": r(2 * d * d), "film_b": r(2 * d),
        "lnout_w": r(d), "lnout_b": r(d),
        "head_w": r(d), "head_b": rng.gauss(0, 0.3),
        "blocks": [{
            "ln1_w": r(d), "ln1_b": r(d), "ln2_w": r(d), "ln2_b": r(d),
            "in_proj_weight": r(3 * d * d), "in_proj_bias": r(3 * d),
            "out_proj_weight": r(d * d), "out_proj_bias": r(d),
            "ff0_w": r(4 * d * d), "ff0_b": r(4 * d),
            "ff2_w": r(d * 4 * d), "ff2_b": r(d),
        } for _ in range(layers)],
    }
    return {"kind": "setxf2_ensemble", "d_model": d, "layers": layers, "heads": heads,
            "opt_end": OPT_END, "state_start": ST_LO, "state_end": ST_HI,
            "members": [member]}


def _prepared(seed: int, **kw) -> dict:
    spec = _random_spec(seed, **kw)
    setnet.validate_spec("MAIN", spec, DIM)
    return setnet.prepare_spec(spec)


def _options(seed: int, n: int) -> list[list[float]]:
    """Sparse rows shaped like the real featurizer, with a SHARED state block — the
    invariant `setnet` relies on when it reads the state from row 0."""
    rng = random.Random(seed)
    state = [rng.gauss(0, 1) for _ in range(ST_HI - ST_LO)]
    rows = []
    for _ in range(n):
        row = [0.0] * DIM
        for j in rng.sample(range(OPT_END), 8):
            row[j] = rng.gauss(0, 1)
        row[ST_LO:ST_HI] = state
        for j in rng.sample(range(ST_HI, DIM), 20):
            row[j] = rng.gauss(0, 1)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------
# correctness (needs torch)
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 11, 25])
def test_stdlib_matches_torch(n: int) -> None:
    torch = pytest.importorskip("torch", reason="torch is not in the project venv")
    nn = torch.nn
    d, layers, heads = 32, 2, 4
    spec = _random_spec(0, d, layers, heads)
    m = spec["members"][0]

    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.opt_proj = nn.Linear(OPT_END, d)
            self.state_enc = nn.Sequential(
                nn.Linear(ST_HI - ST_LO, d), nn.ReLU(), nn.Linear(d, d))
            self.film = nn.Linear(d, 2 * d)
            self.blocks = nn.ModuleList(nn.ModuleDict({
                "ln1": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, heads, batch_first=True),
                "ln2": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d)),
            }) for _ in range(layers))
            self.ln_out = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            gamma, beta = self.film(self.state_enc(x[:, 0, ST_LO:ST_HI])).chunk(2, dim=-1)
            h = self.opt_proj(x[..., :OPT_END]) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            for b in self.blocks:
                q = b["ln1"](h)
                h = h + b["attn"](q, q, q, need_weights=False)[0]
                h = h + b["ff"](b["ln2"](h))
            return self.head(self.ln_out(h)).squeeze(-1)

    ref = Ref().double()
    load = {
        "opt_proj.weight": m["opt_w"], "opt_proj.bias": m["opt_b"],
        "state_enc.0.weight": m["state0_w"], "state_enc.0.bias": m["state0_b"],
        "state_enc.2.weight": m["state2_w"], "state_enc.2.bias": m["state2_b"],
        "film.weight": m["film_w"], "film.bias": m["film_b"],
        "ln_out.weight": m["lnout_w"], "ln_out.bias": m["lnout_b"],
        "head.weight": m["head_w"], "head.bias": [m["head_b"]],
    }
    for i, blk in enumerate(m["blocks"]):
        load.update({
            f"blocks.{i}.ln1.weight": blk["ln1_w"], f"blocks.{i}.ln1.bias": blk["ln1_b"],
            f"blocks.{i}.ln2.weight": blk["ln2_w"], f"blocks.{i}.ln2.bias": blk["ln2_b"],
            f"blocks.{i}.attn.in_proj_weight": blk["in_proj_weight"],
            f"blocks.{i}.attn.in_proj_bias": blk["in_proj_bias"],
            f"blocks.{i}.attn.out_proj.weight": blk["out_proj_weight"],
            f"blocks.{i}.attn.out_proj.bias": blk["out_proj_bias"],
            f"blocks.{i}.ff.0.weight": blk["ff0_w"], f"blocks.{i}.ff.0.bias": blk["ff0_b"],
            f"blocks.{i}.ff.2.weight": blk["ff2_w"], f"blocks.{i}.ff.2.bias": blk["ff2_b"],
        })
    shapes = ref.state_dict()
    ref.load_state_dict({k: torch.tensor(v, dtype=torch.float64).reshape(shapes[k].shape)
                         for k, v in load.items()})
    ref.eval()

    X = _options(n, n)
    with torch.no_grad():
        want = ref(torch.tensor([X], dtype=torch.float64))[0].tolist()
    for a, b in zip(setnet.score_set(setnet.prepare_spec(spec), X), want):
        assert a == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------------------
# regression + invariants (stdlib only, always run)
# --------------------------------------------------------------------------------

#: Captured from the kernel at the commit where `test_stdlib_matches_torch` passed at
#: 1e-9. The two sizes here are deliberately BOTH in that test's parametrisation and
#: built from the same `_random_spec(0)` / `_options(n, n)` inputs, so every golden is a
#: number torch signed off on. Regenerate ONLY alongside a fresh torch parity run —
#: silently refreshing them would turn this guard into a rubber stamp.
_GOLDEN = {
    2: [-0.1563764536175914, 0.04156912243552535],
    11: [0.1408443999516797, -0.30171769777040697, -0.33733646088581304,
         -0.14578623782283268, -0.02584950466808905, -0.1866853390668336,
         -0.7250648395767783, 0.5076043811930642, -0.16792410277346004,
         -0.5764187231904014, -0.32427469013640714],
}


@pytest.mark.parametrize("n", sorted(_GOLDEN))
def test_golden_scores(n: int) -> None:
    """Pins the torch-validated numbers without needing torch. This is the check that
    actually runs in CI, so a broken attention kernel fails here or nowhere."""
    got = setnet.score_set(_prepared(0), _options(n, n))
    assert got == pytest.approx(_GOLDEN[n], abs=1e-12)


def test_duplicate_options_score_identically() -> None:
    """The property that justifies shipping NO positional encoding: two copies of the
    same card are byte-identical rows, so they must score identically. If this ever
    fails, the model is choosing between indistinguishable options on noise."""
    X = _options(7, 5)
    X[3] = list(X[1])
    scores = setnet.score_set(_prepared(0), X)
    assert scores[1] == scores[3], "duplicates must be exactly equal, not merely close"


def test_permutation_equivariance() -> None:
    """Reordering the options permutes the scores; it must not change them."""
    prepared = _prepared(1)
    X = _options(11, 6)
    base = setnet.score_set(prepared, X)
    order = [4, 0, 5, 2, 1, 3]
    shuffled = setnet.score_set(prepared, [X[i] for i in order])
    for pos, src in enumerate(order):
        assert shuffled[pos] == pytest.approx(base[src], abs=1e-12)


def test_single_option_is_finite() -> None:
    """N=1 makes the attention softmax degenerate to 1.0. Cheap, but this is the shape
    that would produce NaNs if the softmax were written wrong, and it is common in
    real play (a forced MAIN action)."""
    scores = setnet.score_set(_prepared(0), _options(1, 1))
    assert len(scores) == 1 and math.isfinite(scores[0])


def test_k_caps_ensemble_members() -> None:
    """The time guard sheds cost by evaluating fewer members: k=1 must equal the first
    member alone, and k past the ensemble size must clamp instead of crashing."""
    one, two = _prepared(0), _prepared(1)
    combined = dict(one, members=one["members"] + two["members"])
    X = _options(3, 4)
    assert setnet.score_set(combined, X, k=1) == pytest.approx(
        setnet.score_set(one, X), abs=1e-12)
    assert setnet.score_set(combined, X, k=99) == pytest.approx(
        setnet.score_set(combined, X), abs=1e-12)


def test_validate_spec_rejects_wrong_shapes() -> None:
    spec = _random_spec(0)
    setnet.validate_spec("MAIN", spec, DIM)                  # sane baseline

    bad = dict(spec, members=[{**spec["members"][0], "opt_b": [0.0] * 7}])
    with pytest.raises(ValueError, match="opt_b"):
        setnet.validate_spec("MAIN", bad, DIM)

    bad = dict(spec, members=[{**spec["members"][0], "head_b": [0.0]}])
    with pytest.raises(ValueError, match="head_b"):
        setnet.validate_spec("MAIN", bad, DIM)

    blocks = spec["members"][0]["blocks"]
    bad = dict(spec, members=[{**spec["members"][0],
                               "blocks": [{**blocks[0], "ff0_b": [0.0]}] + blocks[1:]}])
    with pytest.raises(ValueError, match="ff0_b"):
        setnet.validate_spec("MAIN", bad, DIM)

    with pytest.raises(ValueError, match="slice bounds"):
        setnet.validate_spec("MAIN", dict(spec, state_end=DIM + 1), DIM)

    with pytest.raises(ValueError, match="d_model"):
        setnet.validate_spec("MAIN", dict(spec, heads=5), DIM)   # 32 % 5 != 0
