"""Pure-stdlib forward pass for the M37 two-stream set transformer.

The Kaggle runtime has no numpy (ADR-0014), so the GPU-trained model has to be
re-implemented here by hand: layernorm, multi-head attention and the FFN, in plain
Python. Everything in this module must stay importable with the standard library
alone — `tests/unit/test_imitation_ship_safety.py` enforces it.

WHY A SEPARATE ENTRY POINT. `policy._score(spec, x) -> float` takes ONE option vector.
A set model cannot be expressed that way: every option's score depends on the other
options. `score_set(spec, X) -> list[float]` takes the whole option matrix, which
`features.featurize_decision` already produces.

THE MODEL (mirrors scratchpad/colab_train_set.py::SetTransformerV2 exactly):

    X [N, 658]
      state = X[0][69:103]          identical for every option — VERIFIED 100% on 3000
                                    real decisions, so reading row 0 is safe
      opt_i = X[i][0:69]            the only part that varies per option
      z     = W_s2 @ relu(W_s0 @ state + b_s0) + b_s2
      g, b  = split(W_film @ z + b_film)
      h_i   = (W_opt @ opt_i + b_opt) * (1 + g) + b        <- FiLM, same for all options
      L x { h += attn(LN1(h));  h += FFN(LN2(h)) }          pre-LN
      s_i   = W_head @ LN_out(h_i) + b_head

There is no attention mask: at inference every option is real, so unlike training
(which pads ragged decisions into a batch) there is nothing to mask out. That is also
why the trainer's `torch.nan_to_num` has no counterpart here.

NO POSITIONAL ENCODING — deliberately. Two copies of the same card produce identical
feature vectors, and attention without positions is permutation-equivariant, so they
necessarily receive identical scores. Adding positions would let the model fit noise
choosing between indistinguishable cards. `tests/unit/test_setnet.py` pins this.

PERFORMANCE. This is the hot path: ~49 MAIN decisions/game x ~11 options, and every
option costs ~600k multiply-adds per ensemble member. Two things keep it tractable and
both matter more than they look:
  * `prepare_spec` splits each weight matrix into per-output-row lists ONCE at load, so
    the inner loop never re-slices a flat list (measured 1.18x);
  * every matvec and dot is `sum(map(mul, ...))`, which runs the loop in C rather than
    in the interpreter. The attention value-mixing is transposed per head for the same
    reason — accumulating per (query, key) pair in Python costs O(N^2 * d) interpreted
    steps, while transposing lets each output channel be one C-level `sum(map(...))`.
"""

from __future__ import annotations

import math
from operator import mul
from typing import Any, Mapping, Sequence

#: PyTorch's nn.LayerNorm default. Must match or parity drifts in the 4th decimal.
_LN_EPS = 1e-5


def _rows(flat: Sequence[float], n_out: int, n_in: int) -> list[list[float]]:
    return [list(flat[o * n_in:(o + 1) * n_in]) for o in range(n_out)]


def _matvec(rows: Sequence[Sequence[float]], b: Sequence[float],
            x: Sequence[float]) -> list[float]:
    """y = W @ x + b, with W already split into rows by `prepare_spec`."""
    return [sum(map(mul, r, x)) + bi for r, bi in zip(rows, b)]


def _layernorm(x: Sequence[float], w: Sequence[float], b: Sequence[float]) -> list[float]:
    """PyTorch LayerNorm: BIASED variance (divide by n, not n-1)."""
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + _LN_EPS)
    return [(v - mean) * inv * wi + bi for v, wi, bi in zip(x, w, b)]


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(v - m) for v in xs]
    s = sum(ex)
    return [v / s for v in ex]


def prepare_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Flat payload -> row-split structure, done once at policy construction.

    Returns a NEW dict; the payload is left untouched so `load_weights` output stays
    serialisable and comparable.
    """
    d = int(spec["d_model"])
    layers = int(spec["layers"])
    opt_end = int(spec["opt_end"])
    st = int(spec["state_end"]) - int(spec["state_start"])
    out: dict[str, Any] = {
        "d_model": d,
        "layers": layers,
        "heads": int(spec["heads"]),
        "opt_end": opt_end,
        "state_start": int(spec["state_start"]),
        "state_end": int(spec["state_end"]),
        "members": [],
    }
    for m in spec["members"]:
        pm = {
            "opt_w": _rows(m["opt_w"], d, opt_end), "opt_b": list(m["opt_b"]),
            "state0_w": _rows(m["state0_w"], d, st), "state0_b": list(m["state0_b"]),
            "state2_w": _rows(m["state2_w"], d, d), "state2_b": list(m["state2_b"]),
            "film_w": _rows(m["film_w"], 2 * d, d), "film_b": list(m["film_b"]),
            "lnout_w": list(m["lnout_w"]), "lnout_b": list(m["lnout_b"]),
            "head_w": list(m["head_w"]), "head_b": float(m["head_b"]),
            "blocks": [],
        }
        for blk in m["blocks"]:
            pm["blocks"].append({
                "ln1_w": list(blk["ln1_w"]), "ln1_b": list(blk["ln1_b"]),
                "ln2_w": list(blk["ln2_w"]), "ln2_b": list(blk["ln2_b"]),
                "in_proj_w": _rows(blk["in_proj_weight"], 3 * d, d),
                "in_proj_b": list(blk["in_proj_bias"]),
                "out_proj_w": _rows(blk["out_proj_weight"], d, d),
                "out_proj_b": list(blk["out_proj_bias"]),
                "ff0_w": _rows(blk["ff0_w"], 4 * d, d), "ff0_b": list(blk["ff0_b"]),
                "ff2_w": _rows(blk["ff2_w"], d, 4 * d), "ff2_b": list(blk["ff2_b"]),
            })
        out["members"].append(pm)
    return out


def _attention(hs: list[list[float]], blk: Mapping[str, Any], d: int, heads: int) -> list[list[float]]:
    """Multi-head self-attention over the option set, matching nn.MultiheadAttention.

    `in_proj_weight` is [3d, d]; q/k/v are its three row-blocks, exactly how torch
    slices the projection with `.chunk(3, dim=-1)`.
    """
    dh = d // heads
    scale = 1.0 / math.sqrt(dh)
    ipw, ipb = blk["in_proj_w"], blk["in_proj_b"]
    n = len(hs)

    qkv = [_matvec(ipw, ipb, h) for h in hs]
    ctx = [[0.0] * d for _ in range(n)]
    for hd in range(heads):
        lo, hi = hd * dh, (hd + 1) * dh
        qs = [v[lo:hi] for v in qkv]
        ks = [v[d + lo:d + hi] for v in qkv]
        # transpose the head's values once: each output channel then costs a single
        # C-level sum(map(mul, ...)) instead of an interpreted per-key accumulation
        vT = list(zip(*(v[2 * d + lo:2 * d + hi] for v in qkv)))
        for i in range(n):
            qi = qs[i]
            attn = _softmax([sum(map(mul, qi, kj)) * scale for kj in ks])
            row = ctx[i]
            for t in range(dh):
                row[lo + t] = sum(map(mul, attn, vT[t]))
    return [_matvec(blk["out_proj_w"], blk["out_proj_b"], c) for c in ctx]


def _forward_member(m: Mapping[str, Any], X: Sequence[Sequence[float]],
                    d: int, heads: int, opt_end: int,
                    st_lo: int, st_hi: int) -> list[float]:
    n = len(X)
    # --- state stream: encoded ONCE per decision, not once per option -------------
    state = list(X[0][st_lo:st_hi])
    z = _matvec(m["state0_w"], m["state0_b"], state)
    z = [v if v > 0.0 else 0.0 for v in z]
    z = _matvec(m["state2_w"], m["state2_b"], z)
    film = _matvec(m["film_w"], m["film_b"], z)
    gamma, beta = film[:d], film[d:]
    scale = [1.0 + g for g in gamma]

    # --- option stream + FiLM (identical modulation -> permutation-equivariant) ---
    ow, ob = m["opt_w"], m["opt_b"]
    hs = []
    for i in range(n):
        h = _matvec(ow, ob, X[i][:opt_end])
        hs.append([hv * s + b for hv, s, b in zip(h, scale, beta)])

    for blk in m["blocks"]:
        ln1w, ln1b = blk["ln1_w"], blk["ln1_b"]
        att = _attention([_layernorm(h, ln1w, ln1b) for h in hs], blk, d, heads)
        ln2w, ln2b = blk["ln2_w"], blk["ln2_b"]
        f0w, f0b, f2w, f2b = blk["ff0_w"], blk["ff0_b"], blk["ff2_w"], blk["ff2_b"]
        for i in range(n):
            hi = [a + b for a, b in zip(hs[i], att[i])]
            y = _layernorm(hi, ln2w, ln2b)
            y = _matvec(f0w, f0b, y)
            y = [v if v > 0.0 else 0.0 for v in y]
            y = _matvec(f2w, f2b, y)
            hs[i] = [a + b for a, b in zip(hi, y)]

    hw, hb = m["head_w"], m["head_b"]
    lw, lb = m["lnout_w"], m["lnout_b"]
    return [sum(map(mul, hw, _layernorm(h, lw, lb))) + hb for h in hs]


def score_set(spec: Mapping[str, Any], X: Sequence[Sequence[float]],
              k: int | None = None) -> list[float]:
    """Score every option of one decision. Ensemble = MEAN, matching policy._score.

    `spec` must already have gone through `prepare_spec`. `k` caps how many members are
    evaluated — the time guard in `ImitationPolicy` uses it to shed cost mid-game
    without swapping models.
    """
    d, heads = spec["d_model"], spec["heads"]
    opt_end, st_lo, st_hi = spec["opt_end"], spec["state_start"], spec["state_end"]
    members = spec["members"]
    if k is not None:
        members = members[:max(1, min(k, len(members)))]
    n = len(X)
    total = [0.0] * n
    for m in members:
        s = _forward_member(m, X, d, heads, opt_end, st_lo, st_hi)
        for i in range(n):
            total[i] += s[i]
    inv = 1.0 / len(members)
    return [v * inv for v in total]


def validate_spec(ctx: str, spec: Mapping[str, Any], dim: int) -> None:
    """Shape-check a RAW (flat, pre-`prepare_spec`) setxf2 spec.

    Raises ValueError so `load_weights` fails loudly at agent construction rather than
    letting `policy._decide` swallow it and silently play greedy for a whole ladder run.
    """
    d = int(spec.get("d_model", 0))
    layers = int(spec.get("layers", 0))
    heads = int(spec.get("heads", 0))
    opt_end = int(spec.get("opt_end", 0))
    st_lo, st_hi = int(spec.get("state_start", 0)), int(spec.get("state_end", 0))
    if d <= 0 or layers <= 0 or heads <= 0 or d % heads:
        raise ValueError(f"context {ctx!r} setxf2 bad d_model/layers/heads")
    if not (0 < opt_end <= st_lo < st_hi <= dim):
        raise ValueError(
            f"context {ctx!r} setxf2 slice bounds {opt_end}/{st_lo}/{st_hi} "
            f"inconsistent with profile dim {dim}")
    members = spec.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"context {ctx!r} setxf2 has no members")
    st = st_hi - st_lo
    for mi, m in enumerate(members):
        exp = {
            "opt_w": d * opt_end, "opt_b": d,
            "state0_w": d * st, "state0_b": d,
            "state2_w": d * d, "state2_b": d,
            "film_w": 2 * d * d, "film_b": 2 * d,
            "lnout_w": d, "lnout_b": d, "head_w": d,
        }
        for key, want in exp.items():
            got = len(m.get(key, ()))
            if got != want:
                raise ValueError(
                    f"context {ctx!r} setxf2 member {mi} {key}: {got} != {want}")
        if not isinstance(m.get("head_b"), (int, float)):
            raise ValueError(f"context {ctx!r} setxf2 member {mi} head_b must be a scalar")
        if len(m.get("blocks", ())) != layers:
            raise ValueError(f"context {ctx!r} setxf2 member {mi} block count != {layers}")
        for bi, blk in enumerate(m["blocks"]):
            bexp = {
                "ln1_w": d, "ln1_b": d, "ln2_w": d, "ln2_b": d,
                "in_proj_weight": 3 * d * d, "in_proj_bias": 3 * d,
                "out_proj_weight": d * d, "out_proj_bias": d,
                "ff0_w": 4 * d * d, "ff0_b": 4 * d,
                "ff2_w": d * 4 * d, "ff2_b": d,
            }
            for key, want in bexp.items():
                got = len(blk.get(key, ()))
                if got != want:
                    raise ValueError(
                        f"context {ctx!r} setxf2 member {mi} block {bi} {key}: "
                        f"{got} != {want}")
