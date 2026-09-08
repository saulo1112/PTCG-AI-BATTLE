"""M37 — architecture sweep on Colab GPU: deep MLP / DeepSets / set transformer.

WHY. The shipped scorer is `658 -> 48 -> 1` (ONE hidden layer) and, more importantly,
`policy._score(spec, x) -> float` scores **every option in total isolation** before the
softmax. "This card is the best of the ones I have" is inexpressible at ANY depth.
M31 closed WIDTH (h=48->96 = +0.0056, inside seed noise); nobody ever tested depth or
cross-option context. This sweep tests both, on GPU, in one session.

ARMS (all share the split, the seeds and the loss; only the model differs):
  mlp       h=48, one hidden layer          — the control, reproduces the shipped arch
  deep      658 -> 256 -> 128 -> 1          — depth, still per-option (no cross-option)
  deepsets  per-option encoder + masked mean/max pooled context concatenated back
  setxf     d=64, 2 layers, all 658 dims per token  (the naive version)
  setxf2    TWO-STREAM: state encoded once + FiLM + attention. See its docstring —
            only 22.7% of the 658 dims vary across options and the snapshot block is
            identical 100% of the time, so `setxf` wastes most of d_model re-encoding
            the same board state N times.

SHIPPING COST is what breaks ties, and it is NOT uniform:
  ALAKAZAM_MEM (a PROFILE change, run as `--arm mlp --profile ALAKAZAM_MEM`)
      ... free: longer feature vector, existing `_mlp_forward` unchanged.
  deep ... cheap: a loop over layers in `_mlp_forward`.
  deepsets / setxf / setxf2
      ... expensive: all need a DECISION-LEVEL scorer (today `_score(spec, x)` takes ONE
      option), so `policy.py`'s two call sites must change. deepsets then needs ~100
      lines of stdlib kernel (matmul/relu/mean/max); the transformers need ~300-500
      (multi-head attention, layernorm, softmax).

INFERENCE BUDGET — measured, not guessed: pure-stdlib Python does **11.7 M MAC/s**
(timed on a 128x512 dense layer, the same nested-list pattern `_mlp_forward` uses).
Against the 600 s/agent/episode budget at ~84 decisions and ~11 options per game:
      d=64  L=2  x3  =  25 s/game        d=128 L=3  x3  = 143 s/game
      d=64  L=2  x10 =  83 s/game        d=128 L=3  x5  = 238 s/game
      d=192 L=4  x3  = 424 s/game (no margin)
      d=128 L=3  x10 = 476 s/game (79% of budget — too tight, games vary a lot)
So member count and model size TRADE OFF. Do not inherit "3 members" from M31 (it
trimmed a different model for speed): read the `ensemble curve` this script prints and
pick the point from data.

>>> DESIGN RULE: NO POSITIONAL ENCODING. <<<
The options are a SET. Two copies of the same card produce byte-identical feature
vectors, and self-attention without positions is permutation-equivariant, so identical
inputs necessarily produce identical scores. Add positions and that symmetry breaks —
the model would start fitting pure noise to choose between identical cards. Same reason
`deepsets` pools with mean/max (symmetric) and never with anything order-dependent.
`--check-perm` asserts this invariant on real data before training.

COMPARABILITY — read before quoting any number from this script.

  **The baseline is the `mlp` arm of THIS sweep. Always run it first.** Do NOT compare
  against historical figures: M31's GPU 0.7831, the local 0.7685 and the champion's
  0.767 were each measured on a SMALLER held-out set. Yushin is still playing (the
  corpus went 1284 -> 1829 -> ~2300 games in four days), and every new game re-hashes
  into train/val/test, so the test set itself moves. Only a same-sweep, same-split,
  same-seed-count comparison means anything. Pass the mlp arm's TEST via `--baseline`
  to get the delta printed for you.

  All arms train with the SAME optimiser and minibatching, so `deep`/`deepsets`/`setxf`
  vs `mlp` isolates the architecture. (Minibatching is not optional here: a padded
  [n_dec, 49, 658] tensor for the full train split is ~6.6 GB. The shipped numpy recipe
  is full-batch, which is why the `mlp` arm must be re-run here rather than reused.)

Accuracy here is index-based argmax, which under-reports decks with duplicate cards
(R21, see scratchpad/semantic_fidelity.py). The bias is identical across arms so the
COMPARISON is valid; re-score the winner with the strict card-identity metric locally
before shipping.

Run on Colab. ONE variable at a time — mixing changes is how M32 burned two weeks.
  # 1. what did the LayerNorm bug cost? same config as the first sweep, fixed init:
  python colab_train_set.py --arm setxf  --seeds 10 --baseline 0.7561 --d-model 64 --layers 2
  # 2. the ambitious redesign (this is the one that answers "is the model too small?"):
  python colab_train_set.py --arm setxf2 --seeds 10 --baseline 0.7561 --d-model 128 --layers 3 --warmup 3
  # 3. do the 555 hand-crafted cross dims still earn their place?
  python colab_train_set.py --arm setxf2 --seeds 10 --baseline 0.7561 --d-model 128 --layers 3 --warmup 3 --use-cross
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from colab_train_mlp import DEV, build_cache
from ptcg_ai.imitation.deck_profiles import get_profile


# ------------------------------------------------------------------ data

class PaddedSplit:
    """Decisions as a padded [n_dec, max_opts, dim] view + a key-padding mask.

    Built from the same flat arrays `colab_train_mlp.build_cache` emits. The padded
    form is what a set model needs; we keep X flat on device and gather per minibatch
    so the full padded tensor is never materialised.
    """

    def __init__(self, X, lens, pick, won=None, arch=None):
        self.n_dec = len(lens)
        starts = np.zeros(self.n_dec, dtype=np.int64)
        np.cumsum(lens[:-1], out=starts[1:])
        m = int(lens.max())
        idx = np.full((self.n_dec, m), -1, dtype=np.int64)
        for i, (s, ln) in enumerate(zip(starts, lens)):
            idx[i, :ln] = np.arange(s, s + ln)
        self.X = torch.from_numpy(X).to(DEV)
        self.pad_idx = torch.from_numpy(np.clip(idx, 0, None)).to(DEV)
        self.pad_mask = torch.from_numpy(idx < 0).to(DEV)      # True where padding
        self.pick = torch.from_numpy(pick).to(DEV)
        self.won = torch.from_numpy(np.asarray(won, dtype=bool)).to(DEV) if won is not None else None
        self.arch = torch.from_numpy(np.asarray(arch, dtype=np.int8)).to(DEV) if arch is not None else None
        self.dim = X.shape[1]
        self.max_opts = m

    def batch(self, sel):
        """(options [b, L, dim], mask [b, L], target [b]) for decision indices `sel`."""
        return self.X[self.pad_idx[sel]], self.pad_mask[sel], self.pick[sel]


# ---------------------------------------------------------------- models

class PerOption(nn.Module):
    """Depth only — each option scored independently. Arms `mlp` and `deep`."""

    def __init__(self, dim: int, hidden: list[int]):
        super().__init__()
        layers, prev = [], dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask):
        return self.net(x).squeeze(-1)


class DeepSets(nn.Module):
    """Cross-option context via SYMMETRIC pooling — the cheap-to-ship arm.

    Masked mean and max over the option set are permutation-invariant, so the pooled
    summary is identical however the options are ordered and two identical options
    still receive identical scores.
    """

    def __init__(self, dim: int, d: int = 128, h: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask):
        e = self.enc(x)                                        # [b, L, d]
        keep = (~mask).unsqueeze(-1).float()
        mean = (e * keep).sum(1) / keep.sum(1).clamp(min=1.0)   # [b, d]
        mx = e.masked_fill(mask.unsqueeze(-1), float("-inf")).max(1).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        ctx = torch.cat([mean, mx], dim=-1).unsqueeze(1).expand(-1, e.shape[1], -1)
        return self.head(torch.cat([e, ctx], dim=-1)).squeeze(-1)


class SetTransformerV2(nn.Module):
    """Two-stream, state-conditioned set transformer. Arm `setxf2`.

    WHY THIS AND NOT `setxf`. Measured on 2,000 real MAIN decisions: only **149 of the
    658 dims (22.7%) vary across the options of a decision**, and the 29-dim snapshot
    block is byte-identical across options **100%** of the time. `setxf` projects all
    658 dims per token, so it re-encodes the same board state once per option and
    spends most of d_model representing content that cannot discriminate anything.

    It also eats the hand-crafted crosses. Layout (verified):
        A[0:12] B[12:35] C[35:44] D[44:63] E[63:69] | F[69:74] S[74:103]
        | A(x)S[103:451] B(x)R[451:658]
    A(x)S and B(x)R are 555 of the 658 dims (84%) and are pure second-order products —
    they exist because a LINEAR model cannot learn interactions. A transformer can, so
    feeding them is at best redundant. `--use-cross` keeps them so the question is
    testable rather than assumed.

    Design:
        state = X[0, 69:103]   opponent-active + snapshot, encoded ONCE per decision
        opt   = X[:, 0:69]     the genuinely per-option content
        FiLM(z) modulates every option IDENTICALLY, so permutation equivariance and the
        duplicate-card symmetry survive (the invariant `check_permutation_invariance`
        enforces). Still NO positional encoding, for the same reason.
    """

    OPT_END, STATE_START, STATE_END = 69, 69, 103

    def __init__(self, dim: int, d_model: int = 128, layers: int = 3, heads: int = 4,
                 ff: int = 4, use_cross: bool = False):
        super().__init__()
        self.use_cross = use_cross
        self.dim = dim
        self.opt_proj = nn.Linear(self.OPT_END, d_model)
        self.cross_proj = nn.Linear(dim - self.STATE_END, d_model) if use_cross else None
        self.state_enc = nn.Sequential(
            nn.Linear(self.STATE_END - self.STATE_START, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.film = nn.Linear(d_model, 2 * d_model)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, ff * d_model), nn.ReLU(),
                                    nn.Linear(ff * d_model, d_model)),
            }))
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        # the state block is constant across options; take it from the first REAL row
        first = (~mask).float().argmax(dim=1)                         # [b]
        state = x[torch.arange(x.shape[0], device=x.device), first,
                  self.STATE_START:self.STATE_END]                    # [b, 34]
        z = self.state_enc(state)                                     # [b, d]
        gamma, beta = self.film(z).chunk(2, dim=-1)                   # [b, d] each
        h = self.opt_proj(x[..., :self.OPT_END])
        if self.cross_proj is not None:
            h = h + self.cross_proj(x[..., self.STATE_END:])
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)        # same for all options
        for b in self.blocks:
            q = b["ln1"](h)
            a, _ = b["attn"](q, q, q, key_padding_mask=mask, need_weights=False)
            h = h + torch.nan_to_num(a)
            h = h + b["ff"](b["ln2"](h))
        return self.head(self.ln_out(h)).squeeze(-1)


class SetTransformer(nn.Module):
    """Self-attention over the option set. NO positional encoding (see module docstring)."""

    def __init__(self, dim: int, d_model: int = 64, layers: int = 2, heads: int = 4, ff: int = 4):
        super().__init__()
        self.proj = nn.Linear(dim, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, ff * d_model), nn.ReLU(),
                                    nn.Linear(ff * d_model, d_model)),
            }))
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        h = self.proj(x)
        for b in self.blocks:
            q = b["ln1"](h)
            a, _ = b["attn"](q, q, q, key_padding_mask=mask, need_weights=False)
            h = h + torch.nan_to_num(a)        # fully-masked rows can emit NaN
            h = h + b["ff"](b["ln2"](h))
        return self.head(self.ln_out(h)).squeeze(-1)


def build_model(arm: str, dim: int, args) -> nn.Module:
    if arm == "mlp":
        return PerOption(dim, [args.h])
    if arm == "deep":
        return PerOption(dim, [256, 128])
    if arm == "deepsets":
        return DeepSets(dim, d=args.d_model * 2, h=args.d_model * 2)
    if arm == "setxf":
        return SetTransformer(dim, d_model=args.d_model, layers=args.layers, heads=args.heads)
    if arm == "setxf2":
        return SetTransformerV2(dim, d_model=args.d_model, layers=args.layers,
                                heads=args.heads, use_cross=args.use_cross)
    raise ValueError(f"unknown arm {arm!r}")


# -------------------------------------------------------------- train/eval

def _scores(model, sp: PaddedSplit, sel):
    x, mask, tgt = sp.batch(sel)
    s = model(x, mask).masked_fill(mask, float("-inf"))
    return s, tgt


@torch.no_grad()
def evaluate(models, sp: PaddedSplit, bs: int, subset=None) -> float:
    """Ensemble accuracy = argmax of the MEAN score (matches policy._score)."""
    idx = torch.arange(sp.n_dec, device=DEV) if subset is None else subset
    if len(idx) == 0:
        return float("nan")
    ok = 0
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        acc = None
        for m in models:
            m.eval()
            s, tgt = _scores(m, sp, sel)
            acc = s if acc is None else acc + s
        ok += (acc.argmax(dim=1) == tgt).sum().item()
    return ok / len(idx)


def init_weights(model: nn.Module, seed: int) -> None:
    """Per-seed init. Type-aware — a blanket rule on p.dim() silently breaks LayerNorm.

    BUG FOUND 2026-08-02, after the first sweep. The original rule was
        if p.dim() > 1: xavier_uniform_(p) else: zeros_(p)
    and LayerNorm's `weight` is 1-D, so ALL FIVE LayerNorm gains were zeroed. A pre-LN
    block with gain 0 emits a constant, so the whole network returned **0.0 for every
    option at initialisation** (verified: variance across options = 0.000e+00). The set
    transformer started training dead and had to climb out through the LayerNorm
    gradients; `mlp`/`deep`/`deepsets` have no LayerNorm and were untouched. So the
    first sweep handicapped exactly one arm — the one under test — and its +0.0375 is a
    LOWER BOUND, not its real number.
    """
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)


def train_one(model, tr: PaddedSplit, va: PaddedSplit, args, seed: int):
    init_weights(model, seed)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(tr.n_dec / args.bs))
    warm = args.warmup * steps_per_epoch          # 0 disables
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_acc, best_state, wait, best_ep, step = -1.0, None, 0, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(tr.n_dec, generator=g).to(DEV)
        for i in range(0, tr.n_dec, args.bs):
            step += 1
            if warm and step <= warm:             # linear LR warmup
                for grp in opt.param_groups:
                    grp["lr"] = args.lr * step / warm
            sel = perm[i:i + args.bs]
            s, tgt = _scores(model, tr, sel)
            opt.zero_grad(set_to_none=True)
            Fn.cross_entropy(s, tgt).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = evaluate([model], va, args.eval_bs)
        if acc > best_acc + 1e-6:
            best_acc, wait, best_ep = acc, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    # `stopped` tells converged (early-stop) from truncated (hit the cap) — without it
    # an under-trained arm is indistinguishable from a converged one.
    return model, best_acc, best_ep, ep


def check_permutation_invariance(model, sp: PaddedSplit, args) -> None:
    """Duplicate an option and assert both copies score identically.

    This is the load-bearing invariant for a set model on this data: ~65% of MAIN
    decisions contain duplicate cards with byte-identical feature vectors. If the
    architecture can tell them apart it will fit noise. Fails loudly rather than
    silently degrading.
    """
    sel = torch.arange(min(64, sp.n_dec), device=DEV)
    x, mask, _ = sp.batch(sel)
    lens = (~mask).sum(1)
    ok = torch.ones_like(lens, dtype=torch.bool)
    x2, m2 = x.clone(), mask.clone()
    for i in range(len(sel)):
        n = int(lens[i])
        if n < 2 or n >= x.shape[1]:
            ok[i] = False
            continue
        x2[i, n] = x2[i, 0]            # append a copy of option 0
        m2[i, n] = False
    model.eval()
    with torch.no_grad():
        s = model(x2, m2)
    worst = 0.0
    for i in range(len(sel)):
        if not ok[i]:
            continue
        n = int(lens[i])
        worst = max(worst, abs(float(s[i, 0] - s[i, n])))
    print(f"  permutation/duplicate invariance: max |score(dup) - score(orig)| = {worst:.2e}")
    if worst > 1e-4:
        raise SystemExit("FAIL: identical options got different scores — is there a "
                         "positional encoding or an order-dependent pooling op?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mlp", "deep", "deepsets", "setxf", "setxf2"])
    ap.add_argument("--dataset", default="yushinito_pooled.jsonl.gz")
    ap.add_argument("--profile", default="ALAKAZAM")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--eval-bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--use-cross", action="store_true",
                    help="setxf2: also feed the hand-crafted A(x)S / B(x)R blocks "
                         "(555 of 658 dims). Off by default so the question "
                         "'do they still earn their place?' is measured, not assumed.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="epochs of linear LR warmup (transformers usually need it; "
                         "0 = off, matching the other arms)")
    ap.add_argument("--check-perm", action="store_true", default=True)
    ap.add_argument("--baseline", type=float, default=None,
                    help="TEST accuracy of the `mlp` arm from THIS sweep, to print a delta")
    ap.add_argument("--emit-raw", default=None, help="write the trained ensemble as JSON")
    args = ap.parse_args()

    profile = get_profile(args.profile)
    # cache key includes the DATASET, not just the profile: colab_train_mlp keys on
    # profile alone and would silently reuse a stale cache built from other data.
    cache = Path(f"cache_{args.profile}_{Path(args.dataset).stem}.npz")
    data = dict(np.load(cache)) if cache.exists() else build_cache(
        Path(args.dataset), Path("split.json"), profile, cache)

    sp = {k: PaddedSplit(data[f"{k}_X"], data[f"{k}_lens"], data[f"{k}_pick"],
                         data.get(f"{k}_won"), data.get(f"{k}_arch"))
          for k in ("train", "val", "test")}
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"device={DEV} arm={args.arm} dim={tr.dim} max_opts={tr.max_opts} "
          f"train={tr.n_dec} val={va.n_dec} test={te.n_dec}", flush=True)

    models, per_seed, epochs_used = [], [], []
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        m = build_model(args.arm, tr.dim, args)
        if seed == 0:
            n_par = sum(p.numel() for p in m.parameters())
            print(f"  parameters/member: {n_par:,}", flush=True)
        m, v, best_ep, last_ep = train_one(m, tr, va, args, seed)
        t = evaluate([m], te, args.eval_bs)
        per_seed.append(t)
        models.append(m)
        epochs_used.append((best_ep, last_ep))
        trunc = "  <-- HIT EPOCH CAP (under-trained?)" if last_ep >= args.epochs else ""
        print(f"  seed {seed}: val {v:.4f}  test {t:.4f}  "
              f"(best@ep{best_ep}, ran {last_ep}){trunc}", flush=True)
        if seed == 0 and args.check_perm:
            check_permutation_invariance(m, te, args)

    ens_tr = evaluate(models, tr, args.eval_bs)
    ens_te = evaluate(models, te, args.eval_bs)
    arr = np.array(per_seed)
    se = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    print(f"\n{'='*70}")
    print(f"ARM {args.arm}  ({args.seeds} seeds, {time.perf_counter()-t0:.0f}s)")
    print(f"  per-seed TEST: mean {arr.mean():.4f} +- SE {se:.4f}"
          f"  [min {arr.min():.4f}, max {arr.max():.4f}]")
    print(f"  ENSEMBLE  train {ens_tr:.4f}   TEST {ens_te:.4f}   gap {ens_tr-ens_te:+.4f}")
    # Members cost inference time LINEARLY and the budget is 600 s/game (measured:
    # 11.7 M MAC/s in pure stdlib Python). d=128/L=3 fits 3 members at ~143 s/game but
    # 10 members needs ~476 s — 79% of budget, no margin. So the member count must be
    # chosen from THIS curve, not inherited from M31's trim of a different model.
    curve = [k for k in (1, 2, 3, 5, 10) if k <= len(models)]
    if len(curve) > 1:
        print("  ensemble curve (first-k members):")
        for k in curve:
            print(f"    k={k:<3} TEST {evaluate(models[:k], te, args.eval_bs):.4f}")
    if args.baseline is not None:
        print(f"  vs baseline {args.baseline:.4f}: {ens_te-args.baseline:+.4f}")
    else:
        print("  NOTE: the baseline is the `mlp` arm of THIS sweep, not a historical number.")
        print("        Yushin keeps playing, so the corpus (and therefore the TEST set) grows;")
        print("        M31's 0.7831 was measured on a smaller held-out set and is NOT comparable.")
    if te.arch is not None:
        for lab, code in (("vs Marnie's Grimmsnarl", 1), ("vs everyone else", 2)):
            sub = (te.arch == code).nonzero(as_tuple=True)[0]
            if len(sub):
                print(f"  {lab:<24} n={len(sub):>6}  TEST {evaluate(models, te, args.eval_bs, sub):.4f}")
    if te.won is not None:
        for lab, want in (("teacher WON", True), ("teacher LOST", False)):
            sub = (te.won == want).nonzero(as_tuple=True)[0]
            if len(sub):
                print(f"  {lab:<24} n={len(sub):>6}  TEST {evaluate(models, te, args.eval_bs, sub):.4f}")
    print("=" * 70)

    if args.emit_raw:
        Path(args.emit_raw).write_text(json.dumps({
            "kind": f"{args.arm}_ensemble", "arm": args.arm, "dim": tr.dim,
            "config": {"d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                       "h": args.h},
            "members": [{k: v.cpu().numpy().tolist() for k, v in m.state_dict().items()}
                        for m in models],
        }), encoding="utf-8")
        print(f"wrote {args.emit_raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
