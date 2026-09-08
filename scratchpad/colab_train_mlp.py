"""M31 — GPU (PyTorch) port of the ALAKAZAM MAIN MLP trainer, to run on Colab.

Faithful port of ``scratchpad/train_mlp_main.py::_train_mlp`` so results are comparable to the local
numpy runs. Everything that affects the result is replicated exactly:

  * score(option) = w2 . relu(W1 x + b1) + b2
  * conditional logit: softmax over EACH DECISION's options, weighted cross-entropy vs the taken one
  * L2 on W1/w2 only (never the biases), added as ``l2 * W`` to the gradient (i.e. (l2/2)||W||^2)
  * Adam, lr 0.01, FULL BATCH, <=400 epochs, val-accuracy check every 10 epochs, patience 6 checks,
    never stopping before epoch 100
  * identical initialisation: numpy ``default_rng(seed)``, W1 ~ N(0,1)*sqrt(2/dim), w2 ~ N(0,1)*sqrt(1/h)
  * ensemble score = SUM of member scores (not mean) -- matches ``_mlp_accuracy``
  * the split is read from ``split.json`` (computed locally), so it is bit-identical to our runs

VALIDATION GATE: run ``--h 48`` first. It must reproduce the local TEST accuracy of ~0.7685
(local reference: linear 0.5377, MLP h=48 0.7685 gap 0.018, h=96 0.7741). If it does not match
within seed noise (~0.006), the port is wrong -- do not trust any further GPU numbers.

Colab usage:
    !unzip -q colab_export.zip -d work && cd work
    !python colab_train_mlp.py --h 48 --seeds 3          # validation run
    !python colab_train_mlp.py --h 48 --seeds 3 --emit bc_alakazam_mlp_gpu.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

# the pruned stdlib-only subtree staged next to this file
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- featurise

#: M32 residual fix: opponent-archetype label per decision, from the optional
#: archmap.json (game_id -> archetype string) staged by export_for_colab.py.
#: 0 = unmapped/no label (e.g. a mirror match), 1 = vs Marnie's Grimmsnarl, 2 = everyone else.
_ARCH_GRIMM = "Marnie's Grimmsnarl"


def build_cache(dataset: Path, split_path: Path, profile, cache: Path) -> dict:
    """Featurise MAIN once (CPU, stdlib code path) and cache as npz."""
    cards = CardDatabase.from_records(json.loads(Path("card_data.json").read_text(encoding="utf-8")))
    parser = ObservationParser()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    archmap_path = Path("archmap.json")
    archmap = json.loads(archmap_path.read_text(encoding="utf-8")) if archmap_path.exists() else {}

    per = {"train": [], "val": [], "test": []}
    t0 = time.perf_counter()
    with gzip.open(dataset, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            part = split.get(r["game_id"])
            if part is None:
                continue
            obs = parser.parse(r["raw_observation"])
            if obs.select is None or obs.current is None:
                continue
            if obs.select.context is not SelectContextKind.MAIN or F.is_prize_pick(obs.select):
                continue
            n = len(obs.select.option)
            chosen = [a for a in r["action"] if 0 <= a < n]
            if not chosen or len(set(chosen)) != len(chosen):
                continue
            gs = GameState.build(obs, cards)
            X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards), dtype=np.float32)
            arch = archmap.get(r["game_id"])
            label = 1 if arch == _ARCH_GRIMM else (2 if arch is not None else 0)
            per[part].append((X, chosen[0], bool(r["won"]), label))
    print(f"featurised in {time.perf_counter()-t0:.0f}s: "
          + " ".join(f"{k}={len(v)}" for k, v in per.items()), flush=True)

    out = {}
    for k, recs in per.items():
        out[f"{k}_X"] = np.concatenate([X for X, _, _, _ in recs], axis=0)
        out[f"{k}_lens"] = np.array([X.shape[0] for X, _, _, _ in recs], dtype=np.int64)
        out[f"{k}_pick"] = np.array([c for _, c, _, _ in recs], dtype=np.int64)
        out[f"{k}_won"] = np.array([w for _, _, w, _ in recs], dtype=bool)
        out[f"{k}_arch"] = np.array([a for _, _, _, a in recs], dtype=np.int8)
    np.savez_compressed(cache, **out)
    return out


def mask_columns(profile, groups: list[str]) -> list[int]:
    """Column indices to ZERO so the model cannot use the named snapshot groups.

    Layout: [a _A][b b_len][c c_len][d d_len][e _E][f _F][s S][a(x)s _A*S][b(x)r b_len*R].
    Zeroing snapshot slot j must also kill its _A interaction copies in the a(x)s block,
    otherwise the group leaks back in through the interactions.
    """
    from ptcg_ai.imitation.deck_profiles import ALAKAZAM_V2_GROUPS

    S = profile.snapshot_len
    s0 = 12 + profile.b_len + profile.c_len + profile.d_len + 6 + 5   # start of the s block
    as0 = s0 + S                                                      # start of the a(x)s block
    cols: list[int] = []
    for g in groups:
        lo, hi = ALAKAZAM_V2_GROUPS[g]
        for j in range(lo, hi):
            cols.append(s0 + j)
            cols.extend(as0 + i * S + j for i in range(12))
    return cols


def filter_by_arch(X, lens, pick, won, arch, label: int):
    """M32 Strategy B: keep only DECISIONS whose archetype label matches (1=vs
    Marnie's Grimmsnarl), for training the Grimmsnarl-routed specialist ensemble
    on an undiluted slice. Filters at the decision level (not a flat row mask),
    since each decision spans a variable number of option-rows in X."""
    starts = np.zeros(len(lens), dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    keep = np.where(arch == label)[0]
    if len(keep) == 0:
        raise SystemExit(f"filter_by_arch: no decisions with label={label} -- check archmap.json")
    row_mask = np.zeros(X.shape[0], dtype=bool)
    for i in keep:
        row_mask[starts[i]: starts[i] + lens[i]] = True
    return X[row_mask], lens[keep], pick[keep], (won[keep] if won is not None else None)


class Split:
    """Flat option matrix + per-decision segment bookkeeping, on device."""

    def __init__(self, X, lens, pick, won=None, alpha=1.0):
        self.n_dec = len(lens)
        # M31 Track A: down-weight decisions from LOST games (alpha). Linear triage found
        # alpha=0.5 strictly better on BOTH held-out metrics; alpha=1.0 reproduces the baseline.
        w = np.ones(self.n_dec, dtype=np.float32)
        if won is not None and alpha != 1.0:
            w[~np.asarray(won, dtype=bool)] = alpha
        self.w = torch.from_numpy(w).to(DEV)
        starts = np.zeros(self.n_dec, dtype=np.int64)
        np.cumsum(lens[:-1], out=starts[1:])
        self.X = torch.from_numpy(X).to(DEV)
        self.seg = torch.from_numpy(np.repeat(np.arange(self.n_dec), lens)).to(DEV)
        self.chosen_global = torch.from_numpy(starts + pick).to(DEV)
        # padded row-index map, for a vectorised per-decision argmax
        m = int(lens.max())
        idx = np.full((self.n_dec, m), -1, dtype=np.int64)
        for i, (s, ln) in enumerate(zip(starts, lens)):
            idx[i, :ln] = np.arange(s, s + ln)
        self.pad_idx = torch.from_numpy(np.clip(idx, 0, None)).to(DEV)
        self.pad_mask = torch.from_numpy(idx < 0).to(DEV)
        self.pick = torch.from_numpy(pick).to(DEV)

    def scores(self, P):
        return torch.relu(self.X @ P["W1"] + P["b1"]) @ P["w2"] + P["b2"]

    def loss(self, P, l2):
        s = self.scores(P)
        seg_max = torch.full((self.n_dec,), float("-inf"), device=DEV)
        seg_max = seg_max.scatter_reduce(0, self.seg, s, reduce="amax", include_self=True)
        ex = torch.exp(s - seg_max[self.seg])
        seg_sum = torch.zeros(self.n_dec, device=DEV).index_add_(0, self.seg, ex)
        lse = torch.log(seg_sum) + seg_max                      # per-decision logsumexp
        per = -(s[self.chosen_global] - lse)                    # per-decision NLL
        nll = (self.w * per).sum() / self.w.sum()               # weighted mean (matches numpy trainer)
        return nll + 0.5 * l2 * (P["W1"].pow(2).sum() + P["w2"].pow(2).sum())

    @torch.no_grad()
    def accuracy(self, params_list) -> float:
        s = sum(self.scores(P) for P in params_list)            # ensemble = SUM of scores
        padded = s[self.pad_idx].masked_fill(self.pad_mask, float("-inf"))
        return (padded.argmax(dim=1) == self.pick).float().mean().item()


def init_params(dim: int, h: int, seed: int) -> dict:
    """Identical initialisation to the numpy trainer (same RNG, same scales)."""
    rng = np.random.default_rng(seed)
    mk = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV, requires_grad=True)
    return {
        "W1": mk(rng.standard_normal((dim, h)) * np.sqrt(2.0 / dim)),
        "b1": mk(np.zeros(h)),
        "w2": mk(rng.standard_normal(h) * np.sqrt(1.0 / h)),
        "b2": mk(np.array(0.0)),
    }


def train_one(tr: Split, va: Split, dim, h, l2, seed, epochs=400, lr=0.01, patience=6):
    P = init_params(dim, h, seed)
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    best_acc, best, wait = -1.0, None, 0
    for t in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        tr.loss(P, l2).backward()
        opt.step()
        if t % 10 == 0:
            acc = va.accuracy([P])
            if acc > best_acc:
                best_acc, wait = acc, 0
                best = {k: v.detach().clone() for k, v in P.items()}
            else:
                wait += 1
                if wait >= patience and t > 100:
                    break
    return best, best_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dataset", default="yushinito_full.jsonl.gz")
    ap.add_argument("--profile", default="ALAKAZAM")
    ap.add_argument("--cache", default="main_cache.npz")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight on decisions from LOST games (M31 Track A: 0.5 won the linear triage)")
    ap.add_argument("--mask-groups", default="", help="comma-separated V2 snapshot groups to ZERO, "
                    "e.g. 'G1,G2,G3,G4' reproduces the V1 feature set (M31 Track E ablation)")
    ap.add_argument("--emit", default=None, help="write a full shippable weights payload")
    ap.add_argument("--emit-raw", default=None,
                    help="write JUST the {kind:mlp_ensemble, members:[...]} spec, no base-payload "
                         "merge -- for offline counterfactual re-scoring before a full bundle is worth "
                         "building (M32 residual-fix check)")
    ap.add_argument("--base-payload", default=None,
                    help="linear payload to merge MAIN into; defaults to the one matching --profile. "
                         "It MUST have been trained at the same profile dim: load_weights validates "
                         "feature_dim AND the length of every linear context vector.")
    ap.add_argument("--filter-train-arch", default=None, choices=["grimm", "other"],
                     help="M32 Strategy B: train+val ONLY on decisions from this archetype "
                          "(requires archmap.json staged by export_for_colab.py). TEST is never "
                          "filtered, so the existing sliced eval stays apples-to-apples against "
                          "the unfiltered general model.")
    args = ap.parse_args()

    profile = get_profile(args.profile)
    # cache is profile-specific: a cache built at dim 658 would be silently stale for V2
    cache = Path(args.cache if args.cache != "main_cache.npz"
                 else f"main_cache_{profile.name.lower()}.npz")
    data = dict(np.load(cache)) if cache.exists() else build_cache(
        Path(args.dataset), Path("split.json"), profile, cache)

    groups = [g.strip() for g in args.mask_groups.split(",") if g.strip()]
    if groups:
        cols = mask_columns(profile, groups)
        for k in ("train", "val", "test"):
            data[f"{k}_X"][:, cols] = 0.0
        print(f"masked groups {groups} -> {len(cols)} columns zeroed", flush=True)

    if args.filter_train_arch:
        label = 1 if args.filter_train_arch == "grimm" else 2
        for k in ("train", "val"):
            before = len(data[f"{k}_lens"])
            fX, flens, fpick, fwon = filter_by_arch(
                data[f"{k}_X"], data[f"{k}_lens"], data[f"{k}_pick"], data.get(f"{k}_won"),
                data[f"{k}_arch"], label)
            data[f"{k}_X"], data[f"{k}_lens"], data[f"{k}_pick"] = fX, flens, fpick
            if fwon is not None:
                data[f"{k}_won"] = fwon
            print(f"--filter-train-arch={args.filter_train_arch}: {k} {before} -> {len(flens)} decisions",
                  flush=True)

    parts = {}
    for k in ("train", "val", "test"):
        # alpha applies to TRAINING only; evaluation is always unweighted
        a = args.alpha if k == "train" else 1.0
        parts[k] = Split(data[f"{k}_X"], data[f"{k}_lens"], data[f"{k}_pick"],
                         won=data.get(f"{k}_won"), alpha=a)
    dim = profile.feature_dim
    print(f"device={DEV} dim={dim} h={args.h} seeds={args.seeds} alpha={args.alpha} | "
          + " ".join(f"{k}={v.n_dec}" for k, v in parts.items()), flush=True)

    members, seed_tests = [], []
    t0 = time.perf_counter()
    for s in range(args.seeds):
        best_P, best_va = None, -1.0
        for l2 in (1e-4, 1e-3):                                  # same grid as local
            P, acc = train_one(parts["train"], parts["val"], dim, args.h, l2, s)
            if acc > best_va:
                best_P, best_va = P, acc
        members.append(best_P)
        te_s = parts["test"].accuracy([best_P])
        seed_tests.append(te_s)
        print(f"  seed {s}: val {best_va:.4f}  test {te_s:.4f}", flush=True)

    tr_a = parts["train"].accuracy(members)
    va_a = parts["val"].accuracy(members)
    te_a = parts["test"].accuracy(members)
    # The GATE is the per-seed MEAN +- SE, not the single ensemble number: one lucky seed
    # must not be able to carry a result (M31 plan / Lux-9th p-value discipline).
    arr = np.array(seed_tests)
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    print(f"\nper-seed TEST  mean {arr.mean():.4f} +- {se:.4f} (SE)  "
          f"std {arr.std(ddof=1):.4f}  min {arr.min():.4f}  max {arr.max():.4f}  n={len(arr)}")
    print(f"MLP-ens  train {tr_a:.4f}  val {va_a:.4f}  TEST {te_a:.4f}  "
          f"(train-test gap {tr_a-te_a:+.4f})   [{time.perf_counter()-t0:.0f}s total]")
    won = data.get("test_won")
    if won is not None:                                   # matched-target metric for alpha runs
        sel = torch.from_numpy(np.asarray(won, dtype=bool)).to(DEV)
        te = parts["test"]
        s = sum(te.scores(P) for P in members)
        padded = s[te.pad_idx].masked_fill(te.pad_mask, float("-inf"))
        ok = (padded.argmax(dim=1) == te.pick)
        print(f"TEST-won (games the teacher won): {ok[sel].float().mean().item():.4f}  n={int(sel.sum())}")
    arch = data.get("test_arch")
    if arch is not None:                                    # M32 residual-fix slice
        te = parts["test"]
        s = sum(te.scores(P) for P in members)
        padded = s[te.pad_idx].masked_fill(te.pad_mask, float("-inf"))
        ok = (padded.argmax(dim=1) == te.pick)
        arch_t = torch.from_numpy(arch)
        for label, name in ((1, "vs Marnie's Grimmsnarl"), (2, "vs everyone else (labelled)")):
            sel = (arch_t == label).to(DEV)
            if int(sel.sum()) == 0:
                continue
            print(f"TEST {name}: {ok[sel].float().mean().item():.4f}  n={int(sel.sum())}")
    print(f"LOCAL REFERENCE h=48 TEST 0.7685 (gap 0.018) / h=96 TEST 0.7741 — "
          f"the port is valid only if h=48 lands within ~0.006 of 0.7685.")

    spec = {"kind": "mlp_ensemble", "members": [{
        "kind": "mlp", "h": int(args.h),
        "W1": P["W1"].detach().cpu().numpy().tolist(),
        "b1": P["b1"].detach().cpu().numpy().tolist(),
        "w2": P["w2"].detach().cpu().numpy().tolist(),
        "b2": float(P["b2"].detach().cpu()),
    } for P in members]}

    if args.emit_raw:
        Path(args.emit_raw).write_text(json.dumps(spec), encoding="utf-8")
        print(f"wrote {args.emit_raw} (profile={profile.name}, dim={dim}, {len(members)} members) — "
              f"raw ensemble only, no base-payload merge; download for offline counterfactual re-scoring")

    if args.emit:
        base = args.base_payload or ("bc_alakazam_v2_full.json"
                                     if profile.name == "ALAKAZAM_V2" else "bc_alakazam_full.json")
        payload = json.loads(Path(base).read_text(encoding="utf-8"))
        # Fail loudly rather than emit a payload the shipped loader will reject.
        if int(payload.get("feature_dim", -1)) != dim:
            raise SystemExit(f"base payload {base} is dim {payload.get('feature_dim')} but profile "
                             f"{profile.name} is dim {dim} — retrain the base at this profile first")
        if args.mask_groups:
            print(f"WARNING: emitting weights trained with masked groups {groups}; the SHIPPED "
                  f"featurizer still computes them, so masked columns must stay unused by design.")
        payload["version"] = 2
        payload["profile"] = profile.name
        payload["contexts"]["MAIN"] = spec
        payload["mlp_h"], payload["mlp_seeds"] = int(args.h), int(args.seeds)
        Path(args.emit).write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {args.emit} (base={base}, profile={profile.name}, dim={dim}) — "
              f"download it and build the bundle locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
