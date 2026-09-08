"""M35 — fidelity measured by CARD IDENTITY, not by option index.

WHY THIS EXISTS. `train.py::_bc_accuracy` (line 216) scores a prediction as

    set(np.argsort(-(X @ w))[:k]) == set(chosen)

i.e. it compares sets of OPTION INDICES. When several options are the same card,
picking a different copy is behaviourally identical — the live policy resolves it
to the same game action — yet the metric counts it as an error. Decks with many
duplicate copies are therefore systematically understated.

Measured on the val split (val_fraction=0.2, seed=0, the split `train_bc` uses):

    deck                 context   index    strict    delta
    Luca (Grimmsnarl)    TO_HAND   0.543    0.736    +0.193
    Luca (Grimmsnarl)    MAIN      0.599    0.626    +0.027
    Yushin (ALAKAZAM)    TO_HAND   0.644    0.669    +0.025
    Yushin (ALAKAZAM)    MAIN      0.645    0.645    +0.000   <-- the control

Yushin's MAIN correction is EXACTLY ZERO. This is not a metric that flatters
everyone: it corrects a deck-specific distortion. Luca's list runs 10 copies of one
basic {D} Energy; Yushin's 7 energies are spread over 3 different cards.

TWO KEYS, and the difference matters. A "loose" key of (kind, card_id,
target_card_id, attack_id, number) reads Luca's MAIN at 0.674 — but it is WRONG,
because it merges two Impidimp on the bench that carry different damage. The
STRICT key replaces target_card_id with the target's unique `serial`, so distinct
in-play Pokemon stay distinct. Strict is the number to gate on (0.626, not 0.674).

TRAINING IS NOT AFFECTED. Duplicate options produce identical feature vectors, so
the softmax NLL is symmetric and the gradient is correct. Only the *measurement*
breaks — nothing needs retraining, it needs re-measuring.

`train.py::_bc_accuracy` is deliberately NOT patched: it is also the L2 selection
criterion, so changing it would change which weights get shipped and would break
comparability with every historical payload. Gate with this tool instead.

READ-ONLY. Run:
  uv run --group dev python scratchpad/semantic_fidelity.py \
      data/imitation/luca_full.jsonl.gz data/models/bc_luca_full.json GRIMMSNARL
  # add --no-control to skip the Yushin control (not recommended)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import ptcg_ai.imitation.features as F
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.observation.models import AreaKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

#: The control. If its MAIN delta is not ~0.000 the instrument is wrong, not the
#: candidate — Yushin's deck has no duplicate-heavy context, so it must read flat.
CONTROL = ("data/imitation/yushinito_full.jsonl.gz", "data/models/bc_alakazam_full.json",
           "ALAKAZAM", "Yushin (control)")
CONTROL_MAIN_DELTA_MAX = 0.01


def _target_serial(opt, obs):
    """Unique (player, serial) of the in-play Pokemon this option acts on, else None.

    `serial` is what separates two Pokemon of the same species — without it a
    bench of two Impidimp with different damage collapses into one key and the
    metric reads high for the wrong reason.
    """
    state = obs.current
    if state is None:
        return None
    pairs = ((getattr(opt, "inPlayArea", None), getattr(opt, "inPlayIndex", None)),
             (opt.area, opt.index))
    for area, idx in pairs:
        if area not in (AreaKind.ACTIVE, AreaKind.BENCH) or idx is None:
            continue
        pi = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
        if not (0 <= pi < len(state.players)):
            continue
        player = state.players[pi]
        zone = player.active if area is AreaKind.ACTIVE else player.bench
        if zone and 0 <= idx < len(zone) and zone[idx] is not None:
            return (pi, zone[idx].serial)
    return None


def strict_key(opt, obs):
    """What actually distinguishes two options behaviourally."""
    r = resolve_option(opt, obs)
    return (opt.type.name, r.card_id, r.attack_id, r.number, _target_serial(opt, obs))


def _setxf2_scorer(spec):
    """M37: torch-backed scorer for a `setxf2_ensemble` spec.

    Rebuilds the trained `SetTransformerV2` from the FLAT shipped payload and runs it in
    **float64**, which is what makes this a faithful stand-in for the stdlib kernel:
    `tests/unit/test_setnet.py` pins `setnet.score_set` to this same class at 1e-9, so
    scoring here is scoring what the agent ships. Doing it in pure Python instead would
    cost ~4 h for one sweep of the held-out set; torch does it in about a minute.

    Imported lazily so payloads without a set model still work on a torch-less machine.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from colab_train_set import SetTransformerV2

    d, L, H = int(spec["d_model"]), int(spec["layers"]), int(spec["heads"])
    dim = int(spec.get("dim", 658))
    models = []
    for m in spec["members"]:
        net = SetTransformerV2(dim, d_model=d, layers=L, heads=H, use_cross=False).double()
        sd = net.state_dict()
        flat = {
            "opt_proj.weight": m["opt_w"], "opt_proj.bias": m["opt_b"],
            "state_enc.0.weight": m["state0_w"], "state_enc.0.bias": m["state0_b"],
            "state_enc.2.weight": m["state2_w"], "state_enc.2.bias": m["state2_b"],
            "film.weight": m["film_w"], "film.bias": m["film_b"],
            "ln_out.weight": m["lnout_w"], "ln_out.bias": m["lnout_b"],
            "head.weight": m["head_w"], "head.bias": [m["head_b"]],
        }
        for i, blk in enumerate(m["blocks"]):
            flat.update({
                f"blocks.{i}.ln1.weight": blk["ln1_w"], f"blocks.{i}.ln1.bias": blk["ln1_b"],
                f"blocks.{i}.ln2.weight": blk["ln2_w"], f"blocks.{i}.ln2.bias": blk["ln2_b"],
                f"blocks.{i}.attn.in_proj_weight": blk["in_proj_weight"],
                f"blocks.{i}.attn.in_proj_bias": blk["in_proj_bias"],
                f"blocks.{i}.attn.out_proj.weight": blk["out_proj_weight"],
                f"blocks.{i}.attn.out_proj.bias": blk["out_proj_bias"],
                f"blocks.{i}.ff.0.weight": blk["ff0_w"], f"blocks.{i}.ff.0.bias": blk["ff0_b"],
                f"blocks.{i}.ff.2.weight": blk["ff2_w"], f"blocks.{i}.ff.2.bias": blk["ff2_b"],
            })
        net.load_state_dict({k: torch.tensor(v, dtype=torch.float64).reshape(sd[k].shape)
                             for k, v in flat.items()})
        net.eval()
        models.append(net)

    def score(X):
        xt = torch.tensor(np.asarray(X, dtype=np.float64)).unsqueeze(0)
        mask = torch.zeros(1, xt.shape[1], dtype=torch.bool)
        with torch.no_grad():
            out = sum(net(xt, mask)[0] for net in models) / len(models)
        return out.numpy()
    return score


def _scorer(spec):
    """Return f(X) -> scores for a linear vector or an mlp / mlp_ensemble spec.

    Mirrors `policy._score` exactly (mean over ensemble members, relu hidden), so
    the numbers here are the ones the shipped agent would actually produce.
    """
    if isinstance(spec, list):
        w = np.asarray(spec, dtype=np.float32)
        return lambda X: X @ w
    if not isinstance(spec, dict):
        return None
    kind = spec.get("kind")
    if kind == "setxf2_ensemble":
        return _setxf2_scorer(spec)
    members = spec.get("members") if kind == "mlp_ensemble" else ([spec] if kind == "mlp" else None)
    if not members:
        return None                      # mlp_switch etc. — not handled, report skip
    packed = [(np.asarray(m["W1"], dtype=np.float32).reshape(len(m["W1"]) // m["h"], m["h"])
               if np.asarray(m["W1"]).ndim == 1 else np.asarray(m["W1"], dtype=np.float32),
               np.asarray(m["b1"], dtype=np.float32),
               np.asarray(m["w2"], dtype=np.float32),
               float(m["b2"])) for m in members]

    def score(X):
        total = np.zeros(X.shape[0], dtype=np.float32)
        for W1, b1, w2, b2 in packed:
            total += np.maximum(X @ W1 + b1, 0.0) @ w2 + b2
        return total / len(packed)
    return score


def measure(dataset: str, payload_path: str, profile_name: str, context: str,
            cards: CardDatabase, parser: ObservationParser) -> dict | None:
    """Score one context on the held-out 20%.

    `split_by_game(rows, 0.2, seed=0)` is BOTH `train_bc`'s val split and the TEST
    split of `train_mlp_alakazam.py`'s 3-way scheme — the same 20% either way. It
    hashes the game_id, so a game's bucket is stable when new games are added:
    a model trained on a subset of today's data never saw any of this held-out set,
    which makes "did more data help?" a clean comparison.
    """
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    spec = payload.get("contexts", {}).get(context)
    score_fn = _scorer(spec)
    if score_fn is None:
        return None                      # not a scorable context in this payload
    profile = get_profile(profile_name)
    rows = list(read_decision_dataset(Path(dataset)))
    _, val = split_by_game(rows, val_fraction=0.2, seed=0)

    n = dup = trivial = idx_ok = strict_ok = 0
    for r in val:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context.name != context or F.is_prize_pick(obs.select):
            continue
        options = obs.select.option
        chosen = [a for a in r.action if 0 <= a < len(options)]
        # same filters train.py::_featurize applies, so n matches its n_val
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards),
                       dtype=np.float32)
        scores = score_fn(X)
        # lowest-index tie-break, matching the live policy._rank_order. np.argsort
        # is unstable and would not reproduce shipped behaviour.
        pred = sorted(range(len(options)), key=lambda i: (-scores[i], i))[:len(chosen)]
        keys = [strict_key(o, obs) for o in options]
        n += 1
        dup += len(set(keys)) < len(keys)
        trivial += len(set(keys)) == 1
        idx_ok += set(pred) == set(chosen)
        strict_ok += sorted(keys[i] for i in pred) == sorted(keys[i] for i in chosen)

    if n == 0:
        return None
    return {"n": n, "dup": dup / n, "trivial": trivial / n,
            "index": idx_ok / n, "strict": strict_ok / n,
            "delta": (strict_ok - idx_ok) / n}


def _row(label: str, context: str, m: dict) -> str:
    return (f"  {label:<22}{context:<10}{m['n']:>7}{m['dup']:>8.1%}{m['trivial']:>8.1%}"
            f"{m['index']:>9.3f}{m['strict']:>9.3f}{m['delta']:>+8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("payload")
    ap.add_argument("profile")
    ap.add_argument("--contexts", nargs="*", default=None,
                    help="default: every linear context in the payload")
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the Yushin control (not recommended — it is what "
                         "proves the correction is deck-specific, not universal)")
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    contexts = args.contexts or sorted(
        c for c, s in payload.get("contexts", {}).items()
        if isinstance(s, list) and any(v != 0.0 for v in s))

    print("=" * 84)
    print("M35 — fidelity by CARD IDENTITY vs by option index  (val split, seed=0)")
    print("=" * 84)
    print(f"  {'deck':<22}{'context':<10}{'n':>7}{'dup':>8}{'trivial':>8}"
          f"{'index':>9}{'strict':>9}{'delta':>8}")

    results = {}
    for ctx in contexts:
        m = measure(args.dataset, args.payload, args.profile, ctx, cards, parser)
        if m:
            results[ctx] = m
            print(_row(args.label, ctx, m))

    if not args.no_control:
        print()
        ctrl_main = None
        for ctx in ("MAIN", "TO_HAND"):
            m = measure(CONTROL[0], CONTROL[1], CONTROL[2], ctx, cards, parser)
            if m:
                print(_row(CONTROL[3], ctx, m))
                if ctx == "MAIN":
                    ctrl_main = m
        if ctrl_main is not None:
            ok = abs(ctrl_main["delta"]) <= CONTROL_MAIN_DELTA_MAX
            print(f"\n  CONTROL: Yushin MAIN delta = {ctrl_main['delta']:+.3f} "
                  f"(must be <= {CONTROL_MAIN_DELTA_MAX:.3f})  -> {'OK' if ok else 'FAIL'}")
            if not ok:
                print("  The control moved. Distrust the candidate numbers above: the")
                print("  instrument is suspect, not the deck.")

    if "MAIN" in results:
        print(f"\n  Gate this on the STRICT column: MAIN = {results['MAIN']['strict']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
