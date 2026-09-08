"""M12 Phase 1: residual linear-backbone MLP for MAIN, validated leave-archetype-out.

M11's pure MLP replaced the linear scorer wholesale — risky out-of-distribution
(OOD). This trains a SAFER form: score(x) = linear_backbone·x + c·tanh(mlp(x)/c),
where the linear part is v1's proven scorer and the MLP only adds a BOUNDED
correction (init 0 ⇒ starts exactly at v1, must earn every deviation; tanh-bounded
+ L2 ⇒ shrinks toward v1 OOD instead of extrapolating wildly).

Rigor centerpiece: **leave-archetype-out CV**. The teacher's 325 games are grouped
by opponent archetype (Lucario/Cinderace/Alakazam/Archaludon/other); each fold holds
one archetype out entirely and trains on the rest — the honest proxy for "how does
this scorer behave vs an opponent TYPE it never saw" (the ladder-OOD concern M11
never measured). Per fold we compare linear / pure-MLP / residual held-out accuracy.

Selection: best MEAN held-out (OOD) accuracy, and residual must not lose to linear
on any single archetype by more than noise. If it clears, a FINAL residual model is
trained on all data → data/models/bc_650_v3.json.

Run:  uv run --group dev python scratchpad/train_residual_main.py [h]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import _pack, _seg_softmax, _train_single, Decision
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

sys.path.insert(0, "scratchpad")
import extract_top_decks as et  # noqa: E402
from train_mlp_main import _Adam, _mlp_scores  # noqa: E402

PROFILE = get_profile("TR_650")
DATASET = Path("data/imitation/greengreenpurple.jsonl.gz")
TEACHER_LOGS = Path("Logs/Higher ranking logs/650 elo")
V1 = Path("data/models/bc_650_v1.json")
OUT = Path("data/models/bc_650_v3.json")
OUR = "greengreenpurple"
MARK = {678: "Lucario", 666: "Cinderace", 743: "Alakazam", 190: "Archaludon"}
FOLDS = ["Lucario", "Cinderace", "Alakazam", "Archaludon", "other"]


def archetype_map() -> dict[str, str]:
    amap: dict[str, str] = {}
    for f in sorted(TEACHER_LOGS.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ag = [a.get("Name", "") for a in d["info"]["Agents"]]
        ours = [i for i, n in enumerate(ag) if n.lower() == OUR.lower()]
        if len(ours) != 1:
            amap[f.stem] = "other"; continue
        deck = et.extract_deck(d, 1 - ours[0])
        labels = [lab for cid, lab in MARK.items() if deck and cid in deck]
        amap[f.stem] = labels[0] if labels else "other"
    return amap


def featurize_main(parser, cards):
    """MAIN decisions as (game_id, Decision), reusing the live featurizer."""
    out = []
    for r in read_decision_dataset(DATASET):
        if SelectContextKind(r.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(PROFILE, obs.select, obs, gs, cards), dtype=np.float32)
        out.append((r.game_id, Decision(X=X, chosen=chosen, weight=1.0)))
    return out


# --- residual model: score = backbone·x + c·tanh(mlp(x)/c) ---

def _residual_scores(X, backbone, P, c):
    lin = X @ backbone
    raw, Z1, A1 = _mlp_scores(X, P)
    corr = c * np.tanh(raw / c)
    return lin + corr, Z1, A1, raw


def train_residual(decisions, backbone, dim, h, l2=1e-3, c=2.0, seed=0, epochs=250, lr=0.01):
    rng = np.random.default_rng(seed)
    bigX, starts, lens, chosen_global, weights = _pack(decisions)
    onehot = np.zeros(bigX.shape[0]); onehot[chosen_global] = 1.0
    wsum = weights.sum(); per_opt_w = np.repeat(weights, lens)
    P = {"W1": rng.standard_normal((dim, h)) * np.sqrt(2.0 / dim), "b1": np.zeros(h),
         "w2": np.zeros(h), "b2": 0.0}  # correction head init 0 -> starts at backbone
    opt = {k: _Adam(np.shape(P[k]), lr) for k in P}
    for _ in range(epochs):
        scores, Z1, A1, raw = _residual_scores(bigX, backbone, P, c)
        probs = _seg_softmax(scores, starts, lens)
        g = (probs - onehot) * per_opt_w / wsum
        dcorr = g * (1.0 - np.tanh(raw / c) ** 2)         # d(c·tanh(raw/c))/draw
        dw2 = A1.T @ dcorr + l2 * P["w2"]
        db2 = dcorr.sum()
        dZ1 = np.outer(dcorr, P["w2"]) * (Z1 > 0)
        dW1 = bigX.T @ dZ1 + l2 * P["W1"]
        db1 = dZ1.sum(axis=0)
        P["W1"] = opt["W1"].step(P["W1"], dW1); P["b1"] = opt["b1"].step(P["b1"], db1)
        P["w2"] = opt["w2"].step(P["w2"], dw2); P["b2"] = float(opt["b2"].step(np.array(P["b2"]), db2))
    return P


def _acc(decisions, score_fn) -> float:
    ok = 0
    for d in decisions:
        if int(np.argmax(score_fn(d.X))) == d.chosen[0]:
            ok += 1
    return ok / max(len(decisions), 1)


def train_pure_mlp(decisions, dim, h, l2=1e-3, seed=0, epochs=250, lr=0.01):
    rng = np.random.default_rng(seed)
    bigX, starts, lens, chosen_global, weights = _pack(decisions)
    onehot = np.zeros(bigX.shape[0]); onehot[chosen_global] = 1.0
    wsum = weights.sum(); per_opt_w = np.repeat(weights, lens)
    P = {"W1": rng.standard_normal((dim, h)) * np.sqrt(2.0 / dim), "b1": np.zeros(h),
         "w2": rng.standard_normal(h) * np.sqrt(1.0 / h), "b2": 0.0}
    opt = {k: _Adam(np.shape(P[k]), lr) for k in P}
    for _ in range(epochs):
        scores, Z1, A1 = _mlp_scores(bigX, P)
        probs = _seg_softmax(scores, starts, lens)
        g = (probs - onehot) * per_opt_w / wsum
        dw2 = A1.T @ g + l2 * P["w2"]; db2 = g.sum()
        dZ1 = np.outer(g, P["w2"]) * (Z1 > 0)
        dW1 = bigX.T @ dZ1 + l2 * P["W1"]; db1 = dZ1.sum(axis=0)
        P["W1"] = opt["W1"].step(P["W1"], dW1); P["b1"] = opt["b1"].step(P["b1"], db1)
        P["w2"] = opt["w2"].step(P["w2"], dw2); P["b2"] = float(opt["b2"].step(np.array(P["b2"]), db2))
    return P


def main() -> int:
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    dim = PROFILE.feature_dim

    amap = archetype_map()
    data = featurize_main(parser, cards)
    print(f"MAIN decisions: {len(data)}  dim={dim}  h={h}")
    by_fold = {g: sum(1 for gid, _ in data if amap.get(gid, 'other') == g) for g in FOLDS}
    print("held-out sizes:", by_fold)

    c_bound = 2.0
    print(f"\n{'held-out archetype':<20}{'linear':>9}{'pureMLP':>9}{'residual':>9}")
    print("-" * 47)
    lin_accs, pure_accs, res_accs = [], [], []
    for held in FOLDS:
        tr = [d for gid, d in data if amap.get(gid, "other") != held]
        te = [d for gid, d in data if amap.get(gid, "other") == held]
        if not te:
            continue
        backbone = _train_single(tr, dim, 1e-3)
        Ppure = train_pure_mlp(tr, dim, h)
        Pres = train_residual(tr, backbone, dim, h, c=c_bound)
        a_lin = _acc(te, lambda X: X @ backbone)
        a_pure = _acc(te, lambda X, P=Ppure: _mlp_scores(X, P)[0])
        a_res = _acc(te, lambda X, P=Pres: _residual_scores(X, backbone, P, c_bound)[0])
        lin_accs.append(a_lin); pure_accs.append(a_pure); res_accs.append(a_res)
        print(f"{held:<20}{a_lin:>9.4f}{a_pure:>9.4f}{a_res:>9.4f}")
    print("-" * 47)
    ml, mp, mr = np.mean(lin_accs), np.mean(pure_accs), np.mean(res_accs)
    print(f"{'MEAN OOD':<20}{ml:>9.4f}{mp:>9.4f}{mr:>9.4f}")
    worst_vs_lin = min(r - l for r, l in zip(res_accs, lin_accs))
    print(f"\nresidual vs linear: mean {mr-ml:+.4f}, worst single fold {worst_vs_lin:+.4f}")
    print(f"residual vs pureMLP: mean {mr-mp:+.4f}")
    passed = mr >= ml and mr >= mp and worst_vs_lin > -0.01
    print(f"G1-OOD? {'PASS' if passed else 'FAIL'} "
          f"(residual >= both baselines in mean AND no fold worse than linear by >0.01)")

    if passed:
        backbone = _train_single(data_dec := [d for _, d in data], dim, 1e-3)
        Pfinal = train_residual(data_dec, backbone, dim, h, c=c_bound, epochs=400)
        payload = json.loads(V1.read_text(encoding="utf-8"))
        payload["version"] = 2
        payload["contexts"]["MAIN"] = {
            "kind": "residual_mlp", "c": c_bound,
            "backbone": [float(x) for x in backbone],
            "W1": Pfinal["W1"].tolist(), "b1": Pfinal["b1"].tolist(),
            "w2": Pfinal["w2"].tolist(), "b2": float(Pfinal["b2"]),
        }
        OUT.write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
