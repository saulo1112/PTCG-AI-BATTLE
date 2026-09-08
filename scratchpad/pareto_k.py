"""M37 Paso 2 — pick the ensemble size `k` on the measured fidelity/cost Pareto front.

Replaces running `semantic_fidelity.py` once per k. That was both slow and fragile: the
featurisation of the 23,645 held-out MAIN decisions is BY FAR the dominant cost and is
*identical* for every k, yet it was being redone five times, each run holding the whole
229k-row corpus in memory. The five-process version died of memory pressure partway
through. Here the corpus is streamed once, each decision is featurised once, and every
scorer sees the same matrix before it is dropped.

Scorers compared, all on the same held-out 20% (`split_by_game(rows, 0.2, seed=0)`):
  * `mlp`  — the MAIN scorer that ships in `bc_alakazam_fetch.json` today, the baseline
             that matters. The Colab sweep's 0.7561 is NOT comparable: different split
             (TEST vs val) and different metric (option index vs card identity).
  * `setxf2` at k = 1, 3, 5, 7.

Reported on the STRICT (card-identity) column — R21: index-based agreement understates
decks with duplicate copies, and 63.6% of these decisions contain duplicates.

Run:  PYTHONPATH=src python -u scratchpad/pareto_k.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ptcg_ai.imitation.features as F
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState
from semantic_fidelity import _scorer, strict_key

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
BASE = Path("data/models/bc_alakazam_fetch.json")
KS = (1, 3, 5, 7)
#: per-decision cost model fitted from the stdlib kernel benchmark (ms, one member)
COST_A, COST_B = (614.0 - 150.2) / 20.0, 150.2 - 5 * ((614.0 - 150.2) / 20.0)


def main() -> int:
    cards = CardDatabase.from_sdk(load_sdk(load_config(profile="benchmark").paths.sdk_dir))
    parser = ObservationParser()
    profile = get_profile("ALAKAZAM")

    scorers: dict[str, object] = {}
    spec = json.loads(BASE.read_text(encoding="utf-8"))["contexts"]["MAIN"]
    scorers["mlp (base)"] = _scorer(spec)
    for k in KS:
        p = Path(f"data/models/bc_alakazam_setxf2_k{k}.json")
        scorers[f"setxf2 k={k}"] = _scorer(json.loads(p.read_text(encoding="utf-8"))
                                           ["contexts"]["MAIN"])
        print(f"loaded {p.name}", flush=True)

    rows = list(read_decision_dataset(DATASET))
    _, val = split_by_game(rows, val_fraction=0.2, seed=0)
    del rows
    print(f"held-out rows: {len(val)}", flush=True)

    # M37: the headline +0.017 is below the +0.030 bar, but the decision that matters is
    # the Grimmsnarl matchup -- 30.8% of imitation-fetch's real ladder episodes and the
    # documented wall (clone 37% vs teacher 51.9%). A model that is flat overall but much
    # better THERE could still be worth shipping, so slice before concluding.
    archmap: dict[str, str] = {}
    ap = Path("build/colab_export/archmap.json")
    if ap.is_file():
        archmap = json.loads(ap.read_text(encoding="utf-8"))
        print(f"archmap: {len(archmap)} games", flush=True)

    n = 0
    per_game: dict[str, float] = {}
    strict = {name: 0 for name in scorers}
    index = {name: 0 for name in scorers}
    sl_n = {"Marnie's Grimmsnarl": 0, "other archetypes": 0}
    sl_hit = {k: {name: 0 for name in scorers} for k in sl_n}
    for r in val:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context.name != "MAIN" or F.is_prize_pick(obs.select):
            continue
        options = obs.select.option
        chosen = [a for a in r.action if 0 <= a < len(options)]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(profile, obs.select, obs, gs, cards),
                       dtype=np.float32)
        keys = [strict_key(o, obs) for o in options]
        want = sorted(keys[i] for i in chosen)
        n += 1
        # seconds of set-transformer time this decision would cost ONE member, from the
        # stdlib kernel benchmark; accumulated per game so the tail can be read off
        per_game[r.game_id] = per_game.get(r.game_id, 0.0) + \
            (COST_B + COST_A * len(options)) / 1000
        arch = archmap.get(r.game_id)
        bucket = ("Marnie's Grimmsnarl" if arch == "Marnie's Grimmsnarl"
                  else "other archetypes" if arch else None)
        if bucket:
            sl_n[bucket] += 1
        for name, fn in scorers.items():
            s = fn(X)
            pred = sorted(range(len(options)), key=lambda i: (-s[i], i))[:len(chosen)]
            index[name] += set(pred) == set(chosen)
            ok = sorted(keys[i] for i in pred) == want
            strict[name] += ok
            if bucket:
                sl_hit[bucket][name] += ok
        if n % 2000 == 0:
            print(f"  {n} decisions...", flush=True)

    print(f"\n{'scorer':<14}{'n':>7}{'index':>9}{'strict':>9}{'vs base':>10}"
          f"{'p50 s/game':>12}{'p99 s/game':>12}")
    base_strict = strict["mlp (base)"] / n
    costs = list(per_game.values())
    for name in scorers:
        st = strict[name] / n
        k = int(name.split("=")[1]) if "k=" in name else 0
        if k:
            s = sorted(v * k for v in costs)
            p50, p99 = s[len(s) // 2], s[min(len(s) - 1, int(0.99 * len(s)))]
            cost = f"{p50:>12.0f}{p99:>12.0f}"
        else:
            cost = f"{'~0':>12}{'~0':>12}"
        print(f"{name:<14}{n:>7}{index[name] / n:>9.3f}{st:>9.3f}"
              f"{st - base_strict:>+10.3f}{cost}")
    print(f"\nGATE: strict >= {base_strict + 0.030:.3f} (base +0.030) AND p99 < 200 s")

    # The matchup that actually decides our ladder, measured against the SHIPPED model.
    print(f"\n{'STRICT by archetype':<14}" + "".join(f"{k:>22}" for k in sl_n))
    print(f"{'':<14}" + "".join(f"{'(n=' + str(sl_n[k]) + ')':>22}" for k in sl_n))
    for name in scorers:
        row = f"{name:<14}"
        for b in sl_n:
            if not sl_n[b]:
                row += f"{'-':>22}"
                continue
            v = sl_hit[b][name] / sl_n[b]
            d = v - sl_hit[b]["mlp (base)"] / sl_n[b]
            row += f"{v:>15.3f}{d:>+7.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
