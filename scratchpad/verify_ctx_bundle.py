"""M42 G-A4 — verify the EXTRACTED tarball, including that the new head is LIVE.

Why this exists in this exact form: in M37 a bundle shipped the SDK's sample deck
next to ALAKAZAM-trained weights (0 of 9 profile cards present, every option hashed
to the OOV bucket = a blind agent), and it cost a whole upload at ~500 elo.
`validate_submission` passed. `smoke_test_entrypoint` passed with bc_failures: 0 —
an unknown card scores as OOV, it does not raise. The extracted-tarball check that
was run verified profile and feature_dim, both of which were CORRECT.

So structural checks are not enough. This script asserts the thing that actually
matters for M42: driving the SHIPPED entrypoint with REAL ACTIVATE observations
from Yushin's corpus, does the agent decline where the champion never could?

Run:
  PYTHONPATH=src python scratchpad/verify_ctx_bundle.py build/imitation-ctx-a.tar.gz
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.observation.parser import ObservationParser

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
DECK = Path("decks/yushinito.csv")
N_PROBE = 600


def main() -> int:
    tarball = Path(sys.argv[1] if len(sys.argv) > 1 else "build/imitation-ctx-a.tar.gz")
    tmp = Path(tempfile.mkdtemp(prefix="ctxverify_"))
    with tarfile.open(tarball) as tf:
        tf.extractall(tmp)
    print(f"extracted {tarball.name} -> {tmp}")

    # --- 1. deck vs profile vocabulary (the M37 failure) ----------------------
    payload = json.loads((tmp / "bc_weights.json").read_text(encoding="utf-8"))
    deck_ids = [int(x) for x in (tmp / "deck.csv").read_text().split()]
    repo_deck = [int(x) for x in DECK.read_text().split()]
    print(f"  deck: {len(deck_ids)} cards, matches repo deck: {sorted(deck_ids)==sorted(repo_deck)}")
    print(f"  payload profile={payload['profile']} feature_dim={payload['feature_dim']}")
    print(f"  contexts: {sorted(payload['contexts'])}")
    print(f"  profile_overrides: {payload.get('profile_overrides')}")
    assert sorted(deck_ids) == sorted(repo_deck), "BUNDLED DECK != repo deck"
    assert "ACTIVATE" in payload["contexts"], "ACTIVATE head missing from the bundle"

    # --- 2. load the REAL entrypoint out of the extracted tree ---------------
    sys.path.insert(0, str(tmp))
    spec = importlib.util.spec_from_file_location("bundle_main", tmp / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"  _AGENT_READY={getattr(mod, '_AGENT_READY', None)}  "
          f"_IMITATION_READY={getattr(mod, '_IMITATION_READY', None)}")
    assert getattr(mod, "_IMITATION_READY", False), "imitation tier did NOT load"

    # --- 3. drive it with REAL held-out ACTIVATE observations ----------------
    parser = ObservationParser()
    _, test_rows = split_by_game(list(read_decision_dataset(DATASET)), val_fraction=0.2, seed=0)
    probes = []
    for r in test_rows:
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        if obs.select.context.name != "ACTIVATE" or F.is_prize_pick(obs.select):
            continue
        probes.append(r)
        if len(probes) >= N_PROBE:
            break

    out = collections.Counter()
    agree = 0
    for r in probes:
        got = mod.agent(dict(r.raw_observation))
        assert isinstance(got, list) and got and 0 <= got[0] < r.n_options, f"illegal: {got}"
        out["NO" if got != [0] else "YES"] += 1
        agree += got == [a for a in r.action if 0 <= a < r.n_options]
    n = len(probes)
    teacher_no = sum(1 for r in probes if [a for a in r.action if 0 <= a < r.n_options] != [0])
    print(f"\n  drove the SHIPPED entrypoint on {n} real held-out ACTIVATE states:")
    print(f"    teacher declined : {teacher_no}/{n} = {teacher_no/n:.4f}")
    print(f"    bundle declined  : {out['NO']}/{n} = {out['NO']/n:.4f}")
    print(f"    agreement        : {agree/n:.4f}")
    ok = out["NO"] > 0
    print(f"\n  ACTIVATE head is LIVE in the shipped artifact: {'YES' if ok else 'NO — IT IS NOT'}")
    assert ok, "the bundle never declined — the head is not actually firing"
    print("\nG-A4 extracted-bundle verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
