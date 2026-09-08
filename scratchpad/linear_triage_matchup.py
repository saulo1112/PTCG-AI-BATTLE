"""M32 fix-attempt, step 1 — cheap LINEAR triage of ALAKAZAM_MATCHUP.

Question: does adding the 2 Grimmsnarl/hand-disruption-conditioning features
(deck_profiles.ALAKAZAM_MATCHUP) improve held-out MAIN fidelity SPECIFICALLY on
Yushin's own Marnie's Grimmsnarl games, without regressing everything else? This
is the pre-registered gate before spending an MLP retrain (~4h) or a
head_to_head/ladder slot on it.

Same train/val/test SPLIT as train_mlp_alakazam.py (game-hash, seed 0/1) so the
two profiles are directly comparable; TEST is further sliced by opponent
archetype (Grimmsnarl vs everything else) using a game_id -> archetype map built
the same way as diagnose_mlp_teacher_matchup.py (opponent seat found by NOT being
Yushin's deck identity; archetype via diagnose_mlp._archetype / ARCHETYPE_MARKERS).

Run:  uv run --group dev python scratchpad/linear_triage_matchup.py
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import ReplayDecision, read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.train import _L2_GRID, _bc_accuracy, _featurize, _train_single
from ptcg_ai.observation.parser import ObservationParser

import diagnose_mlp as dm
import extract_top_decks as et

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
TEACHER_FOLDER = Path("replays/54773249")


def build_archetype_map() -> dict[str, str]:
    """game_id (file stem) -> opponent archetype, for every teacher game on disk."""
    yushin_deck = tuple(sorted(int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()))
    out: dict[str, str] = {}
    for fp in sorted(glob.glob(str(TEACHER_FOLDER / "*.json"))):
        if "metadata" in fp:
            continue
        f = Path(fp)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        decks = [et.extract_deck(d, s) for s in (0, 1)]
        seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin_deck]
        if len(seats) != 1:
            continue
        our = seats[0]
        out[f.stem] = dm._archetype(decks[1 - our])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-train", type=int, default=0,
                     help="cap TRAIN rows for a fast first-pass triage (0 = full data). "
                          "VAL/TEST are never subsampled -- the reported numbers stay honest, "
                          "only the model quality is a coarser proxy.")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    print(f"[{time.time()-t0:6.1f}s] card DB loaded", flush=True)

    rows = list(read_decision_dataset(DATASET))
    temp_rows, test_rows = split_by_game(rows, val_fraction=0.2, seed=0)
    tr_rows, va_rows = split_by_game(temp_rows, val_fraction=0.25, seed=1)
    print(f"[{time.time()-t0:6.1f}s] rows: train={len(tr_rows)} val={len(va_rows)} test={len(test_rows)}",
          flush=True)
    if args.sample_train:
        random.Random(0).shuffle(tr_rows)
        tr_rows = tr_rows[: args.sample_train]
        print(f"[{time.time()-t0:6.1f}s] SUBSAMPLED train -> {len(tr_rows)} rows (fast triage mode)",
              flush=True)

    archmap = build_archetype_map()
    print(f"[{time.time()-t0:6.1f}s] archetype map built", flush=True)
    grimm_ids = {gid for gid, a in archmap.items() if a == "Marnie's Grimmsnarl"}
    print(f"archetype map: {len(archmap)} games, {len(grimm_ids)} vs Grimmsnarl")

    def _split_test(rows: list[ReplayDecision]) -> tuple[list, list]:
        g = [r for r in rows if r.game_id in grimm_ids]
        o = [r for r in rows if r.game_id in archmap and r.game_id not in grimm_ids]
        return g, o

    test_grimm, test_other = _split_test(test_rows)
    print(f"test decisions belonging to known games: grimm-games rows={len(test_grimm)}  "
          f"other-games rows={len(test_other)}  (unmapped rows dropped from the slice, kept in overall)")

    for profile_name in ("ALAKAZAM", "ALAKAZAM_MATCHUP"):
        profile = get_profile(profile_name)
        dim = profile.feature_dim
        tr = _featurize(profile, tr_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
        print(f"[{time.time()-t0:6.1f}s] {profile_name}: train featurized ({len(tr)} decisions)",
              flush=True)
        va = _featurize(profile, va_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
        te_all = _featurize(profile, test_rows, {"MAIN"}, parser, cards, 1.0)["MAIN"]
        te_grimm = _featurize(profile, test_grimm, {"MAIN"}, parser, cards, 1.0)["MAIN"]
        te_other = _featurize(profile, test_other, {"MAIN"}, parser, cards, 1.0)["MAIN"]
        print(f"[{time.time()-t0:6.1f}s] {profile_name}: val/test featurized, training linear...",
              flush=True)

        best_w, best_va = None, -1.0
        for l2 in _L2_GRID:
            w = _train_single(tr, dim, l2)
            a = _bc_accuracy(va, w)
            if a > best_va:
                best_w, best_va = w, a

        acc_all = _bc_accuracy(te_all, best_w)
        acc_grimm = _bc_accuracy(te_grimm, best_w) if te_grimm else float("nan")
        acc_other = _bc_accuracy(te_other, best_w) if te_other else float("nan")
        print(f"\n{profile_name} (dim={dim})")
        print(f"  val (model-select) acc: {best_va:.4f}")
        print(f"  TEST overall:        {acc_all:.4f}  (n={len(te_all)})")
        print(f"  TEST vs Grimmsnarl:  {acc_grimm:.4f}  (n={len(te_grimm)})")
        print(f"  TEST vs everyone else: {acc_other:.4f}  (n={len(te_other)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
