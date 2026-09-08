"""M31 — stage everything Colab needs to train the ALAKAZAM MLP on a GPU. READ-ONLY wrt src/.

One local MLP ablation costs ~4 h, which now caps how many FEATURE variants we can try (Track E is
the remaining lever — data and capacity are both saturated, see docs/m31_plan.md). Training is
full-batch matrix-matrix work, i.e. exactly what a GPU eats; on a T4 the same run should be minutes.

What gets uploaded (~10 MB, no vendored SDK):
  * ``card_data.json``   — ``CardDatabase.to_records()``, the same derived card table the Kaggle
                            bundle ships, so Colab needs no engine/SDK checkout;
  * ``ptcg_ai/``         — the pruned STDLIB-ONLY subtree the submission ships (``_AGENT_MODULES``),
                            so featurisation on Colab is the identical code path (no train/serve skew);
  * the dataset ``.jsonl.gz``;
  * ``split.json``       — the exact game_id -> train/val/test assignment computed HERE, so the Colab
                            run reproduces our local split bit-for-bit and the numbers are comparable.

Keep the Colab notebook PRIVATE: ``card_data.json`` is derived from competition material.

Run:  uv run --group dev python scratchpad/export_for_colab.py [out_dir]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.submission.builder import _AGENT_MODULES, _AGENT_PACKAGES

import diagnose_mlp as dm
import extract_top_decks as et

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")

#: M37: optional pooled corpus (new + Yushin's older submission 54486275). If the
#: manifest is present, every game it lists is forced into the TRAIN split. The old
#: bot is a WEAKER version of the same pilot, so letting it reach val/test would both
#: corrupt model selection and break comparability with the GPU baseline (TEST 0.7831),
#: which is defined on the new corpus's held-out 20%. See scratchpad/merge_corpora.py.
POOLED = Path("data/imitation/yushinito_pooled.jsonl.gz")
POOLED_MANIFEST = Path("data/imitation/pooled_manifest.json")
TEACHER_FOLDER = Path("replays/54773249")
_PKG_ROOT = Path("src/ptcg_ai")


def _build_archetype_map() -> dict[str, str]:
    """game_id (file stem) -> opponent archetype ('Marnie's Grimmsnarl'|other names),
    for the M32 residual fix (see ALAKAZAM_MATCHUP). Local-only (uses the full env +
    raw replay JSONs), never shipped or run on Colab."""
    import glob
    import json as _json

    yushin_deck = tuple(sorted(int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()))
    out: dict[str, str] = {}
    for fp in sorted(glob.glob(str(TEACHER_FOLDER / "*.json"))):
        if "metadata" in fp:
            continue
        f = Path(fp)
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        decks = [et.extract_deck(d, s) for s in (0, 1)]
        seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin_deck]
        if len(seats) != 1:
            continue
        our = seats[0]
        out[f.stem] = dm._archetype(decks[1 - our])
    return out


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else Path("build/colab_export")
    if out.exists():
        shutil.rmtree(out)
    (out / "ptcg_ai").mkdir(parents=True)

    # 1. pruned stdlib-only package (same modules the Kaggle bundle ships)
    for pkg in _AGENT_PACKAGES:
        d = out / "ptcg_ai" / pkg
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("", encoding="utf-8")
    for rel in _AGENT_MODULES:
        src = _PKG_ROOT / rel
        dst = out / "ptcg_ai" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"staged {len(_AGENT_MODULES)} stdlib-only modules")

    # 2. card table (no SDK needed on Colab)
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    (out / "card_data.json").write_text(json.dumps(cards.to_records()), encoding="utf-8")

    # 3. dataset
    shutil.copyfile(DATASET, out / DATASET.name)

    # 4. the EXACT split, so Colab numbers are comparable to the local baselines.
    # The split is computed on the NEW corpus only, so val/TEST are byte-for-byte the
    # sets every previous milestone used. Pooled-in old games are appended as TRAIN.
    rows = list(read_decision_dataset(DATASET))
    temp, test = split_by_game(rows, val_fraction=0.2, seed=0)
    tr, va = split_by_game(temp, val_fraction=0.25, seed=1)
    split: dict[str, str] = {}
    for part, name in ((tr, "train"), (va, "val"), (test, "test")):
        for r in part:
            split[r.game_id] = name
    n_forced = 0
    if POOLED_MANIFEST.is_file():
        manifest = json.loads(POOLED_MANIFEST.read_text(encoding="utf-8"))
        for gid in manifest.get("train_only_games", []):
            if gid in split:          # must not silently reassign a val/test game
                raise SystemExit(
                    f"pooled manifest lists {gid}, which is already in the "
                    f"{split[gid]!r} split of the new corpus — corpora overlap")
            split[gid] = "train"
            n_forced += 1
    (out / "split.json").write_text(json.dumps(split), encoding="utf-8")
    n = {k: sum(1 for v in split.values() if v == k) for k in ("train", "val", "test")}
    print(f"split games: {n}  (rows: train={len(tr)} val={len(va)} test={len(test)})")
    if n_forced:
        print(f"  + {n_forced} old-submission games forced to TRAIN (val/TEST untouched)")

    # 4b. M32 residual fix: opponent archetype per game, so Colab can report TEST fidelity
    # sliced by "vs Marnie's Grimmsnarl" (the verified matchup-specific gap) vs everyone else.
    archmap = _build_archetype_map()
    (out / "archmap.json").write_text(json.dumps(archmap), encoding="utf-8")
    n_grimm = sum(1 for a in archmap.values() if a == "Marnie's Grimmsnarl")
    print(f"staged archmap.json: {len(archmap)} games ({n_grimm} vs Grimmsnarl)")

    # linear payloads: Colab merges the GPU-trained MAIN scorer into the one matching the profile
    # (--emit validates dim, so a V2 run needs the V2-dim payload present).
    for name in ("bc_alakazam_full.json", "bc_alakazam_v2_full.json"):
        src_payload = Path("data/models") / name
        if src_payload.is_file():
            shutil.copyfile(src_payload, out / name)
            print(f"staged {name}")
        else:
            print(f"NOTE: {name} not found — --emit for that profile will fail until it is trained")
    shutil.copyfile("scratchpad/colab_train_mlp.py", out / "colab_train_mlp.py")
    shutil.copyfile("scratchpad/colab_train_set.py", out / "colab_train_set.py")
    if POOLED.is_file():
        shutil.copyfile(POOLED, out / POOLED.name)
        print(f"staged {POOLED.name} (pooled corpus for --dataset)")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    mb = Path(archive).stat().st_size / (1024 * 1024)
    print(f"\nwrote {archive} ({mb:.1f} MB) — upload this one file to Colab")
    print("then in a GPU runtime:  !unzip -q colab_export.zip && python colab_train_mlp.py --h 48")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
