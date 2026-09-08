"""Build + validate an imitation (rung-6) Kaggle submission.

Wires a BC weights file + its deck into the bundle. The default
`ptcg build-submission` is left untouched (still greedy-v5): only this script
ships the weights. Args (all optional, default to imitation-v1):

    python scratchpad/build_imitation.py [deck.csv] [weights.json] [out_name]

  imitation-v1:  decks/greengreenpurple.csv  data/models/bc_650_v1.json
  imitation-v2:  decks/lucario800.csv        data/models/bc_800_v1.json  imitation-v2
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from ptcg_ai.config import load_config
from ptcg_ai.submission.builder import build_submission
from ptcg_ai.submission.validate import smoke_test_entrypoint, validate_submission


def main(argv: list[str]) -> int:
    deck = Path(argv[0]) if len(argv) > 0 else Path("decks/greengreenpurple.csv")
    weights = Path(argv[1]) if len(argv) > 1 else Path("data/models/bc_650_v1.json")
    out = argv[2] if len(argv) > 2 else "imitation-v1"

    base = load_config()
    config = dataclasses.replace(
        base, paths=dataclasses.replace(base.paths, deck_path=deck)
    )
    tarball = build_submission(config, output_name=out, weights_path=weights)
    size_mib = tarball.stat().st_size / (1024 * 1024)
    print(f"built {tarball} ({size_mib:.1f} MiB / {config.submission.size_limit_mib} MiB limit)")
    print(f"  deck={deck}  weights={weights}")

    problems = validate_submission(tarball, config.submission.size_limit_mib)
    problems += smoke_test_entrypoint(tarball)
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {tarball} passes structural checks and the entrypoint smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
