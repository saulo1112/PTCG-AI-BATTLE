"""Build / read the behavior-cloning decision dataset (dev-only, M7-1).

Materializes the target player's logged decisions (see
:mod:`~ptcg_ai.imitation.kaggle_replay`) to a gzip-JSONL file — one decision per
line — so the featurizer and trainer never re-read the 325 source replays and
so a feature-set change re-derives from disk. Rows are split *by game* for a
leak-free train/val boundary.
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ptcg_ai.imitation.kaggle_replay import ReplayDecision, load_replay_dir
from ptcg_ai.observation.models import SelectContextKind

_ROW_KEYS = (
    "game_id", "seat", "step_index", "raw_observation", "action", "won",
    "context", "select_type", "min_count", "max_count", "n_options",
)


@dataclass(frozen=True)
class DatasetSummary:
    """Aggregate counts from a dataset build (printed as the M7-1 exit table)."""

    n_games: int
    n_rows: int
    n_illegal: int
    context_counts: dict[str, int]

    def render(self) -> str:
        lines = [
            f"games={self.n_games}  rows={self.n_rows}  illegal_actions={self.n_illegal}",
            "context                         rows",
            "-" * 40,
        ]
        for ctx, count in sorted(self.context_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"{ctx:<28} {count:>8}")
        return "\n".join(lines)


def _context_name(ctx: int) -> str:
    try:
        return SelectContextKind(ctx).name
    except ValueError:  # pragma: no cover - TolerantEnum never raises, but be safe
        return f"CTX_{ctx}"


def build_decision_dataset(
    log_dir: Path, player_name: "str | Iterable[str]", out_path: Path
) -> DatasetSummary:
    """Write ``player_name``'s decisions from ``log_dir`` to gzip-JSONL.

    Illegal logged actions (should be zero) are counted but still written, so
    the trainer can filter them explicitly rather than silently dropping data.

    ``player_name`` may be a SET of display names. A Kaggle team can rename itself,
    so the name is not a stable key -- see `kaggle_replay.player_seats`. Passing one
    name when the team used three silently drops the other two eras' games and writes
    an undersized corpus with no error.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    games: set[str] = set()
    context_counts: collections.Counter[str] = collections.Counter()
    n_rows = n_illegal = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for dec in load_replay_dir(log_dir, player_name):
            games.add(dec.game_id)
            context_counts[_context_name(dec.context)] += 1
            if not dec.legal:
                n_illegal += 1
            n_rows += 1
            fh.write(json.dumps({k: getattr(dec, k) for k in _ROW_KEYS}, separators=(",", ":")))
            fh.write("\n")
    return DatasetSummary(
        n_games=len(games), n_rows=n_rows, n_illegal=n_illegal,
        context_counts=dict(context_counts),
    )


def read_decision_dataset(path: Path) -> Iterator[ReplayDecision]:
    """Stream :class:`ReplayDecision` rows back from a built dataset."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ReplayDecision(**json.loads(line))


def split_by_game(
    rows: list[ReplayDecision], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[ReplayDecision], list[ReplayDecision]]:
    """Partition rows into (train, val) with every game's rows on one side.

    Uses a deterministic hash of ``game_id`` (stdlib ``zlib.crc32``, stable
    across runs and machines — unlike salted ``hash()``), so the split is
    reproducible without materializing a per-game shuffle.
    """
    threshold = int(val_fraction * (2**32))
    train: list[ReplayDecision] = []
    val: list[ReplayDecision] = []
    for row in rows:
        key = f"{seed}:{row.game_id}".encode("utf-8")
        (val if zlib.crc32(key) < threshold else train).append(row)
    return train, val


def _main(argv: list[str]) -> None:
    if len(argv) < 3:
        print("usage: python -m ptcg_ai.imitation.dataset <log_dir> <player[,player2,...]> <out.jsonl.gz>")
        raise SystemExit(2)
    log_dir, player, out = Path(argv[0]), argv[1], Path(argv[2])
    # Comma-separated accepts a renamed team's full name set in one argument.
    names = [n.strip() for n in player.split(",") if n.strip()]
    summary = build_decision_dataset(log_dir, names if len(names) > 1 else names[0], out)
    print(f"wrote {out}")
    print(summary.render())


if __name__ == "__main__":
    _main(sys.argv[1:])
