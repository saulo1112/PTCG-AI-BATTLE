"""Load one player's decisions out of Kaggle replay JSONs (dev-only, M7-1).

A Kaggle replay is ``{info, rewards, statuses, steps, ...}`` where ``steps[i]``
holds one cell per seat: ``{action, observation, reward, status}``. Two facts
(verified in docs/replay_analysis.md and the M7-0 spike) drive this loader:

- **Action lag +1.** The action answering ``steps[i][seat].observation`` is
  stored at ``steps[i+1][seat].action`` — a list of option indices, or a 60-int
  deck list on the first decision. We pair observation ``i`` with action ``i+1``.
- **Only ACTIVE cells hold a fresh decision.** INACTIVE / DONE cells carry a
  stale or ``None`` observation; training on them would be garbage. We keep only
  ``status == "ACTIVE"`` cells that carry a non-empty ``select``.

The heavy ``logs`` and opaque ``search_begin_input`` keys are dropped from the
stored observation: the featurizer never reads them and they dominate the row
size. Everything the featurizer needs (``select`` + ``current``) is preserved
verbatim so features can be re-derived offline without re-reading the replays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Kaggle per-seat step status where the seat owns a fresh decision.
_ACTIVE = "ACTIVE"
#: Deck-submission action length (the first decision each game).
_DECK_LEN = 60


@dataclass(frozen=True)
class ReplayDecision:
    """One logged decision by the target player.

    ``raw_observation`` is the seat's POV observation (``select`` + ``current``,
    ``logs``/``search_begin_input`` stripped) — the exact live-agent input shape.
    ``action`` is the chosen option indices into ``select.option``. ``won`` is
    the game's terminal outcome for this seat (``rewards[seat] > 0``).
    """

    game_id: str
    seat: int
    step_index: int
    raw_observation: dict[str, Any]
    action: list[int]
    won: bool
    context: int
    select_type: int
    min_count: int
    max_count: int
    n_options: int

    @property
    def legal(self) -> bool:
        """Whether the logged action is a legal answer to its own select."""
        if len(set(self.action)) != len(self.action):
            return False
        if any(not (0 <= a < self.n_options) for a in self.action):
            return False
        return self.min_count <= len(self.action) <= self.max_count


def player_seats(data: dict, player_name: "str | Iterable[str]") -> list[int]:
    """Seats occupied by ``player_name`` (usually one; both in a self-match).

    Accepts a SET of names, not just one, because a Kaggle team can rename itself and
    the display name is therefore **not a stable key**. M49 caught this live: the
    teacher's submission 54773249 appears as ``Yushin Ito`` in the 2026-08-02 snapshot
    and as ``AlphaStarmie`` / ``AlphaTCG`` in later episodes -- same team, same
    submission, three strings. Matching on one of them silently yields ZERO seats for
    the others, which would rebuild a corpus that looks fine and is missing a third of
    its games, with no error raised anywhere.

    The stable identifier is ``submissionId``, which lives in the Kaggle API's episode
    listing rather than in the replay JSON, so callers resolve the name set from there
    (``scratchpad/m49_resolve_teacher_names.py``) and pass all of it here.

    A plain string still behaves exactly as before.
    """
    if isinstance(player_name, str):
        wanted = {player_name.lower()}
    else:
        wanted = {n.lower() for n in player_name}
    agents = data.get("info", {}).get("Agents", [])
    return [i for i, a in enumerate(agents) if (a.get("Name") or "").lower() in wanted]


def _strip_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the featurizer consumes."""
    return {"select": obs.get("select"), "current": obs.get("current")}


def iter_player_decisions(path: Path, player_name: "str | Iterable[str]") -> Iterator[ReplayDecision]:
    """Yield every ACTIVE-with-select decision ``player_name`` made in one replay.

    ``game_id`` is the file stem (stable + unique), so game-level dataset splits
    never leak a game across train/val even for the self-match (both seats share
    the same ``game_id``).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    game_id = path.stem
    steps = data.get("steps", [])
    rewards = data.get("rewards") or []
    for seat in player_seats(data, player_name):
        won = seat < len(rewards) and (rewards[seat] or 0) > 0
        for i in range(len(steps) - 1):
            row = steps[i]
            nxt = steps[i + 1]
            if seat >= len(row) or seat >= len(nxt):
                continue
            cell = row[seat]
            if not cell or cell.get("status") != _ACTIVE:
                continue
            obs = cell.get("observation") or {}
            select = obs.get("select")
            options = select.get("option") if select else None
            if not options:
                continue
            action = nxt[seat].get("action")
            if not isinstance(action, list) or len(action) == _DECK_LEN:
                continue  # deck submission or non-index action
            yield ReplayDecision(
                game_id=game_id,
                seat=seat,
                step_index=i,
                raw_observation=_strip_observation(obs),
                action=[int(a) for a in action],
                won=won,
                context=int(select.get("context", -1)),
                select_type=int(select.get("type", -1)),
                min_count=int(select.get("minCount", 1)),
                max_count=int(select.get("maxCount", 1)),
                n_options=len(options),
            )


def load_replay_dir(log_dir: Path, player_name: "str | Iterable[str]") -> Iterator[ReplayDecision]:
    """Yield ``player_name``'s decisions across every ``*.json`` in ``log_dir``.

    ``player_name`` may be a set of names -- see `player_seats` for why that matters."""
    for path in sorted(Path(log_dir).glob("*.json")):
        yield from iter_player_decisions(path, player_name)
