"""Loader + dataset round-trip and leak-free game split (M7-1, SDK-free)."""

from __future__ import annotations

import json
from pathlib import Path

from ptcg_ai.imitation.dataset import (
    build_decision_dataset,
    read_decision_dataset,
    split_by_game,
)
from ptcg_ai.imitation.kaggle_replay import ReplayDecision, iter_player_decisions

PLAYER = "target"


def _obs(context: int, n_options: int) -> dict:
    return {
        "select": {
            "type": 0, "context": context, "minCount": 1, "maxCount": 1,
            "option": [{"type": 7, "index": i} for i in range(n_options)],
        },
        "current": {"turn": 3, "yourIndex": 0, "players": []},
        "logs": [{"type": 2}],
        "search_begin_input": "AAAA",
    }


def _replay(seat: int, reward: int) -> dict:
    """Two decisions by ``seat``: obs at step i, action at step i+1 (lag +1)."""
    other = 1 - seat
    agents = [{"Name": "x"}, {"Name": "x"}]
    agents[seat] = {"Name": PLAYER}
    step = lambda a, s, obs=None: {"action": a, "status": s, "observation": obs or {}}
    steps = [
        # step 0: deck submission observation (select=None-ish), action next step
        [None, None],
        # step 1: active decision A (obs), its action lands in step 2
        [None, None],
        # step 2: carries action for step1 + a new active decision B
        [None, None],
        # step 3: carries action for step2
        [None, None],
    ]
    # seat cells
    steps[1][seat] = step(None, "ACTIVE", _obs(context=0, n_options=3))
    steps[2][seat] = step([2], "ACTIVE", _obs(context=7, n_options=4))  # action for step1 = [2]
    steps[3][seat] = step([1], "DONE", {})  # action for step2 = [1]
    # opponent cells (inactive / irrelevant)
    for s in range(4):
        if steps[s][other] is None:
            steps[s][other] = step([], "INACTIVE", {})
        if steps[s][seat] is None:
            steps[s][seat] = step([], "INACTIVE", {})
    rewards = [0, 0]
    rewards[seat] = reward
    return {"info": {"Agents": agents}, "rewards": rewards, "steps": steps}


def test_iter_pairs_observation_with_next_action(tmp_path: Path) -> None:
    f = tmp_path / "g1.json"
    f.write_text(json.dumps(_replay(seat=0, reward=1)), encoding="utf-8")
    decs = list(iter_player_decisions(f, PLAYER))
    assert [d.context for d in decs] == [0, 7]
    assert [d.action for d in decs] == [[2], [1]]  # +1 lag pairing
    assert all(d.won for d in decs)
    assert all(d.legal for d in decs)
    # logs/search_begin_input stripped; select+current kept.
    assert set(decs[0].raw_observation) == {"select", "current"}


def test_build_read_roundtrip_and_split(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for g in range(10):
        (log_dir / f"g{g}.json").write_text(
            json.dumps(_replay(seat=g % 2, reward=1 if g % 3 else -1)), encoding="utf-8"
        )
    out = tmp_path / "ds.jsonl.gz"
    summary = build_decision_dataset(log_dir, PLAYER, out)
    assert summary.n_games == 10
    assert summary.n_rows == 20  # 2 decisions/game
    assert summary.n_illegal == 0

    rows = list(read_decision_dataset(out))
    assert len(rows) == 20
    assert all(isinstance(r, ReplayDecision) for r in rows)

    train, val = split_by_game(rows, val_fraction=0.3, seed=0)
    assert len(train) + len(val) == 20
    # No game_id appears on both sides of the split.
    assert set(r.game_id for r in train).isdisjoint(r.game_id for r in val)
    # Deterministic across calls.
    assert split_by_game(rows, 0.3, 0)[0] == train
