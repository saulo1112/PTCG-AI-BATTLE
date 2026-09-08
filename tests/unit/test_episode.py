"""Episode JSONL round-trip (ADR-0008; format v2 per ADR-0011)."""

import json
from pathlib import Path

from ptcg_ai.replay.episode import load_episode, save_episode
from ptcg_ai.replay.recorder import EpisodeRecorder
from tests.conftest import make_raw_obs


def _record_episode():
    recorder = EpisodeRecorder(metadata={"note": "unit test"})
    raw = make_raw_obs()
    recorder.on_step(raw, [0], player=0, elapsed_ms=0.5)
    recorder.on_step(raw, [1, 2], player=1, elapsed_ms=1.25)
    terminal = make_raw_obs()
    terminal["current"]["result"] = 0
    recorder.set_final(terminal)
    return recorder.finalize({"result": 0, "winner": 0, "reason": 1, "duration_s": 2.5}), raw, terminal


def test_round_trip_v2(tmp_path: Path) -> None:
    episode, raw, terminal = _record_episode()
    path = tmp_path / "episode.jsonl"
    save_episode(episode, path)
    loaded = load_episode(path)

    assert loaded.metadata == {"note": "unit test"}
    assert loaded.outcome == {"result": 0, "winner": 0, "reason": 1, "duration_s": 2.5}
    assert len(loaded.steps) == 2
    assert loaded.steps[0].raw_obs == raw
    assert loaded.steps[0].elapsed_ms == 0.5
    assert loaded.steps[1].action == [1, 2]
    assert loaded.steps[1].player == 1
    assert loaded.final_obs == terminal

    meta = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert meta["version"] == 2


def test_gzip_round_trip(tmp_path: Path) -> None:
    episode, _, terminal = _record_episode()
    path = tmp_path / "episode.jsonl.gz"
    save_episode(episode, path)
    loaded = load_episode(path)
    assert len(loaded.steps) == 2
    assert loaded.final_obs == terminal
    # Actually compressed: gzip magic bytes.
    assert path.read_bytes()[:2] == b"\x1f\x8b"


def test_v1_files_still_load(tmp_path: Path) -> None:
    """v1 layout: no version, no final line, no elapsed_ms."""
    raw = make_raw_obs()
    path = tmp_path / "v1.jsonl"
    lines = [
        json.dumps({"kind": "meta", "metadata": {"old": True}}),
        json.dumps({"kind": "step", "player": 0, "action": [0], "raw_obs": raw}),
        json.dumps({"kind": "outcome", "outcome": {"result": 1}}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = load_episode(path)
    assert loaded.metadata == {"old": True}
    assert loaded.steps[0].elapsed_ms is None
    assert loaded.final_obs is None
    assert loaded.outcome == {"result": 1}


def test_recorder_rejects_use_after_finalize(tmp_path: Path) -> None:
    recorder = EpisodeRecorder()
    recorder.finalize(None)
    import pytest

    with pytest.raises(RuntimeError):
        recorder.on_step(make_raw_obs(), [0], player=0)
    with pytest.raises(RuntimeError):
        recorder.set_final(make_raw_obs())
