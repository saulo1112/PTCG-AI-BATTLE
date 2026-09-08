"""Per-episode record extraction (streaming-friendly).

One episode in, one :class:`EpisodeExtract` out; the caller aggregates and
discards. Conventions implemented here (normative text: docs/methodology.md):

- **Terminal observation is never a decision** (stale select quirk).
- **Event counting**: engine events are counted from the *counting viewer's*
  stream — the terminal observation's ``yourIndex`` — i.e. logs of steps
  where ``step.player == v`` plus the terminal observation's logs. Validated
  by tests/integration/test_log_convention.py.
- **Zone sampling**: card presence is sampled at the first MAIN/MAIN
  decision of each turn (canonical post-draw state; ``State.turn`` already
  identifies the acting player).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ptcg_ai.observation.models import OptionKind, SelectContextKind, SelectKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.replay.episode import Episode


@dataclass(frozen=True)
class DecisionRecord:
    """One decision's shape and choice, ready for aggregation."""

    turn: int
    player: int
    select_kind: str
    select_context: str
    n_options: int
    min_count: int
    max_count: int
    option_kinds: Counter
    chosen_kinds: Counter
    elapsed_ms: float | None


@dataclass(frozen=True)
class GameRecord:
    """Whole-game facts."""

    decisions: int
    final_turn: int | None
    duration_s: float | None
    end_reason: int | None
    winner: int | None
    result: int | None


@dataclass
class EpisodeExtract:
    """Everything the aggregator needs from one episode."""

    game: GameRecord
    decisions: list[DecisionRecord] = field(default_factory=list)
    #: LogKind name -> count, per the counting-viewer convention.
    event_counts: Counter = field(default_factory=Counter)
    #: zone name -> Counter[card_id]; sampled once per turn (see module doc).
    zone_counts: dict[str, Counter] = field(default_factory=dict)
    #: number of (turn) samples contributing to zone_counts.
    zone_samples: int = 0
    #: True when no terminal observation exists (v1 episode) and event
    #: counts fell back to viewer 0's stream (possibly missing the tail).
    events_partial: bool = False


def extract_episode(episode: Episode, parser: ObservationParser | None = None) -> EpisodeExtract:
    """Extract all Phase 1 records from one episode."""
    parser = parser if parser is not None else ObservationParser()

    final_turn: int | None = None
    counting_viewer = 0
    events_partial = True
    if episode.final_obs is not None:
        final_current = episode.final_obs.get("current") or {}
        final_turn = final_current.get("turn")
        counting_viewer = int(final_current.get("yourIndex", 0))
        events_partial = False

    outcome = episode.outcome or {}
    game = GameRecord(
        decisions=len(episode.steps),
        final_turn=final_turn,
        duration_s=outcome.get("duration_s"),
        end_reason=outcome.get("reason"),
        winner=outcome.get("winner"),
        result=outcome.get("result"),
    )

    extract = EpisodeExtract(game=game, events_partial=events_partial)
    zone_counts: dict[str, Counter] = {
        "hand": Counter(), "active": Counter(), "bench": Counter(), "discard": Counter()
    }
    sampled_turns: set[int] = set()

    for step in episode.steps:
        obs = parser.parse(step.raw_obs)
        select = obs.select
        state = obs.current
        if select is None or state is None:
            continue  # deck-submission shapes never occur in local episodes

        option_kinds = Counter(option.type.name for option in select.option)
        chosen_kinds = Counter(
            select.option[i].type.name for i in step.action if 0 <= i < len(select.option)
        )
        extract.decisions.append(
            DecisionRecord(
                turn=state.turn,
                player=step.player,
                select_kind=select.type.name,
                select_context=select.context.name,
                n_options=len(select.option),
                min_count=select.minCount,
                max_count=select.maxCount,
                option_kinds=option_kinds,
                chosen_kinds=chosen_kinds,
                elapsed_ms=step.elapsed_ms,
            )
        )

        # Event stream: counting viewer's steps only (see module docstring).
        if step.player == counting_viewer:
            for log in obs.logs:
                extract.event_counts[log.type.name] += 1

        # Zone sampling: first MAIN/MAIN decision of each turn.
        if (
            select.type is SelectKind.MAIN
            and select.context is SelectContextKind.MAIN
            and state.turn not in sampled_turns
        ):
            sampled_turns.add(state.turn)
            me = state.me
            if me.hand is not None:
                zone_counts["hand"].update(card.id for card in me.hand)
            for player in state.players:
                zone_counts["active"].update(
                    p.id for p in player.active if p is not None
                )
                zone_counts["bench"].update(p.id for p in player.bench)
                zone_counts["discard"].update(card.id for card in player.discard)

    # Terminal observation: complete the counting viewer's event stream.
    if episode.final_obs is not None:
        terminal = parser.parse(episode.final_obs)
        for log in terminal.logs:
            extract.event_counts[log.type.name] += 1

    extract.zone_counts = zone_counts
    extract.zone_samples = len(sampled_turns)
    return extract
