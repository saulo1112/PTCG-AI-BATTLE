"""Render an analysis dict to JSON, CSV tables, and a Markdown summary.

Every headline number carries n and a 95% CI so reports answer "is this
enough games?" by themselves (docs/methodology.md). Reports must always name
the generating policy — random-play numbers describe the decision space,
not competent play.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from ptcg_ai.cards.database import CardDatabase

#: Engine finish reasons (battle_flow.md).
END_REASON_NAMES = {
    "1": "prizes taken",
    "2": "deck-out",
    "3": "no active Pokémon",
    "4": "card effect",
    "None": "unknown/aborted",
}


def write_reports(
    result: Mapping[str, Any],
    out_dir: Path,
    header: Mapping[str, Any] | None = None,
    cards: CardDatabase | None = None,
) -> None:
    """Write ``report.json``, ``report.md`` and ``tables/*.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"header": dict(header or {}), "analysis": result}
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    _write_csvs(result, out_dir / "tables")
    (out_dir / "report.md").write_text(
        render_markdown(result, header=header, cards=cards), encoding="utf-8"
    )


def render_markdown(
    result: Mapping[str, Any],
    header: Mapping[str, Any] | None = None,
    cards: CardDatabase | None = None,
) -> str:
    lines: list[str] = ["# Battle Analysis Report", ""]
    if header:
        for key, value in header.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    lines.append(
        f"Games: **{result['n_games']}** | decisions: **{result['n_decisions']}**. "
        "All intervals are 95% CIs; decision-level stats use per-game means "
        "(see docs/methodology.md)."
    )

    lines += ["", "## Game length", "", "| metric | n | mean | 95% CI | median | max |", "|---|---:|---:|---|---:|---:|"]
    for label, key in (("decisions", "decisions"), ("turns", "turns"), ("duration (s)", "duration_s")):
        m = result["game_length"][key]
        lines.append(
            f"| {label} | {m['n']} | {m['mean']} | {m['ci95_low']}–{m['ci95_high']} "
            f"| {m['median']} | {m['max']} |"
        )

    lines += ["", "## End reasons", "", "| reason | games | share | 95% CI |", "|---|---:|---:|---|"]
    for key, row in result["end_reasons"].items():
        name = END_REASON_NAMES.get(key, key)
        lines.append(
            f"| {name} | {row['count']} | {row['share']:.1%} "
            f"| {row['ci95_low']:.1%}–{row['ci95_high']:.1%} |"
        )

    lines += ["", "## Winners (player index)", "", "| winner | games | share | 95% CI |", "|---|---:|---:|---|"]
    for key, row in result["winners"].items():
        label = "draw" if key == "None" else f"P{key}"
        lines.append(
            f"| {label} | {row['count']} | {row['share']:.1%} "
            f"| {row['ci95_low']:.1%}–{row['ci95_high']:.1%} |"
        )

    branching = result["branching"]["per_game_mean"]
    lines += [
        "",
        "## Branching factor",
        "",
        f"Per-game mean legal options: **{branching['mean']}** "
        f"(95% CI {branching['ci95_low']}–{branching['ci95_high']}, n={branching['n']} games); "
        f"pooled max: **{result['branching']['pooled_max']}**.",
        "",
        "## Decision contexts (top 20 by frequency)",
        "",
        "| context | count | branching mean | median | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in list(result["contexts"].items())[:20]:
        lines.append(
            f"| {name} | {stats['count']} | {stats['branching_mean']} "
            f"| {stats['branching_median']} | {stats['branching_max']} |"
        )

    lines += ["", "## Options by select kind (available vs chosen)", ""]
    for select_kind, kinds in result["options_by_select_kind"].items():
        lines += [f"**{select_kind}**", "", "| option kind | available | chosen |", "|---|---:|---:|"]
        for option_kind, row in kinds.items():
            lines.append(f"| {option_kind} | {row['available']} | {row['chosen']} |")
        lines.append("")

    events = result["events"]
    lines += ["## Engine events (counting-viewer convention)", "", "| event | total | per turn |", "|---|---:|---:|"]
    for name, count in events["counts"].items():
        per_turn = events["per_turn"].get(name, "")
        lines.append(f"| {name} | {count} | {per_turn} |")
    if events["partial_games"]:
        lines.append(f"\n> {events['partial_games']} game(s) lacked a terminal observation (v1 episodes) — tails missing.")

    latency = result["latency_ms"]
    lines += [
        "",
        "## Decision latency (agent-side, parse included)",
        "",
        f"n={latency['n']} | p50 {latency['p50']} ms | p95 {latency['p95']} ms | max {latency['max']} ms",
        "",
        "## Card presence by zone (top 15; sampled at first MAIN decision per turn)",
        "",
    ]
    for zone, counts in result["zones"]["top_cards"].items():
        lines += [f"**{zone}** ({result['zones']['samples']} turn samples)", ""]
        lines += ["| card | appearances |", "|---|---:|"]
        for card_id, count in list(counts.items())[:15]:
            name = cards.name(int(card_id)) if cards is not None else f"#{card_id}"
            lines.append(f"| {name} | {count} |")
        lines.append("")

    return "\n".join(lines)


def _write_csvs(result: Mapping[str, Any], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    with (tables_dir / "contexts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["context", "count", "branching_mean", "branching_median", "branching_max"])
        for name, stats in result["contexts"].items():
            writer.writerow([
                name, stats["count"], stats["branching_mean"],
                stats["branching_median"], stats["branching_max"],
            ])

    with (tables_dir / "options_by_select_kind.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["select_kind", "option_kind", "available", "chosen"])
        for select_kind, kinds in result["options_by_select_kind"].items():
            for option_kind, row in kinds.items():
                writer.writerow([select_kind, option_kind, row["available"], row["chosen"]])

    with (tables_dir / "end_reasons.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reason", "name", "count", "share", "ci95_low", "ci95_high"])
        for key, row in result["end_reasons"].items():
            writer.writerow([
                key, END_REASON_NAMES.get(key, key), row["count"],
                row["share"], row["ci95_low"], row["ci95_high"],
            ])

    with (tables_dir / "events.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event", "count", "per_turn"])
        for name, count in result["events"]["counts"].items():
            writer.writerow([name, count, result["events"]["per_turn"].get(name, "")])

    with (tables_dir / "zones.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["zone", "card_id", "appearances"])
        for zone, counts in result["zones"]["top_cards"].items():
            for card_id, count in counts.items():
                writer.writerow([zone, card_id, count])
