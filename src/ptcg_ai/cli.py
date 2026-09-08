"""``ptcg`` command-line interface.

Subcommands cover the Phase 0/1 workflows: run local battles (with tracing),
inspect observations, capture test fixtures, run benchmarks, validate decks,
and build/validate the Kaggle submission tarball.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ptcg_ai.config import AppConfig, load_config
from ptcg_ai.logutil import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(profile=args.profile)
    setup_logging(config.logging.level, config.logging.log_dir)
    return args.handler(config, args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptcg", description=__doc__)
    parser.add_argument(
        "--profile", default="development", help="config profile name (configs/<name>.yaml)"
    )
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("battle", help="run local battles between two policies")
    p.add_argument("--games", type=int, default=None, help="number of games (default from config)")
    p.add_argument("--policy", default=None, help="policy for player A (default from config)")
    p.add_argument("--opponent", default=None, help="policy for player B (default: same)")
    p.add_argument("--trace", type=Path, default=None, help="write decision trace JSONL here")
    p.add_argument("--record", type=Path, default=None, help="write episode JSONL here")
    p.add_argument("--vis", type=Path, default=None, help="write engine visualizer JSON here")
    p.add_argument(
        "--verbose", action="store_true",
        help="render every decision: board, named legal actions, choice, timing",
    )
    p.set_defaults(handler=_cmd_battle)

    p = sub.add_parser("collect", help="run N games into an experiment dir (ADR-0012)")
    p.add_argument("--name", required=True, help="experiment name (data/experiments/<name>)")
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--policy", default=None, help="policy for side A (default from config)")
    p.add_argument("--opponent", default=None, help="policy for side B (default: same)")
    p.add_argument("--seed", type=int, default=None, help="policy seed base (engine RNG not seedable)")
    p.set_defaults(handler=_cmd_collect)

    p = sub.add_parser("analyze", help="aggregate an experiment dir into reports")
    p.add_argument("dir", type=Path, help="experiment directory (from `ptcg collect`)")
    p.set_defaults(handler=_cmd_analyze)

    p = sub.add_parser("show-episode", help="summarize a recorded episode")
    p.add_argument("file", type=Path, help="episode .jsonl or .jsonl.gz")
    p.add_argument("--step", type=int, default=None, help="render decision N in full")
    p.set_defaults(handler=_cmd_show_episode)

    p = sub.add_parser("bench", help="run benchmarks (ADR-0006)")
    p.add_argument("target", choices=["battle", "parser", "search", "all"])
    p.add_argument("--n", type=int, default=None, help="op count override")
    p.set_defaults(handler=_cmd_bench)

    p = sub.add_parser("show-obs", help="pretty-print a captured observation JSON")
    p.add_argument("file", type=Path)
    p.add_argument("--logs", action="store_true", help="also print the event log summary")
    p.set_defaults(handler=_cmd_show_obs)

    p = sub.add_parser("capture-fixtures", help="run one battle and save observation fixtures")
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/observations"))
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(handler=_cmd_capture_fixtures)

    p = sub.add_parser("validate-deck", help="check a deck.csv against the rules")
    p.add_argument("file", type=Path)
    p.set_defaults(handler=_cmd_validate_deck)

    p = sub.add_parser("build-submission", help="build the Kaggle submission tarball")
    p.add_argument("--out", default=None, help="output name (default from config)")
    p.set_defaults(handler=_cmd_build_submission)

    p = sub.add_parser("validate-submission", help="validate a submission tarball")
    p.add_argument("file", type=Path)
    p.add_argument("--no-smoke", action="store_true", help="skip the entrypoint smoke test")
    p.set_defaults(handler=_cmd_validate_submission)

    return parser


def _cmd_battle(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.agent.agent import PTCGAgent
    from ptcg_ai.agent.deck import load_deck
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.debug.trace import DecisionTracer, TracingPolicy
    from ptcg_ai.decision.registry import build_policy
    from ptcg_ai.environment.adapter import BattleEnvironment
    from ptcg_ai.environment.runner import BattleRunner
    from ptcg_ai.environment.sdk import load_sdk
    from ptcg_ai.replay.episode import save_episode
    from ptcg_ai.replay.recorder import EpisodeRecorder

    games = args.games if args.games is not None else config.battle.games
    sdk = load_sdk(config.paths.sdk_dir)
    cards = CardDatabase.from_sdk(sdk)
    deck_a = load_deck(config.paths.deck_path)
    deck_b = load_deck(config.paths.opponent_deck_path)

    policy_a = build_policy(config, deck=deck_a.as_list(), name=args.policy)
    policy_b = build_policy(config, deck=deck_b.as_list(), name=args.opponent or args.policy)
    tracer = DecisionTracer()
    if args.trace is not None or config.battle.trace:
        policy_a = TracingPolicy(policy_a, tracer)

    agent_a = PTCGAgent(policy_a, deck=deck_a, cards=cards)
    agent_b = PTCGAgent(policy_b, deck=deck_b, cards=cards)
    env = BattleEnvironment(config, sdk=sdk)
    runner = BattleRunner(env, max_decisions=config.battle.max_decisions)

    on_decision = None
    if args.verbose:
        from ptcg_ai.debug.describe import render_decision
        from ptcg_ai.observation.parser import ObservationParser

        verbose_parser = ObservationParser()
        counter = iter(range(10**9))

        def on_decision(raw_obs, action, player, elapsed_ms):  # type: ignore[misc]
            print(
                render_decision(
                    raw_obs, action, elapsed_ms, cards, verbose_parser,
                    decision_index=next(counter),
                )
            )

    for game in range(games):
        policy_a.on_battle_start()
        policy_b.on_battle_start()
        recorder = EpisodeRecorder({"game": game}) if args.record else None
        record = runner.run(
            agent_a, agent_b, deck_a.as_list(), deck_b.as_list(), recorder,
            on_decision=on_decision,
        )
        outcome = record.outcome
        print(
            f"game {game + 1}/{games}: winner={outcome.winner} "
            f"(result={outcome.result}, reason={outcome.reason}) "
            f"after {record.decisions} decisions"
        )
        if args.record and record.episode is not None:
            path = args.record.with_stem(f"{args.record.stem}_{game}") if games > 1 else args.record
            save_episode(record.episode, path)
            print(f"  episode -> {path}")

    if args.trace is not None:
        tracer.save_jsonl(args.trace)
        print(tracer.report())
        print(f"trace -> {args.trace}")
    if args.vis is not None:
        print("note: --vis captures only the last game's data and requires an open battle;")
        print("      visualizer export will be wired when the HTML viewer lands (roadmap).")
    return 0


def _cmd_collect(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.analytics.collect import run_collection

    experiment_dir = run_collection(
        config,
        name=args.name,
        games=args.games,
        policy_name=args.policy,
        opponent_name=args.opponent,
        seed=args.seed,
    )
    print(f"experiment -> {experiment_dir}")
    print(f"next: uv run ptcg analyze \"{experiment_dir}\"")
    return 0


def _cmd_analyze(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.analytics.analyze import analyze_experiment
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.environment.sdk import load_sdk, sdk_available

    cards = None
    if sdk_available(config.paths.sdk_dir):
        cards = CardDatabase.from_sdk(load_sdk(config.paths.sdk_dir))
    analyze_experiment(args.dir, cards=cards)
    print(f"reports -> {args.dir / 'report.md'} (+ report.json, tables/)")
    return 0


def _cmd_show_episode(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.debug.describe import render_decision
    from ptcg_ai.environment.sdk import load_sdk, sdk_available
    from ptcg_ai.replay.episode import load_episode

    episode = load_episode(args.file)
    cards = None
    if sdk_available(config.paths.sdk_dir):
        cards = CardDatabase.from_sdk(load_sdk(config.paths.sdk_dir))

    if args.step is not None:
        if not (0 <= args.step < len(episode.steps)):
            print(f"step {args.step} out of range (episode has {len(episode.steps)} decisions)")
            return 1
        step = episode.steps[args.step]
        print(
            render_decision(
                step.raw_obs, step.action, step.elapsed_ms, cards,
                decision_index=args.step,
            )
        )
        return 0

    outcome = episode.outcome or {}
    final_turn = None
    if episode.final_obs is not None:
        final_turn = (episode.final_obs.get("current") or {}).get("turn")
    print(f"episode: {args.file}")
    print(f"metadata: {dict(episode.metadata)}")
    print(
        f"decisions: {len(episode.steps)} | final turn: {final_turn} | "
        f"duration: {outcome.get('duration_s', '?')}s"
    )
    print(
        f"outcome: winner={outcome.get('winner')} result={outcome.get('result')} "
        f"reason={outcome.get('reason')}"
    )
    timed = [s.elapsed_ms for s in episode.steps if s.elapsed_ms is not None]
    if timed:
        print(f"decision latency ms: p50={sorted(timed)[len(timed) // 2]:.2f} max={max(timed):.2f}")
    print(f"inspect one decision: ptcg show-episode \"{args.file}\" --step N")
    return 0


def _cmd_bench(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.bench.harness import format_report, save_report
    from ptcg_ai.utils.paths import repo_root

    results = []
    if args.target in ("battle", "all"):
        from ptcg_ai.bench.bench_battle import run_battle_bench

        results.append(run_battle_bench(config, n_battles=args.n))
    if args.target in ("parser", "all"):
        from ptcg_ai.bench.bench_parser import run_parser_bench

        fixtures = repo_root() / "tests" / "fixtures" / "observations"
        results.append(run_parser_bench(config, fixtures, iterations=args.n))
    if args.target in ("search", "all"):
        from ptcg_ai.bench.bench_search import run_search_bench

        results.extend(run_search_bench(config, n_steps=args.n))

    print(format_report(results))
    out = config.bench.output_dir / f"bench_{args.target}.md"
    save_report(results, out)
    print(f"report -> {out}")
    return 0


def _cmd_show_obs(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.debug.inspect import format_observation, summarize_logs
    from ptcg_ai.environment.sdk import sdk_available, load_sdk
    from ptcg_ai.observation.parser import ObservationParser

    raw = json.loads(args.file.read_text(encoding="utf-8"))
    obs = ObservationParser().parse(raw)
    cards = None
    if sdk_available(config.paths.sdk_dir):
        cards = CardDatabase.from_sdk(load_sdk(config.paths.sdk_dir))
    print(format_observation(obs, cards))
    if args.logs and obs.logs:
        print("\n-- events since last decision --")
        print(summarize_logs(obs.logs, cards))
    return 0


def _cmd_capture_fixtures(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.agent.agent import PTCGAgent
    from ptcg_ai.agent.deck import load_deck
    from ptcg_ai.decision.random_policy import RandomPolicy
    from ptcg_ai.environment.adapter import BattleEnvironment
    from ptcg_ai.environment.sdk import load_sdk
    from ptcg_ai.observation.options import decision_signature
    from ptcg_ai.observation.parser import ObservationParser

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    sdk = load_sdk(config.paths.sdk_dir)
    deck = load_deck(config.paths.deck_path)
    parser = ObservationParser()

    # Synthetic Kaggle-side initial call (never occurs in local hosting).
    (out / "000_first_call.json").write_text(
        json.dumps({"select": None, "logs": [], "current": None}, indent=1), encoding="utf-8"
    )

    agents = (
        PTCGAgent(RandomPolicy(seed=args.seed)),
        PTCGAgent(RandomPolicy(seed=args.seed + 1)),
    )
    seen: set[str] = set()
    saved = 1
    with BattleEnvironment(config, sdk=sdk) as env:
        obs = env.start(deck.as_list(), deck.as_list())
        step = 0
        while env.result() is None and step < config.battle.max_decisions:
            parsed = parser.parse(obs)
            assert parsed.select is not None
            signature = decision_signature(parsed.select)
            if signature not in seen:
                seen.add(signature)
                name = f"{saved:03d}_{parsed.select.type.name}_{parsed.select.context.name}.json"
                (out / name).write_text(json.dumps(obs, indent=1), encoding="utf-8")
                saved += 1
            obs = env.select(agents[env.acting_player()](obs))
            step += 1
        (out / f"{saved:03d}_terminal.json").write_text(
            json.dumps(obs, indent=1), encoding="utf-8"
        )
    print(f"captured {saved + 1} fixtures ({len(seen)} decision signatures) -> {out}")
    return 0


def _cmd_validate_deck(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.agent.deck import load_deck, validate_deck
    from ptcg_ai.cards.database import CardDatabase
    from ptcg_ai.environment.sdk import load_sdk, sdk_available

    try:
        deck = load_deck(args.file)
    except ValueError as exc:
        print(f"INVALID: {exc}")
        return 1
    cards = None
    if sdk_available(config.paths.sdk_dir):
        cards = CardDatabase.from_sdk(load_sdk(config.paths.sdk_dir))
    else:
        print("note: SDK unavailable — structural checks only")
    problems = validate_deck(deck, cards)
    if problems:
        print("INVALID:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {args.file} is a valid 60-card deck")
    return 0


def _cmd_build_submission(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.submission.builder import build_submission

    tarball = build_submission(config, output_name=args.out)
    size_mib = tarball.stat().st_size / (1024 * 1024)
    print(f"built {tarball} ({size_mib:.1f} MiB / {config.submission.size_limit_mib} MiB limit)")
    print("run `ptcg validate-submission` before uploading.")
    return 0


def _cmd_validate_submission(config: AppConfig, args: argparse.Namespace) -> int:
    from ptcg_ai.submission.validate import smoke_test_entrypoint, validate_submission

    problems = validate_submission(args.file, config.submission.size_limit_mib)
    if not args.no_smoke:
        problems += smoke_test_entrypoint(args.file)
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {args.file} passes structural checks and the entrypoint smoke test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
