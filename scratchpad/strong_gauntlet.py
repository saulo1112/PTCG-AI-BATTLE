"""Strong-opponent gauntlet (M12/M13) -- the instrument the greedy gauntlet lacked.

M11's field gauntlet saturates (~90%) because opponents are greedy-piloted; it
can't resolve a subtle candidate difference. This pits each candidate against
CLONE-piloted opponents (the strong pilots built in M8/M9), where matchups are
competitive (0.5-0.85, not 0.9), so a real difference between v1 and v1.2 shows.

Each candidate plays each strong opponent n games (swapped sides). We report the
per-matchup score rate + Wilson CI for v1 and v1.2, a per-matchup delta with a
two-proportion normal-approx SE, an equal-weight POOLED delta D +/- 1.96*SE_D,
and a sign-consistency count. The mirror row (candidate vs v1-piloted TR) IS the
head-to-head vs v1: v1-vs-itself is 0.5 by construction (identical weights, sides
swapped, n even), so the v1 baseline for that row is 0.5 exactly.

Run:  uv run python scratchpad/strong_gauntlet.py [n]   (default n=300)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from arena_m8 import run

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import deck_profiles as dp

# hand-authored strong opponents (M8/M9): (label, deck_csv, weights)
OPPONENTS = [
    ("v1-mirror(TR)", "decks/greengreenpurple.csv", "data/models/bc_650_v1.json"),
    ("v2.1(lucario)", "decks/lucario800.csv", "data/models/bc_800_v2.json"),
    ("v3(bellibolt)", "decks/kenn2439.csv", "data/models/bc_940_v1.json"),
    ("cinderace", "decks/yoshiki.csv", "data/models/bc_cinderace_probe.json"),
]
# M14: screen clones of fresh strong teachers. They pilot far above greedy (which
# predicts them at only 0.24-0.39) and are diverse -- a closer proxy to the 600-700
# ladder field than the 4 hand-authored foes. Their payloads carry a GENERIC_<SLUG>
# profile that must be rebuilt + registered in-process before ImitationPolicy loads
# them (build_generic_profile is deterministic, so it reproduces the trained profile).
SCREEN_SLUGS = ["budew", "eduardorochadeandrade", "legendbrothers", "m093jp"]
for _slug in SCREEN_SLUGS:
    OPPONENTS.append((_slug, f"decks/{_slug}.csv", f"data/models/{_slug}_screen.json"))

CANDIDATES = [
    ("v1", "decks/greengreenpurple.csv", "data/models/bc_650_v1.json"),
    ("v1.2", "decks/greengreenpurple.csv", "data/models/bc_650_v2.json"),
]
MIRROR = "v1-mirror(TR)"


def _register_screen_profiles() -> None:
    """Rebuild + register each screen opponent's GENERIC_<SLUG> profile in-process,
    matching scratchpad/quick_screen.py so ImitationPolicy can load its weights."""
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    for slug in SCREEN_SLUGS:
        deck = [int(x) for x in Path(f"decks/{slug}.csv").read_text().split()]
        prof = dp.build_generic_profile(f"GENERIC_{slug.upper()}", tuple(deck), cards)
        dp.PROFILES[prof.name] = prof


def _binom_var(p: float, n: int) -> float:
    """Variance of a score rate under a two-outcome normal approx."""
    return p * (1.0 - p) / max(n, 1)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    _register_screen_profiles()
    # results[(candidate, opponent)] = (score_rate, wilson_lo, wilson_hi, n)
    results: dict[tuple[str, str], tuple[float, float, float, int]] = {}
    for clabel, cdeck, cw in CANDIDATES:
        for olabel, odeck, ow in OPPONENTS:
            if olabel == MIRROR and clabel == "v1":
                continue  # v1 vs v1 is 0.5 by construction -- substituted below
            stats = run("imitation", cdeck, "imitation", odeck,
                        f"{clabel} vs {olabel}", n=n, wa=cw, wb=ow)
            lo, hi = stats.wilson_interval()
            results[(clabel, olabel)] = (stats.score_rate, lo, hi, stats.n)

    print("\n" + "=" * 92)
    print("STRONG-OPPONENT GAUNTLET SUMMARY (candidate score rate vs each clone foe)")
    print("=" * 92)
    print(f"{'opponent':<16}{'v1':>8}{'v1 95%CI':>16}{'v1.2':>8}"
          f"{'v1.2 95%CI':>16}{'delta':>9}{'+/-1.96SE':>12}")
    print("-" * 92)

    deltas: list[float] = []
    ses: list[float] = []
    excluded: list[str] = []
    for olabel, *_ in OPPONENTS:
        v12 = results.get(("v1.2", olabel))
        if v12 is None:
            continue
        p12, lo12, hi12, n12 = v12
        if olabel == MIRROR:
            p1, lo1, hi1, n1 = 0.5, 0.5, 0.5, n12  # v1-vs-v1 by construction
            note = "  (v1 side=0.5 by construction)"
        else:
            v1 = results.get(("v1", olabel))
            p1, lo1, hi1, n1 = v1
            note = ""
        # saturation guard (M11 lesson): both >0.90 => no resolution, drop from pool
        if p1 > 0.90 and p12 > 0.90:
            excluded.append(olabel)
            note += "  [SATURATED-excluded]"
        d = p12 - p1
        se = math.sqrt(_binom_var(p12, n12) + _binom_var(p1, n1))
        if olabel not in excluded:
            deltas.append(d)
            ses.append(se)
        ci1 = f"[{lo1:.3f},{hi1:.3f}]"
        ci12 = f"[{lo12:.3f},{hi12:.3f}]"
        print(f"{olabel:<16}{p1:>8.3f}{ci1:>16}{p12:>8.3f}{ci12:>16}"
              f"{d:>+9.3f}{1.96*se:>12.3f}{note}")

    print("-" * 92)
    if deltas:
        K = len(deltas)
        D = sum(deltas) / K
        se_D = math.sqrt(sum(s * s for s in ses)) / K
        lo, hi = D - 1.96 * se_D, D + 1.96 * se_D
        pos = sum(1 for d in deltas if d > 0)
        print(f"POOLED delta (v1.2 - v1) over {K} matchups: {D:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"sign consistency: {pos}/{K} matchups favor v1.2")
        if excluded:
            print(f"excluded (saturated): {', '.join(excluded)}")
        if lo > 0:
            verdict = "v1.2 BETTER (CI-lo > 0)"
        elif hi < 0:
            verdict = "v1.2 WORSE (CI-hi < 0)"
        else:
            verdict = "INCONCLUSIVE (CI spans 0)"
        print(f"\nGATE A verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
