"""Head-to-head: a screened clone vs imitation-v1 over the SAME diverse field.

The screen's clonability number (offline MAIN accuracy) is a PROXY. This answers the
real question directly — "does this clone actually play better than our champion?" —
by running both over the identical greedy-piloted field (field_gauntlet) and reporting
the paired per-deck delta with a bootstrap CI, exactly like the M8.1 promotion gate.

Worth running even when the screen says DISCARD: the open hypothesis is that cloning a
much stronger pilot (1052 elo vs v1's 661 teacher) could beat v1 even at lower fidelity.
M8 (an 800-elo clone) and M15 (kenN2439 at 0.770 fidelity, −0.214 vs v1) say otherwise,
but those were different teachers — this measures it instead of assuming.

The baseline defaults to imitation-v1 but can be ANY clone, so two challengers can be
compared to each other on the same footing (M22: "does ITF_Esys_Kasu beat the Kangaskhan
clone?"). Both sides always play the identical field in the SAME process, so field noise
is shared and the per-deck pairing is meaningful.

Run:  uv run --group dev python scratchpad/head_to_head.py <deck_csv> <weights_json> [n]
      ... --baseline <deck_csv> <weights_json> [--label NAME] [--baseline-label NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptcg_ai.imitation import deck_profiles as dp

from field_gauntlet import _cards, _paired_delta, extract_field, run_candidate

V1_DECK = "decks/greengreenpurple.csv"
V1_WEIGHTS = "data/models/bc_650_v1.json"


def _ensure_profile(deck_csv: str, weights: str, cards) -> str:
    """Make sure the profile the WEIGHTS were trained under is resolvable.

    The payload names its own profile. Hand-authored ones (TR_650, KANGASKHAN_1052, ...)
    are already in ``dp.PROFILES``; screen-built ``GENERIC_<SLUG>`` ones are not, and must
    be rebuilt here (``build_generic_profile`` is deterministic, so this reproduces the
    exact profile the screen trained against). Registering blind would be wrong — it would
    silently shadow nothing useful and hide a genuine profile mismatch.
    """
    name = json.loads(Path(weights).read_text(encoding="utf-8")).get("profile") or "TR_650"
    if name in dp.PROFILES:
        return name
    deck = [int(x) for x in Path(deck_csv).read_text(encoding="utf-8").split()]
    prof = dp.build_generic_profile(name, tuple(deck), cards)
    dp.PROFILES[prof.name] = prof
    return prof.name


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("deck_csv")
    ap.add_argument("weights")
    ap.add_argument("n", nargs="?", type=int, default=6, help="games per field deck")
    ap.add_argument("--baseline", nargs=2, metavar=("DECK_CSV", "WEIGHTS"),
                    default=[V1_DECK, V1_WEIGHTS],
                    help="what to compare against (default: imitation-v1)")
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--baseline-label", dest="baseline_label", default=None)
    args = ap.parse_args()

    base_deck, base_weights = args.baseline
    base_label = args.baseline_label or (
        "v1" if base_weights == V1_WEIGHTS else Path(base_weights).stem)

    sdk, cards = _cards()
    prof_c = _ensure_profile(args.deck_csv, args.weights, cards)
    prof_b = _ensure_profile(base_deck, base_weights, cards)
    field = extract_field(cards)
    print(f"field: {len(field)} decks | n={args.n}/deck")
    print(f"  {args.label:<12} profile={prof_c}")
    print(f"  {base_label:<12} profile={prof_b}")

    cand = run_candidate(args.label, "imitation", args.deck_csv, args.weights,
                         field, args.n, sdk, cards)
    base = run_candidate(base_label, "imitation", base_deck, base_weights,
                         field, args.n, sdk, cards)

    mean, lo, hi, wins, losses, k = _paired_delta(cand, base)
    c_macro = sum(cand.values()) / max(len(cand), 1)
    b_macro = sum(base.values()) / max(len(base), 1)
    print(f"\n{'='*68}")
    print(f"{args.label:<14} macro WR over field: {c_macro:.3f}")
    print(f"{base_label:<14} macro WR over field: {b_macro:.3f}")
    print(f"paired delta ({args.label} - {base_label}): {mean:+.3f}  "
          f"90%CI [{lo:+.3f},{hi:+.3f}]  decks better/worse = {wins}/{losses} of {k}")
    verdict = (f"BEATS {base_label}" if lo > 0 else
               f"ties {base_label} (CI spans 0)" if hi > 0 else
               f"LOSES to {base_label}")
    print(f"VERDICT: {args.label} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
