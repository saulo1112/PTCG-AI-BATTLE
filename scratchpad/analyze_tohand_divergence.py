"""M34 Track B — the never-measured context: TO_HAND (what the teacher FETCHES).

M33 established the clone's remaining deficit is combo ASSEMBLY SPEED (Alakazam
active-with-{P} at turn 5.16 vs the teacher's 4.42) and card SELECTION -- not
attack timing. Every fidelity upgrade so far went to MAIN: the shipped payload
has MAIN = mlp_ensemble (0.780) but TO_HAND still LINEAR at 0.606 (greedy 0.512).

TO_HAND is where the ~16 tutors of this deck resolve (Dawn x4 fetch 3, Hilda x4,
Poke Pad x4, Buddy-Buddy Poffin x4) -- literally "which combo piece do I pull".
It is the most direct candidate for the assembly delay and has never been looked
at. This script measures it before anything is trained.

PRE-REGISTERED KILL GATE (G-0). If it fires, do NOT train a TO_HAND model.
  G-0a  early-turn (1-5) non-trivial set-match agreement must be < 0.75.
        At >= 0.75 the clone already fetches what the teacher fetches, so
        TO_HAND cannot explain a 0.9-turn assembly gap.
  G-0b  on the TRADEOFF subset (turn <= 5, >= 2 distinct card ids, and at least
        one COMBO option AND one non-COMBO option): require
            T - C >= 0.05   AND   McNemar p < 0.01
        where T/C are the teacher's / clone's COMBO pick rate on the SAME states.
        If |T-C| < 0.05, or C > T, "the clone under-fetches combo pieces" is
        FALSIFIED -- same shape as the measurement that killed M32 (18.0/18.1).
  G-0c  >= 1000 early-turn disagreements AND >= 0.8 early-turn TO_HAND
        disagreements per game. Below that, even a perfect fix moves too few
        decisions per game to shift an assembly turn.

Honest prior: G-0b is expected to FAIL. M33 sec.6 found the clone is already MORE
aggressive than the teacher on identical states, and sec.4 found their hand
composition indistinguishable. Twenty minutes here can save the whole cycle.

Three measurement traps this script is built to avoid (all real in this data):
  * ~39% of TO_HAND rows are prize picks -- informationless, must be dropped
    (features.is_prize_pick), exactly as training does.
  * ~55% of decisions contain duplicate options with the SAME card id, so
    agreement must compare card-id MULTISETS, not index sets.
  * ~25% of decisions have every option the same card id (trivially correct)
    and ~10% have a single option. The headline 0.606 is inflated; every number
    below is therefore reported twice: all decisions, and non-trivial only.
  * np.argsort is unstable while the live _rank_order breaks ties by lowest
    index -- we score through the real ImitationPolicy, so live behaviour is
    reproduced by construction.

READ-ONLY. Nothing is written, trained or uploaded. Run:
  uv run --group dev python scratchpad/analyze_tohand_divergence.py
  uv run --group dev python scratchpad/analyze_tohand_divergence.py --by-archetype
"""

from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

import ptcg_ai.imitation.features as F
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

import diagnose_mlp as dm

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")
WEIGHTS = dm.WEIGHTS

# The pieces that put "Alakazam active with {P}" on the board -- the milestone
# M33 measured as 0.9 turns late (docs/m33_yushin_deep_dive.md sec.3).
COMBO_IDS = {
    741: "Abra", 742: "Kadabra", 743: "Alakazam", 1079: "Rare Candy",
    5: "Basic {P} Energy", 13: "Enriching Energy", 19: "Telepath Psychic Energy",
}
# The teacher's #1 early fetch. Kept as its OWN bucket: folding the draw engine
# into "other" would make any combo-rate gap look larger than it is.
ENGINE_IDS = {305: "Dunsparce", 66: "Dudunsparce", 140: "Fezandipiti ex"}

EARLY_TURN_MAX = 5

# G-0 thresholds, fixed before the run.
G0A_AGREEMENT_MAX = 0.75
G0B_MIN_GAP = 0.05
G0B_MAX_P = 0.01
G0C_MIN_DISAGREEMENTS = 1000
G0C_MIN_PER_GAME = 0.8


def _bucket(cid: int | None) -> str:
    if cid is None:
        return "UNRESOLVED"
    if cid in COMBO_IDS:
        return "COMBO"
    if cid in ENGINE_IDS:
        return "ENGINE"
    return "OTHER"


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.1%}" if den else f"{num}/0 = n/a"


def _mcnemar(b: int, c: int) -> float:
    """Two-sided McNemar p-value on the discordant pairs (b, c).

    Exact binomial test against p=0.5 for small n; for large n the exact sum
    overflows a float, so we fall back to the chi-square form with Yates'
    continuity correction (they agree to several decimals well before the
    crossover).
    """
    n = b + c
    if n == 0:
        return 1.0
    if n <= 1000:
        k = min(b, c)
        tail = math.fsum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                                  - math.lgamma(n - i + 1) + n * math.log(0.5))
                         for i in range(k + 1))
        return min(1.0, 2.0 * tail)
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    # survival function of chi-square with 1 df = erfc(sqrt(chi2/2))
    return math.erfc(math.sqrt(chi2 / 2.0))


class Slice:
    """Agreement counters for one population of decisions."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.agree = 0
        self.total = 0

    def add(self, same: bool) -> None:
        self.total += 1
        self.agree += bool(same)

    @property
    def rate(self) -> float:
        return self.agree / self.total if self.total else 0.0

    def line(self) -> str:
        return f"{self.label:<34}{self.agree:>8}{self.total:>9}{self.rate:>9.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="cap decisions scored (0 = all ~11.5k; the full run is cheap)")
    ap.add_argument("--by-archetype", action="store_true",
                    help="also slice by opponent archetype (parses 1551 replays, slow)")
    ap.add_argument("--weights", default=WEIGHTS,
                    help="payload to score with (default: the shipped champion). Point this "
                         "at a candidate to confirm its TO_HAND head is actually live and "
                         "moving the combo pick rate C toward the teacher's T.")
    args = ap.parse_args()

    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    deck = [int(x) for x in dm.DECK_CSV.read_text(encoding="utf-8").split()]
    pol = ImitationPolicy(args.weights, deck=deck)
    print(f"scoring with: {args.weights}", flush=True)

    arch_map: dict[str, str] = {}
    if args.by_archetype:
        import linear_triage_matchup as ltm
        print("building archetype map (parsing replays, slow)...", flush=True)
        arch_map = ltm.build_archetype_map()

    print("loading dataset...", flush=True)
    rows = [r for r in read_decision_dataset(DATASET)
            if SelectContextKind(r.context) is SelectContextKind.TO_HAND]
    print(f"  {len(rows)} raw TO_HAND decisions", flush=True)
    if args.sample:
        rows = rows[: args.sample]

    slices = {
        "all": Slice("all decisions"),
        "all_nt": Slice("  non-trivial only"),
        "early": Slice(f"turns 1-{EARLY_TURN_MAX}"),
        "early_nt": Slice(f"  turns 1-{EARLY_TURN_MAX}, non-trivial"),
        "late": Slice(f"turns {EARLY_TURN_MAX + 1}+"),
        "late_nt": Slice(f"  turns {EARLY_TURN_MAX + 1}+, non-trivial"),
        "won": Slice("games the teacher WON"),
        "lost": Slice("games the teacher LOST"),
    }
    per_turn: dict[int, Slice] = {t: Slice(f"  turn {t}") for t in range(1, EARLY_TURN_MAX + 1)}
    by_arch: dict[str, Slice] = {}

    n_prize = n_single_opt = n_trivial = n_skipped = 0
    # bucket of what each side FETCHED, early turns only
    t_bucket: collections.Counter = collections.Counter()
    c_bucket: collections.Counter = collections.Counter()
    confusion: collections.Counter = collections.Counter()
    early_games: set[str] = set()
    early_disagreements = 0

    # G-0b tradeoff subset: a real choice between a combo piece and something else
    td_n = 0
    td_teacher_combo = td_clone_combo = 0
    td_b = td_c = 0  # McNemar discordant cells
    # Mechanism probe: does the choice hinge on a variable the model CANNOT see?
    # snapshot_alakazam carries hand counts for Rare Candy / Alakazam / Boss ONLY,
    # and reduced_alakazam (the 9 scalars crossed with the card-id one-hot -- the
    # only block that can express "fetch card X in situation R") carries none of
    # the evolution line. So "do I already hold an Abra/Kadabra" is invisible.
    # If the teacher's rate swings on it and the clone's is flat, blindness is proven.
    probe: dict[tuple[bool, bool], list[int]] = collections.defaultdict(
        lambda: [0, 0, 0])  # (abra_in_hand, kadabra_in_hand) -> [n, teacher_combo, clone_combo]

    for i, r in enumerate(rows):
        if i % 2000 == 0 and i:
            print(f"  ...{i}/{len(rows)}", flush=True)
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None:
            n_skipped += 1
            continue
        if F.is_prize_pick(obs.select):
            n_prize += 1
            continue
        opts = obs.select.option
        if len(opts) < 2:
            n_single_opt += 1
            continue

        ids = [resolve_option(o, obs).card_id for o in opts]
        uniq = {c for c in ids if c is not None}
        trivial = len(uniq) < 2
        if trivial:
            n_trivial += 1

        t_idx = [a for a in r.action if 0 <= a < len(opts)]
        if not t_idx:
            n_skipped += 1
            continue
        try:
            gs = GameState.build(obs, cards)
            ctx = DecisionContext(raw=r.raw_observation, observation=obs, cards=cards)
            pick = pol.choose(ctx)
        except Exception:
            n_skipped += 1
            continue
        if not pick:
            n_skipped += 1
            continue

        t_ids = collections.Counter(ids[a] for a in t_idx)
        c_ids = collections.Counter(ids[a] for a in pick if 0 <= a < len(opts))
        same = t_ids == c_ids

        turn = obs.current.turn
        early = turn <= EARLY_TURN_MAX

        slices["all"].add(same)
        if not trivial:
            slices["all_nt"].add(same)
        (slices["early"] if early else slices["late"]).add(same)
        if not trivial:
            (slices["early_nt"] if early else slices["late_nt"]).add(same)
        (slices["won"] if r.won else slices["lost"]).add(same)
        if early and turn in per_turn:
            per_turn[turn].add(same)
        if arch_map:
            arch = arch_map.get(r.game_id, "unknown")
            by_arch.setdefault(arch, Slice(f"  {arch}")).add(same)

        if early:
            early_games.add(r.game_id)
            if not same:
                early_disagreements += 1
            for cid in t_ids.elements():
                t_bucket[_bucket(cid)] += 1
            for cid in c_ids.elements():
                c_bucket[_bucket(cid)] += 1
            if not same:
                tn = "+".join(sorted(_name(cards, c) for c in t_ids.elements()))
                cn = "+".join(sorted(_name(cards, c) for c in c_ids.elements()))
                confusion[(tn, cn)] += 1

            # --- G-0b tradeoff subset -------------------------------------
            buckets = {_bucket(c) for c in uniq}
            if not trivial and "COMBO" in buckets and buckets - {"COMBO"}:
                td_n += 1
                t_combo = any(_bucket(c) == "COMBO" for c in t_ids.elements())
                c_combo = any(_bucket(c) == "COMBO" for c in c_ids.elements())
                td_teacher_combo += t_combo
                td_clone_combo += c_combo
                if t_combo and not c_combo:
                    td_b += 1
                elif c_combo and not t_combo:
                    td_c += 1
                hand = gs.me.hand if gs.me is not None and gs.me.hand else []
                key = (any(c.id == 741 for c in hand), any(c.id == 742 for c in hand))
                cell = probe[key]
                cell[0] += 1
                cell[1] += t_combo
                cell[2] += c_combo

    # ------------------------------------------------------------------ report
    scored = slices["all"].total
    print("\n" + "=" * 82)
    print("M34 TRACK B — TO_HAND divergence (clone vs teacher, identical states)")
    print("=" * 82)
    print(f"raw TO_HAND rows      : {len(rows)}")
    print(f"  dropped, prize pick : {n_prize}   (informationless, training drops them too)")
    print(f"  dropped, 1 option   : {n_single_opt}")
    print(f"  dropped, error      : {n_skipped}")
    print(f"  SCORED              : {scored}")
    print(f"  ...of which trivial : {n_trivial} "
          f"({n_trivial / max(scored,1):.1%} — every option is the same card id, "
          f"free accuracy)")

    print(f"\nAGREEMENT (card-id multiset match)")
    print(f"  {'slice':<34}{'agree':>8}{'n':>9}{'rate':>9}")
    for key in ("all", "all_nt", "early", "early_nt", "late", "late_nt", "won", "lost"):
        s = slices[key]
        if s.total:
            print(f"  {s.line()}")
    print(f"\n  per early turn:")
    for t in sorted(per_turn):
        if per_turn[t].total:
            print(f"  {per_turn[t].line()}")
    if by_arch:
        print(f"\n  by opponent archetype:")
        for s in sorted(by_arch.values(), key=lambda x: -x.total):
            if s.total >= 30:
                print(f"  {s.line()}")

    print(f"\nWHAT EACH SIDE FETCHES (turns 1-{EARLY_TURN_MAX}, on identical states)")
    print(f"  {'bucket':<14}{'teacher':>10}{'rate':>9}{'clone':>10}{'rate':>9}{'delta':>9}")
    tt = sum(t_bucket.values()) or 1
    ct = sum(c_bucket.values()) or 1
    for b in ("COMBO", "ENGINE", "OTHER", "UNRESOLVED"):
        tr, cr = t_bucket[b] / tt, c_bucket[b] / ct
        print(f"  {b:<14}{t_bucket[b]:>10}{tr:>8.1%}{c_bucket[b]:>10}{cr:>8.1%}{cr - tr:>+9.1%}")

    print(f"\nTOP EARLY DISAGREEMENTS (teacher fetched -> clone would fetch)")
    print(f"  {'teacher':<30}{'clone':<30}{'n':>7}")
    for (t, c), n in confusion.most_common(20):
        print(f"  {t[:29]:<30}{c[:29]:<30}{n:>7}")

    # ---------------------------------------------------------------- G-0 gate
    print("\n" + "=" * 82)
    print("PRE-REGISTERED KILL GATE G-0")
    print("=" * 82)

    a_rate = slices["early_nt"].rate
    g0a = a_rate < G0A_AGREEMENT_MAX
    print(f"G-0a  early non-trivial agreement = {a_rate:.1%} "
          f"(need < {G0A_AGREEMENT_MAX:.0%})  -> {'PASS' if g0a else 'FAIL'}")
    if not g0a:
        print("      the clone already fetches what the teacher fetches; TO_HAND cannot")
        print("      explain a 0.9-turn assembly gap.")

    T = td_teacher_combo / td_n if td_n else 0.0
    C = td_clone_combo / td_n if td_n else 0.0
    p = _mcnemar(td_b, td_c)
    gap = T - C
    g0b = (gap >= G0B_MIN_GAP) and (p < G0B_MAX_P)
    print(f"\nG-0b  tradeoff subset (turn <= {EARLY_TURN_MAX}, >=2 ids, COMBO vs non-COMBO "
          f"both available)")
    print(f"      n = {td_n}")
    print(f"      teacher COMBO pick rate T = {T:.3f}  ({_rate(td_teacher_combo, td_n)})")
    print(f"      clone   COMBO pick rate C = {C:.3f}  ({_rate(td_clone_combo, td_n)})")
    print(f"      gap T-C = {gap:+.3f}   (need >= +{G0B_MIN_GAP})")
    print(f"      McNemar discordant: teacher-only {td_b}, clone-only {td_c}, "
          f"p = {p:.4g}  (need < {G0B_MAX_P})")
    print(f"      -> {'PASS' if g0b else 'FAIL'}")
    if not g0b:
        if gap < 0:
            print("      FALSIFIED in the opposite direction: the clone fetches combo pieces")
            print("      MORE often than the teacher on identical states (cf. M33 sec.6, where")
            print("      the clone was also already MORE aggressive on Powerful Hand).")
        else:
            print("      the clone's fetch policy is not materially different from the")
            print("      teacher's; 'the clone under-fetches combo pieces' is falsified.")

    print(f"\n      MECHANISM PROBE — same subset, split on a variable the model CANNOT see")
    print(f"      (neither snapshot_alakazam nor reduced_alakazam carries an Abra/Kadabra")
    print(f"       hand count; reduced is the ONLY block crossed with the card-id one-hot)")
    print(f"      {'Abra in hand':<14}{'Kadabra in hand':<17}{'n':>7}"
          f"{'teacher COMBO':>15}{'clone COMBO':>13}")
    for key in sorted(probe, key=lambda k: -probe[k][0]):
        n, tc, cc = probe[key]
        if n < 30:
            continue
        print(f"      {str(key[0]):<14}{str(key[1]):<17}{n:>7}"
              f"{tc / n:>14.1%}{cc / n:>13.1%}")
    t_vals = [probe[k][1] / probe[k][0] for k in probe if probe[k][0] >= 30]
    c_vals = [probe[k][2] / probe[k][0] for k in probe if probe[k][0] >= 30]
    if t_vals and c_vals:
        t_swing, c_swing = max(t_vals) - min(t_vals), max(c_vals) - min(c_vals)
        print(f"      swing across cells: teacher {t_swing:.1%}  vs  clone {c_swing:.1%}")
        if t_swing >= 2 * max(c_swing, 1e-9):
            print(f"      => the teacher CONDITIONS on this variable and the clone does not.")
            print(f"         This is a feature-OBSERVABILITY gap, not a capacity gap: more MLP")
            print(f"         capacity over features that omit the discriminating variable")
            print(f"         cannot learn the rule. Fix belongs in reduced_alakazam.")

    n_games = len(early_games) or 1
    per_game = early_disagreements / n_games
    g0c = early_disagreements >= G0C_MIN_DISAGREEMENTS and per_game >= G0C_MIN_PER_GAME
    print(f"\nG-0c  early disagreements = {early_disagreements} "
          f"(need >= {G0C_MIN_DISAGREEMENTS}), "
          f"per game = {per_game:.2f} over {n_games} games (need >= {G0C_MIN_PER_GAME})")
    print(f"      -> {'PASS' if g0c else 'FAIL'}")

    print("\n" + "-" * 82)
    if g0a and g0b and g0c:
        print("VERDICT: G-0 PASSES on all three. Proceed to Track C (MLP for TO_HAND).")
    else:
        failed = [n for n, ok in (("G-0a", g0a), ("G-0b", g0b), ("G-0c", g0c)) if not ok]
        print(f"VERDICT: G-0 FIRES ({', '.join(failed)}). Do NOT train a TO_HAND model.")
        print("         Write it up as a negative and switch to Track E (intra-turn planning).")
    print("-" * 82)
    return 0


def _name(cards: CardDatabase, cid: int | None) -> str:
    if cid is None:
        return "?"
    info = cards.get_card(cid)
    return f"{info.name}" if info and info.name else f"id{cid}"


if __name__ == "__main__":
    raise SystemExit(main())
