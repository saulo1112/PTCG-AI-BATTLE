"""How fragile is the clone's 'PH first-use median turn 6 (n=27) vs teacher's 4
(n=603)' read? Extracts the RAW per-game turn values (not just the median) for
both slices, then bootstraps: repeatedly draw a random sample the SAME SIZE as
the clone's sample from the TEACHER's own population and ask how often that
alone (pure small-n noise, zero skill difference) produces a median >= 6.

If that happens often, the clone's number is not distinguishable from teacher-
level play. If it's rare, the delay is a real, clone-specific signal.
"""
import sys, json, glob, random, statistics
from pathlib import Path
sys.path.insert(0, 'scratchpad')

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.kaggle_replay import ReplayDecision, _strip_observation, _ACTIVE, _DECK_LEN, player_seats
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.state.game_state import GameState

import diagnose_mlp as dm
import extract_top_decks as et

GRIMM_MARKER = 648
TEACHER_FOLDER = Path("replays/54773249")
POWERFUL_HAND = dm.POWERFUL_HAND

cfg = load_config(profile="benchmark")
cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
parser = ObservationParser()


def _decisions_for_seat(path: Path, seat: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    game_id = path.stem
    steps = data.get("steps", [])
    rewards = data.get("rewards") or []
    won = seat < len(rewards) and (rewards[seat] or 0) > 0
    for i in range(len(steps) - 1):
        row = steps[i]; nxt = steps[i + 1]
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
            continue
        yield ReplayDecision(
            game_id=game_id, seat=seat, step_index=i,
            raw_observation=_strip_observation(obs),
            action=[int(a) for a in action], won=won,
            context=int(select.get("context", -1)),
            select_type=int(select.get("type", -1)),
            min_count=int(select.get("minCount", 1)),
            max_count=int(select.get("maxCount", 1)),
            n_options=len(options),
        )


def raw_ph_turns(decisions) -> list[int]:
    """First turn Powerful Hand was used, one value per game that used it at all."""
    ph_turn: dict[str, int] = {}
    for dec in decisions:
        if SelectContextKind(dec.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(dec.raw_observation)
        if obs.select is None or obs.current is None:
            continue
        turn = obs.current.turn
        for a in dec.action:
            if not (0 <= a < len(obs.select.option)):
                continue
            opt = obs.select.option[a]
            if opt.type is OptionKind.ATTACK and opt.attackId == POWERFUL_HAND:
                ph_turn.setdefault(dec.game_id, turn)
    return list(ph_turn.values())


# --- clone's 27 grimmsnarl games ---
clone_files = []
for fp in sorted(glob.glob('replays/55011997/*.json')):
    if 'metadata' in fp:
        continue
    f = Path(fp)
    d = json.loads(f.read_text(encoding='utf-8'))
    seats = player_seats(d, dm.OUR)
    if len(seats) != 1:
        continue
    our = seats[0]
    deck = et.extract_deck(d, 1 - our)
    if deck and GRIMM_MARKER in deck:
        clone_files.append((f, our))

clone_decisions = []
for f, seat in clone_files:
    clone_decisions.extend(_decisions_for_seat(f, seat))
clone_turns = raw_ph_turns(clone_decisions)
print(f"CLONE vs Grimmsnarl: {len(clone_files)} games, {len(clone_turns)} used PH")
print(f"  raw turns: {sorted(clone_turns)}")
print(f"  median: {statistics.median(clone_turns)}  mean: {statistics.mean(clone_turns):.2f}")

# --- teacher's 603 grimmsnarl games ---
yushin_deck = tuple(sorted(int(x) for x in dm.DECK_CSV.read_text(encoding='utf-8').split()))
teacher_seats = {}
for fp in sorted(glob.glob(str(TEACHER_FOLDER / "*.json"))):
    if 'metadata' in fp:
        continue
    f = Path(fp)
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    decks = [et.extract_deck(d, s) for s in (0, 1)]
    seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin_deck]
    if len(seats) != 1:
        continue
    our = seats[0]
    if decks[1 - our] and GRIMM_MARKER in decks[1 - our]:
        teacher_seats[f] = our

teacher_decisions = []
for f, seat in teacher_seats.items():
    teacher_decisions.extend(_decisions_for_seat(f, seat))
teacher_turns = raw_ph_turns(teacher_decisions)
print(f"\nTEACHER vs Grimmsnarl: {len(teacher_seats)} games, {len(teacher_turns)} used PH")
print(f"  median: {statistics.median(teacher_turns)}  mean: {statistics.mean(teacher_turns):.2f}")

from collections import Counter
print(f"  turn histogram: {dict(sorted(Counter(teacher_turns).items()))}")
print(f"  clone turn histogram: {dict(sorted(Counter(clone_turns).items()))}")

# --- bootstrap: draw len(clone_turns)-sized samples from teacher's population ---
random.seed(0)
n = len(clone_turns)
N_TRIALS = 20000
medians = []
for _ in range(N_TRIALS):
    sample = random.choices(teacher_turns, k=n)
    medians.append(statistics.median(sample))

frac_ge_6 = sum(1 for m in medians if m >= 6) / N_TRIALS
frac_ge_5_5 = sum(1 for m in medians if m >= 5.5) / N_TRIALS
print(f"\n=== BOOTSTRAP ({N_TRIALS} resamples of size {n} from TEACHER's own {len(teacher_turns)}-game distribution) ===")
print(f"  P(resampled median >= 6.0) = {frac_ge_6:.1%}   <- if clone played exactly like teacher")
print(f"  P(resampled median >= 5.5) = {frac_ge_5_5:.1%}")
print(f"  distribution of resampled medians: {dict(sorted(Counter(medians).items()))}")

# ============================================================================
# CONTROL CHECK: does the clone ALSO fire PH later than the teacher in a
# matchup it WINS comfortably (Mega Lucario, 678)? If yes, this "slow trigger"
# is a GENERAL clone trait (M31's already-known broad gap), not something
# specific to Grimmsnarl/Munkidori pressure -- which would undercut the case
# for a Grimmsnarl-CONDITIONED feature as the fix.
# ============================================================================
print("\n" + "=" * 70)
print("CONTROL: same instrument on Mega Lucario (clone's best matchup, 89% WR)")
LUCARIO_MARKER = 678

clone_luc_files = []
for fp in sorted(glob.glob('replays/55011997/*.json')):
    if 'metadata' in fp:
        continue
    f = Path(fp)
    d = json.loads(f.read_text(encoding='utf-8'))
    seats = player_seats(d, dm.OUR)
    if len(seats) != 1:
        continue
    our = seats[0]
    deck = et.extract_deck(d, 1 - our)
    if deck and LUCARIO_MARKER in deck:
        clone_luc_files.append((f, our))

clone_luc_decisions = []
for f, seat in clone_luc_files:
    clone_luc_decisions.extend(_decisions_for_seat(f, seat))
clone_luc_turns = raw_ph_turns(clone_luc_decisions)
print(f"CLONE vs Mega Lucario: {len(clone_luc_files)} games, {len(clone_luc_turns)} used PH")
print(f"  raw turns: {sorted(clone_luc_turns)}")
if clone_luc_turns:
    print(f"  median: {statistics.median(clone_luc_turns)}  mean: {statistics.mean(clone_luc_turns):.2f}")

teacher_luc_seats = {}
for fp in sorted(glob.glob(str(TEACHER_FOLDER / "*.json"))):
    if 'metadata' in fp:
        continue
    f = Path(fp)
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    decks = [et.extract_deck(d, s) for s in (0, 1)]
    seats = [s for s in (0, 1) if decks[s] and tuple(sorted(decks[s])) == yushin_deck]
    if len(seats) != 1:
        continue
    our = seats[0]
    if decks[1 - our] and LUCARIO_MARKER in decks[1 - our]:
        teacher_luc_seats[f] = our

teacher_luc_decisions = []
for f, seat in teacher_luc_seats.items():
    teacher_luc_decisions.extend(_decisions_for_seat(f, seat))
teacher_luc_turns = raw_ph_turns(teacher_luc_decisions)
print(f"\nTEACHER vs Mega Lucario: {len(teacher_luc_seats)} games, {len(teacher_luc_turns)} used PH")
if teacher_luc_turns:
    print(f"  median: {statistics.median(teacher_luc_turns)}  mean: {statistics.mean(teacher_luc_turns):.2f}")
    print(f"  turn histogram: {dict(sorted(Counter(teacher_luc_turns).items()))}")

if clone_luc_turns and teacher_luc_turns:
    n2 = len(clone_luc_turns)
    meds2 = []
    for _ in range(20000):
        sample = random.choices(teacher_luc_turns, k=n2)
        meds2.append(statistics.median(sample))
    clone_med2 = statistics.median(clone_luc_turns)
    frac_ge = sum(1 for m in meds2 if m >= clone_med2) / 20000
    print(f"\n  P(resampled median >= clone's {clone_med2}) = {frac_ge:.1%}")
