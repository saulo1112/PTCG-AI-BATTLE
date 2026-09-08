"""M48 -- does hand-size disruption predict winning ACROSS THE WHOLE LADDER, not just
against us?

WHY THIS EXISTS. The user's reframing, stated plainly: most competitors beating us are
probably ALSO cloning some player, not genuinely expert TCG humans hand-crafting every
decision. If the field is clone-dominated, "play more like a strong human" (everything
M7-M48 tried) attacks the wrong lever. What clones share is a specific, MEASURED failure
mode: our own agent (M38) diverges from its teacher's line every ~4.7 decisions and then
plays states with zero training signal. Grimmsnarl -- the one deck-level wall this
project has never broken (M32 onward, confirmed across THREE independent real pilots, so
it is the DECK, not one skilled player) -- wins via hand-size denial (Munkidori moves our
damage counters, Unfair Stamp resets hands). That is consistent with "forces opponents
off the line their teacher demonstrated," which is exactly what breaks a clone.

This checks the hypothesis with data already on disk (M46's daily ladder dump, 4622
episodes / 377 distinct players, CC0, no new games): does having hand-disruption tech in
your deck predict winning BROADLY across the ladder, not just in our one matchup?

CARD LIST, built from TEXT not memory. An earlier attempt cited "Petrel/Spikemuth Gym
attack hand size" from a stale project memory -- checking their ACTUAL card text shows
both just search a card INTO your own hand (card advantage, not disruption). Every ID
below was selected by reading its real effect text (`_m48_disruption_scan.txt`, 91
opponent+hand hits, manually classified). Two families, both a genuine hand-SIZE
reduction for the opponent (not "reveals hand" -- information only, not disruption; not
"opponent can't play X from hand" -- a lock/tax, a different mechanism; not "energy into
opponent's hand" -- that INCREASES their hand):

  DISCARD family (opponent's hand shrinks, a card leaves it for discard/deck):
    Espeon ex(246) Snorunt/N's Purrloin(103,291) Chingling(433) Meowth(470)
    Porygon(473) Sandile/Krokorok/Krookodile "Tighten Up"(538,539,540) Liepard(609)
    Mega Absol ex(687) Houndstone(753) Aipom(843) Scraggy/Mandibuzz ex/Scrafty ex(895,
    896,984) Luxray ex(954) Talonflame(1024) Furfrou "Hand Trim"(1076) Energy
    Swatter(1149) Eri(1186) Xerosic's Machinations(1197)

  ASYMMETRIC-RESET family (both reshuffle hand into deck, but the discloser draws
    fewer -- Unfair Stamp's own mechanism, the card M32 actually named correctly):
    Unfair Stamp(1080) Team Rocket's Archer(1217) Harlequin(1223) Hand Trimmer(1087,
    the exact card M27 tested as a deck swap on FROZEN weights and found weak --
    this checks whether the CARD is strong across the ladder even though a frozen
    clone couldn't learn to use it)

EXCLUDED, deliberately, with the reason: card-type LOCKS ("opponent can't play Item
cards from hand" -- Budew 235, Frillish 597, Jellicent ex 598, Genesect 142, Galvantula
ex 161, Tyranitar 290, Cetitan ex 424, Fraxure 994, etc.) are a different mechanism
(restricts USE of held cards, not hand SIZE) and are tested separately as `LOCK_IDS` so
the two hypotheses don't contaminate each other. "Reveals hand" alone (Hoothoot,
Espurr, Pidove) is information, not disruption. "Damage scales with hand size"
(Chandelure, Mega Froslass ex, our own Alakazam "Powerful Hand") rewards a small hand,
it doesn't create one.

THE TEST. Two readings, reported separately because they answer different questions:

  1. PAIRED (the one that matters): filter to games where EXACTLY ONE side runs
     disruption tech. Report that side's win rate against 50%. This controls for
     "was the disruption player just facing weak opposition" reasonably well since
     it's the SAME game, though it does NOT control for "disruption players are also
     just better on average" -- flagged explicitly in the output, not hidden.
  2. UNPAIRED (weaker, correlational): every game's disruption-side win rate,
     including mirror match-ups on both sides. Reported for completeness, NOT as the
     decisive number.

Run:  PYTHONIOENCODING=utf-8 python -u scratchpad/m48_ladder_disruption_check.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

CARD_DATA = Path("build/colab_export/card_data.json")
DUMP_DIR = Path(r"C:\Users\ASUS\.cache\kagglehub\datasets\kaggle\pokemon-tcg-ai-battle-episodes-2026-08-11\versions\1")
DECK_LEN = 60

DISCARD_IDS = {246, 103, 291, 433, 470, 473, 538, 539, 540, 609, 687, 753, 843, 895,
               896, 984, 954, 1024, 1076, 1149, 1186, 1197}
ASYM_RESET_IDS = {1080, 1217, 1223, 1087}
DISRUPTION_IDS = DISCARD_IDS | ASYM_RESET_IDS

LOCK_IDS = {235, 597, 598, 142, 161, 290, 424, 994, 969, 719, 738, 1151}


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - adj) / denom, (centre + adj) / denom


def _load_card_names():
    d = json.loads(CARD_DATA.read_text(encoding="utf-8"))
    return {c["cardId"]: c["name"] for c in d["cards"]}


def _deck_action(steps, seat):
    """The 60-card deck submission. Confirmed at steps[1][seat] in this dump's format
    (kaggle_replay.py finds it by length==DECK_LEN wherever it falls; this dump puts it
    consistently at index 1, checked on a sample before writing this)."""
    for row in steps[:4]:
        cell = row[seat] if seat < len(row) else None
        if not cell:
            continue
        action = cell.get("action")
        if isinstance(action, list) and len(action) == DECK_LEN:
            return action
    return None


def main() -> int:
    names = _load_card_names()
    files = sorted(DUMP_DIR.glob("*.json"))
    print(f"scanning {len(files)} episodes in {DUMP_DIR}\n")

    n_games = n_parsed = 0
    n_with_disruption_side = Counter()   # 0, 1, or 2 sides running disruption tech
    paired_wins = paired_n = 0
    unpaired_disr_wins = unpaired_disr_n = 0
    lock_paired_wins = lock_paired_n = 0
    disruption_players = set()
    all_players = set()

    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        steps = d.get("steps") or []
        rewards = d.get("rewards") or []
        agents = d.get("info", {}).get("Agents", [])
        if len(rewards) != 2 or len(agents) != 2 or len(steps) < 2:
            continue
        deck0, deck1 = _deck_action(steps, 0), _deck_action(steps, 1)
        if deck0 is None or deck1 is None:
            continue
        n_games += 1
        r0, r1 = rewards[0], rewards[1]
        if r0 is None or r1 is None or r0 == r1:
            continue   # no clean winner (draw or malformed)
        n_parsed += 1
        winner = 0 if r0 > r1 else 1

        disr0 = bool(DISRUPTION_IDS & set(deck0))
        disr1 = bool(DISRUPTION_IDS & set(deck1))
        lock0 = bool(LOCK_IDS & set(deck0))
        lock1 = bool(LOCK_IDS & set(deck1))
        n_with_disruption_side[disr0 + disr1] += 1

        p0, p1 = agents[0].get("Name", "?"), agents[1].get("Name", "?")
        all_players.add(p0); all_players.add(p1)
        if disr0:
            disruption_players.add(p0)
        if disr1:
            disruption_players.add(p1)

        if disr0 != disr1:   # exactly one side runs disruption tech
            paired_n += 1
            disr_side = 0 if disr0 else 1
            paired_wins += int(winner == disr_side)
        if lock0 != lock1:
            lock_paired_n += 1
            lock_side = 0 if lock0 else 1
            lock_paired_wins += int(winner == lock_side)

        if disr0:
            unpaired_disr_n += 1; unpaired_disr_wins += int(winner == 0)
        if disr1:
            unpaired_disr_n += 1; unpaired_disr_wins += int(winner == 1)

    print(f"games with two 60-card decks parsed: {n_games}")
    print(f"games with a clean winner (no draw): {n_parsed}")
    print(f"distinct players seen: {len(all_players)}")
    print(f"distinct players running ANY hand-disruption tech: {len(disruption_players)} "
          f"({len(disruption_players) / max(len(all_players), 1):.1%})")
    print(f"\nside distribution: neither side disruption={n_with_disruption_side[0]}  "
          f"one side={n_with_disruption_side[1]}  both sides={n_with_disruption_side[2]}\n")

    print("=" * 78)
    print("1. PAIRED comparison (exactly one side runs disruption tech) -- THE DECISIVE NUMBER")
    print("=" * 78)
    if paired_n:
        wr = paired_wins / paired_n
        lo, hi = _wilson(paired_wins, paired_n)
        print(f"  disruption side: {paired_wins}W-{paired_n - paired_wins}L of {paired_n} "
              f"mixed pairings")
        print(f"  win rate: {wr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (reference: 0.500)")
        if lo > 0.5:
            print("  -> disruption side wins with significance, ACROSS THE LADDER, not just")
            print("     against us. Supports the reframing.")
        elif hi < 0.5:
            print("  -> disruption side LOSES with significance across the ladder.")
            print("     Grimmsnarl's wall against US specifically would need a different")
            print("     explanation (matchup-specific, not a general clone-fragility effect).")
        else:
            print("  -> CI touches 0.500: no ladder-wide effect detected at this sample size.")
    else:
        print("  no mixed pairings found")

    print("\nCaveat this reading does NOT control for: disruption-tech players might simply")
    print("be better deckbuilders/pilots on average, independent of the mechanism. A truly")
    print("clean test would need per-player Elo, which this dump does not carry.")

    print("\n" + "=" * 78)
    print("2. UNPAIRED (every game a disruption deck appears in, mirrors included) -- weaker")
    print("=" * 78)
    if unpaired_disr_n:
        wr = unpaired_disr_wins / unpaired_disr_n
        lo, hi = _wilson(unpaired_disr_wins, unpaired_disr_n)
        print(f"  {unpaired_disr_wins}W-{unpaired_disr_n - unpaired_disr_wins}L of "
              f"{unpaired_disr_n}  win rate {wr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n" + "=" * 78)
    print("3. CONTROL: card-type LOCKS (different mechanism, same paired test)")
    print("=" * 78)
    if lock_paired_n:
        wr = lock_paired_wins / lock_paired_n
        lo, hi = _wilson(lock_paired_wins, lock_paired_n)
        print(f"  {lock_paired_wins}W-{lock_paired_n - lock_paired_wins}L of {lock_paired_n}  "
              f"win rate {wr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    else:
        print("  no mixed pairings found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
