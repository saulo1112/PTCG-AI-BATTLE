# M29 findings — is a converged ladder score reliable, or field/luck? (2026-07-26)

**Bottom line: the user noticed the SAME `imitation-kanga` bundle posted 772.7 / 856.8 / 899.4 across
uploads (citing 54911514≈773 vs 54988272≈900/905) and asked whether the score correlates with
timing/field or is pure luck — before spending a scarce ladder slot on the yushin clone. We downloaded
both submissions (110 and 73 episodes) and proved: (1) it is the IDENTICAL agent (both play one fixed
deck, byte-identical to each other and to `decks/懒惰的金枪鱼.csv`; kanga's weights never changed);
(2) the two runs faced NEARLY DISJOINT opponent pools — only 2 shared of 108 vs 69 distinct — and the
773-run ran into a Mega-Lucario-HEAVY field (21× vs 3×), a known bad Kanga matchup, dragging its WR to
55.0% vs 59.7%. So the ~127-elo swing is NOT the agent and NOT within-run luck — it is the OPPONENT
DRAW (which opponents matchmaking assigns × the ladder population at upload time). Practical
conclusion: a single converged score carries ~±60–100 elo of opponent-field noise, so sub-±100 ladder
comparisons are weak evidence — which softens the M27 nsr(843)-vs-kanga(899) read (56 elo = within
noise) and dictates how to judge the yushin free-roll.**

New read-only tool: `scratchpad/compare_submissions.py` (reuses `diagnose_kanga.record`/
`kaggle_replay`/`extract_top_decks`).

## Data availability (from the replay JSON)

`info.Agents` carries opponent **Name** but **Rating is null** in these downloads (no rating join
done — the archetype-mix evidence is decisive without it). There is **no wall-clock episode timestamp**
in the replay (only download time in `metadata.json`), but **`info.EpisodeId` is monotonic** → a
chronological order proxy for the "hora" question.

## Phase B — same agent? YES

Both submissions play exactly ONE deck, byte-identical to each other and to the canonical
`decks/懒惰的金枪鱼.csv`. Kanga's `bc_kangaskhan_1052.json` weights were untouched → same bundle. The
score gap is therefore NOT a property of the agent.

## Phase C — record & field

| | 54911514 (~773) | 54988272 (~905) |
|---|---|---|
| record | 60W-49L (**WR 55.0%**) | 43W-29L (**WR 59.7%**) |
| **Mega Lucario faced** | **21** | **3** |
| Archaludon faced | 10 | 15 |
| Alakazam faced | 32 | 19 |
| distinct opponents | 108 | 69 |
| **opponents shared with the other run** | **2** | |

- **Nearly disjoint fields** (2 shared of ~110): the two uploads essentially took different exams.
- **The 773-run faced 21 Mega Lucario vs 3** — Lucario is one of Kanga's worst matchups (M26), a
  concrete, directional reason its WR/score were lower.
- The 55.0% vs 59.7% WR gap alone is only ~1 SE at n≈100 (not individually significant); the STARK,
  directional signal is the field composition, which TrueSkill amplifies via opponent-weighting.
- **Chronology (EpisodeId order):** both stable within-run (first/second-half WR 57%/53% and
  58%/61%) — NOT a runaway early streak off the μ₀=600 restart. The difference is BETWEEN-run field.

## Phase D — the methodological answer

- **Time/field-correlated, not "pure luck in the score."** Given the games each played, the scores are
  earned; but the OPPONENT DRAW itself is luck/timing, and it swings the identical agent ~127 elo
  (772.7 ↔ 899.4). Effective 1σ ≈ 50–65 elo per single upload.
- **Trust only large ladder gaps.** Differences under ~100 elo between separate uploads are weak
  evidence. This SOFTENS "nsr is worse than kanga" (843 vs 899 = 56 elo, within noise) — nsr is
  *indistinguishable*, not confidently worse (they were concurrent, so a bit tighter than these two).
- **How to reduce the noise (not by re-rolling):** re-uploading the same agent until it scores high is
  cherry-picking a lucky field draw (optimizing noise), costs submission budget, and answers nothing.
  Instead, run the two agents you want to compare **concurrently** (uploaded close together, so they
  share the evolving field/epoch) and judge the **GAP** between them after ~50 eps — a paired
  comparison whose between-field variance largely cancels.

## Implication for the yushin free-roll

Upload `imitation-yushin-mlp` alongside a **freshly re-uploaded plain-kanga baseline** (both then
tracked concurrently; absolute restart to μ₀=600 is fine — the comparison is relative). Judge
**yushin − kanga**: a sustained margin beyond ~50–70 elo (concurrent, so tighter than the ±100 of
separate uploads) is a real result; within ~±50 is inconclusive. Re-run `compare_submissions.py` on
yushin's replays once it has ~30–50 eps to read the gap while controlling for each one's field.

## Files

Tracked: this doc, `scratchpad/compare_submissions.py`. Read-only analysis — no `src/` change, no
bundle, nothing uploaded. Replays (`replays/54911514`, `replays/54988272`) are gitignored,
re-downloadable via `tools/download_competitors.py --submissions 54911514 54988272`.
