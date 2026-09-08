# M25 findings — third screening round (batch of 20 by score) + the instrument's ±0.05 blind spot made undeniable (2026-07-23)

**Bottom line: screened the next 20 unscreened candidates by leaderboard score (ranks #125–#423,
score 874–976) through a new self-tracking, two-tier-download pipeline. 18 of 20 were confidently
rejected — 8 fail greedy-pilotability (M6's wall), 4 lose the head-to-head by −0.08 to −0.38, and 6
more DISCARD_CLONE. Two (`backstona`, `vvs`) beat kanga at n=6 with a clean positive CI — but when
re-measured at n=12 with full-data weights, BOTH collapsed to a statistical TIE with kanga
(+0.018 [−0.001,+0.035] and +0.013 [−0.006,+0.031]), while beating v1 (+0.025, +0.032). That is the
*exact* signature of THIRD (ties kanga, beats v1) — the agent the user reports is LOSING on the
ladder. The round's real result is not a new candidate; it is a live, within-round demonstration
that the offline gauntlet does not discriminate in the ±0.05 band: the n=6 "beats kanga" signal was
noise that regressed toward zero when the sample doubled. No candidate clears the bar of "a margin
large enough the instrument cannot misread it." Nothing built, nothing shipped, no `src/` change; the
pipeline is now self-tracking for the next round.**

Continues [m22_findings.md](m22_findings.md)/[m23_findings.md](m23_findings.md). M24 was the
(negative) pooling experiment; M25 returns to the candidate hunt the user chose to continue.

## Why this round, and the pipeline it built

After M24 closed pooling, the standing lever is finding a new clone-friendly deck (M23's
recommendation: the ~900–1050 band, ~270 unscreened candidates). The user asked to continue
screening **20 at a time, by score**, and flagged the real pain: **replay volume** — `replays/` is
wiped between rounds, so 20×150 = ~3000 downloads per round.

Two pipeline fixes addressed that (both reusable):

- **`tools/next_batch.py`** — the `screened` flag in `data/candidates_joined.json` was broken (1 of
  289 marked). The tool reconciles it against disk (any candidate with a `decks/{slug}.csv`, written
  for *every* screened candidate, or a sid in `candidates_m22*.txt`), `--backfill`s the flag (1→22),
  prints the next N unscreened by score with a ready download line, and `--mark`s a batch done so the
  pipeline is **self-tracking** going forward.
- **Two-tier download** — Tier 1 pulls only `--limit 40` for all 20 (deck-strength needs ~1 replay;
  it kills ~⅓ almost free). Only survivors get topped up to `--limit 150` (incremental). This round
  downloaded ~840 + 2 top-ups (~330) instead of 3000.

## Tier 1 — screen of all 20 (subsampled/optimistic clonability, deck-strength early-stopped)

| verdict | candidates |
|---|---|
| DISCARD_DECK (str < 0.40, M6 wall) | Gyoukou 0.264, kaggle_bbgg 0.271, SamuelSanolume 0.310, syuuuuu 0.345, mutualns 0.250, LagrangianLocomotive 0.376, Nobu Kimura ×2 (0.215/0.250) |
| DISCARD_CLONE (passes strength) | vvs, Tanupro, flaty, pokeca2018, Erik van de Ven |
| survived to head-to-head (str ≥ 0.40) | backstona (0.778/0.667), Lester Leong (0.750/0.656), AL Najafi (0.694/0.697), Akihiro Nomura (0.413/**0.764** PROMISING), すーぱーひとで (0.653/0.663), senkin13 (0.667/0.660), Abhi (0.625/0.671) |

**M6's wall, 8 more data points.** Clonability never reached 0.75 except Akihiro Nomura (0.764, on a
subsample that overstates — it fell to nothing at the gate below).

## Tier 3-lite — head-to-head vs kanga on the (handicapped) 40-game weights, n=6

Six survivors run directly against kanga's shippable `KANGASKHAN_1052` to triage cheaply before any
top-up:

| candidate | strength | clonability | Δ vs kanga (n=6) | 90% CI | verdict |
|---|---|---|---|---|---|
| **backstona** | 0.778 | 0.667 | **+0.029** | [+0.003, +0.057] | BEATS kanga |
| **vvs** | 0.736 | 0.585 | **+0.028** | [+0.004, +0.051] | BEATS kanga |
| lesterleong | 0.750 | 0.656 | −0.019 | [−0.044, +0.006] | ties |
| すーぱーひとで | 0.653 | 0.663 | −0.085 | [−0.114, −0.056] | LOSES |
| akihironomura | 0.413 | **0.764** | −0.228 | [−0.265, −0.190] | LOSES |
| alnajafi | 0.694 | 0.697 | −0.382 | [−0.422, −0.342] | LOSES |

**Clonability does not predict the head-to-head — reconfirmed, strongly.** The round's *best* cloner
(Akihiro Nomura, 0.764) loses by −0.228; the *worst* of the survivors I ran (vvs, 0.585) beats kanga.
This is the M22 lesson again, now with the ordering inverted from clonability.

## The decisive step — n=12 on full-data weights collapses the signal

backstona (topped up 40→190 games) and vvs (topped up to 59 = all it has) were re-screened on full
data (honest clonability: backstona 0.546, vvs 0.553 — both DISCARD_CLONE by the 0.75 bar, the ITF
pattern) and re-run at **n=12** vs kanga and vs v1:

| candidate | Δ vs kanga (n=12) | 90% CI | Δ vs v1 (n=12) | 90% CI |
|---|---|---|---|---|
| backstona | **+0.018** | **[−0.001, +0.035]** ties | +0.025 | [+0.004, +0.047] beats |
| vvs | **+0.013** | **[−0.006, +0.031]** ties | +0.032 | [+0.013, +0.051] beats |

**The n=6 "beats kanga" (CI entirely positive) collapsed to a TIE at n=12 for both.** Doubling the
sample pulled the delta toward zero — i.e. the n=6 positive was substantially sampling noise. What
survives is: **ties kanga, beats v1** — the identical signature to THIRD (m23_findings addendum:
THIRD −0.006 vs kanga, slightly below v1), the agent the user observes losing on the ladder.

## The load-bearing finding — the instrument's ±0.05 blind spot, shown within one round

The user's stated doubt ("these tests feel inconclusive; THIRD beat/ tied kanga offline yet loses on
the ladder") is not a hunch — it is the project's documented ground truth (M14: with a *known* +0.13
ladder gap, the gauntlet reads −0.003), and this round demonstrates it live:

- The gauntlet has now mis-ordered the ladder in every direction: kanga read tie-vs-v1 → **+174 elo
  real**; ITF read beats-kanga → **worse on ladder**; THIRD read tie-vs-kanga → **losing on ladder**.
- Within M25, the *same* candidate flipped from "BEATS kanga (CI>0)" at n=6 to "ties kanga" at n=12.
  The instrument cannot even hold its own verdict stable across sample size in this band.
- **Conclusion: the offline gauntlet is a reliable VETO (18/20 rejected here, and a −0.38 loss is
  never a ladder winner) but not a RANKER in the ±0.05 band.** backstona and vvs are lottery tickets
  of the same expected value as THIRD/ITF; the offline number gives essentially no edge among them.

## Verdict & recommendation

- **NO candidate qualifies.** None clears the only bar worth acting on: a margin large enough the
  instrument cannot misread it. backstona/vvs tie kanga; that is not a reason to spend a scarce
  ladder slot (only the 2 most-recent submissions are final-tracked; each restarts at μ₀=600 and
  needs ~50 episodes).
- **Nothing built or shipped; no `src/` change; no upload.** kanga/THIRD untouched.
- **The real bottleneck is ladder-test slots, not candidates.** Generating more offline-
  indistinguishable candidates does not help. The highest-value move is to let the current ladder bet
  (THIRD, or kanga) run to ~50-episode convergence and judge on real data — the only judge with
  resolution here.
- If screening continues, the criterion should change: a candidate is worth a slot only if it FAILS
  cleanly (cheap veto) or WINS by a large, sample-stable margin. A +0.02–0.03 does not.
- 247 of 289 candidates remain unscreened (`next_batch.py` now tracks this reliably); the pipeline is
  cheap to keep running, but per the above its marginal value for *finding a ladder winner* is low.

## Files

New (all deletable, none in production): `docs/m25_findings.md`, `tools/next_batch.py`,
`candidates_m25.txt`, `candidates_m25_winners.txt`,
`decks/{20 slugs}.csv`, `data/models/{20 slugs}_screen.json`,
`data/imitation/{20 slugs}_screen*.jsonl.gz`, `replays/<20 sids>/`.
Modified: `data/candidates_joined.json` (only the `screened` flag — backfilled 1→22, then the 20 of
this round marked → 42 screened / 247 remaining). No `src/` change; no bundle; nothing uploaded.
