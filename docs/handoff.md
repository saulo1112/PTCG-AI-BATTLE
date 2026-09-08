# Handoff — read this first in a new session

**Snapshot date: 2026-07-13. Git HEAD at snapshot: `97c43de` + uncommitted M12–M14
work.** This is a point-in-time summary written so a *new* AI assistant session (no
access to this conversation, no shared memory) can get oriented fast. It is not
authoritative — verify against the live repo before acting on a cited number, file,
or SHA. Deeper detail always lives in the linked doc, never only here.

> **⚠️ TERMINOLOGY NOTE (2026-07-18):** every "player" cited throughout this doc and
> `imitation/*_findings.md` (the 650-elo `greengreenpurple`, the 800-elo Lucario pilot,
> `kenN2439`, `budew`, etc.) is **another Kaggle competitor's submitted BOT/agent** on
> the ladder — sourced from `Logs/` replay JSONs via `kaggle_replay.py`
> (`imitation/dataset.py`), never a live human. This was never stated wrong in the
> docs (they correctly say "player"/"ladder player"), but it was misspoken as "human"
> in conversation once — flagging here so no future session repeats that. It does not
> change any finding: BC still clones a fixed, mediocre (~650-elo) bot's decisions,
> which is exactly why raising imitation fidelity to it (M11/M14) made play worse.
>
> **⚠️ COMPUTE NOTE (2026-07-18):** the M7–M18 conclusion that further gains have low
> EV assumed **no GPU** as a fixed constraint. The user is proposing to use **Google
> Colab** for GPU/compute. This materially reopens exactly one avenue with a different
> disposition: **self-play RL (M16/scratchpad/rl_selfplay.py)**, because self-play
> generates its OWN training data (agent-vs-known-opponents), so it is NOT bottlenecked
> by the scarce single-bot replay dataset that caps BC — only by compute, which Colab
> would relax. Concretely, more compute could support: far larger self-play batches/
> iterations than the 8-22 iters run locally, a more careful λ-anneal (the M17 rerun's
> quick anneal attempt drifted into a K2 kill — more compute allows a gentler,
> better-monitored schedule), extending RL beyond the MAIN-only scope to TO_HAND (a
> named-but-untried structural fix, M16), and/or a materially larger value network.
> This does NOT relax the OTHER three constraints named in conversation (richer/more
> diverse teacher data, deep tree search, real TCG domain expertise) — those are
> unaffected by GPU access. **Not yet planned or started — awaiting the user's
> instructions on scope before building a Colab migration.**
>
> **⚠️ M49-M50 (2026-08-14/15) — read [m49_findings.md](m49_findings.md) FIRST. A silent
> corpus-corrupting bug found and fixed, the teacher's data ceiling measured head by head
> (4/4 negative), and the community's public agent finally measured. Nothing shipped.**
> **(1) THE TEACHER IS STILL PLAYING** — submission 54773249 played 33 episodes on Aug 14.
> An earlier read that he had stopped was WRONG and the reason matters: it inferred
> inactivity from ABSENCE IN THE DAILY DUMP, which is a *sample* (377 players vs the
> leaderboard's 6,725+), not a census. The API by `submissionId` is the authority.
> 555 new episodes downloaded, 0 failures; corpus 2,330 -> **3,074 games (+32%)**,
> 248,163 rows, 0 illegal actions. Deck verified **identical in 392/392** new games (he did
> NOT repeat the M45/M46 deck-swap pattern).
> **(2) THE BUG, and it would have been silent.** The team RENAMED ITSELF TWICE:
> `Yushin Ito` (256 eps) -> `AlphaStarmie` (179) -> `AlphaTCG`/`AlphaTcg` (58/13), all
> confirmed the same submission by API `submissionId`, none colliding with an opponent
> seat. The whole imitation pipeline filters by DISPLAY NAME
> (`build_decision_dataset` -> `player_seats` -> `info.Agents[].Name`), so rebuilding with
> "Yushin Ito" would have matched ZERO seats in every new replay and written a corpus the
> same size as the old one — **no exception, no warning** — and the heads would have been
> retrained believing they had +32% data while having none. Same shape as M37's wrong-deck
> submission. **Fixed:** `player_seats` now accepts a SET of names (a plain string behaves
> exactly as before); 6 new tests incl. one pinning the regression itself. **269 tests
> pass.** A first version of the verification script had the SAME class of bug — it skipped
> all 125 files it looked at and still printed "OK, safe to merge", because with 0 parsed
> `mismatched == 0` is trivially true. It now fails loudly on zero evidence.
> **(3) THE TEACHER IS DECLINING, and it is the PILOT not the field:** per-archetype WR vs
> M33's baseline is down across the board (Grimmsnarl −0.105, mirror −0.163, Kangaskhan
> −0.225; overall 0.569 -> 0.526) *while facing WEAKER opponents* (mean opponent rating
> 1143 -> 1006). Field composition also moved a lot (Grimmsnarl 58.1% -> 14.9% of his field,
> Dragapult now 14.9% at 0.276 WR) but that does not explain the drop WITHIN archetypes.
> **(4) RETRAINING THE CONTEXT HEADS ON +32% DATA: 4/4 FAIL.** SWITCH 0.7724 -> 0.7724
> (+0.0000), SETUP_BENCH 1.0000 -> 1.0000 (saturated, a +0.020 lift is arithmetically
> impossible), ACTIVATE 0.9735 -> 0.9729 (−0.0005), **TO_HAND 0.8536 -> 0.8100 (−0.0437)**.
> TO_HAND had the only real headroom (14.6%) and got the WORST — consistent with (3), since
> it is the tutor-resolution context where a declining pilot's choices degrade.
> **The plan's own premise was falsified by its own experiment:** it argued the heads are
> "small models" that would use extra data where MAIN's MLP could not (M36), but SWITCH /
> ACTIVATE / TO_HAND are `mlp_ensemble(k=3,h=48)` — **architecturally identical to MAIN's
> head**, so M36's architecture ceiling applies to them too. The only genuinely small one
> (SETUP_BENCH, linear) turned out to be saturated at 100%.
> **(5) M50 — the community agent, finally measured.** The user pointed out that 19
> milestones optimised ONE family (clone a teacher with a net) while a notebook of theirs
> sat unopened in the repo root: jazivxt's *A Better Hand / Alakazam Rising Tide v21*,
> published for community use. It is a different family entirely — ~1,200 lines of
> hand-written heuristics with tuned weights, **opponent belief modelling**
> (`_match_archetype`, `_TEMPLATES`), its **own determinized search** over a hand-built leaf
> evaluator, explicit Rocket Energy denial. Same Alakazam archetype, different 60-card build
> (8 cards). Its `agent(obs_dict)` has the same contract as `BasePolicy`, so a thin adapter
> (`scratchpad/m50_notebook_arena.py`) measures it with zero reimplementation:
> **vs greedy 0.980 (49W-1L), vs our champion 0.060 (6W-94L)**, 0 errors, 0 safety
> interventions. **The greedy control is what makes that readable** — their agent is
> genuinely strong and NOT crippled by the adapter; it simply loses this specific matchup.
> Likely mechanism: their belief templates cover Grimmsnarl and Great Tusk/Crustle but
> **not the Alakazam mirror**, which is exactly what our champion is. Caveat recorded: one
> opponent — and the one that hits their blind spot — is not a ladder evaluation.
> **(6) OPERATIONAL: the biggest gain of the whole session came from a re-upload, not an
> algorithm.** The user resubmitted the SAME `imitation-final` binary and the two slots went
> 757.4/776.7 -> **812.2/801.5** purely from re-rolling the day-1 opponent draw, confirming
> M29/M43's field-noise arithmetic live. That also RAISES the bar for any future upload: a
> new candidate evicts a converged, still-climbing 801.5 to restart at 600, and at 812 we
> are already above the middle of the historical day-1 close range (684-862), so re-rolling
> again now has WORSE odds than it did before.
> **Reusable:** `player_seats` with a name set (any future renamed teacher is covered, with
> a test pinning it); `m49_resolve_teacher_names.py` (derive a team's names from the API by
> `submissionId` and check none collides with an opponent); `m49_verify_teacher_deck.py`
> (pre-training gate: multiset deck identity + pilot-vs-field decomposition, fails loudly on
> an empty sample); `m50_notebook_arena.py` (run ANY Kaggle-style `agent(obs_dict)` inside
> our arena — every public competition notebook is now measurable against the champion in
> minutes); the enlarged corpus, with the 2,330-game one preserved.
>
> **⚠️ M47-M48 (2026-08-13/14) — read [m48_findings.md](m48_findings.md) then
> [m47_findings.md](m47_findings.md). Self-play RL closed with a mechanism, a real search
> bug found and fixed but still not enough, and a "the field is clone-dominated" hypothesis
> tested on the real ladder and refuted. Nothing shipped; `imitation-final` unchanged in
> both slots.**
> **M47** fixed self-play RL's four broken inputs (missing PPO clip, unscaled `tau`,
> stale M39 opponent weights, base = a weaker pre-M42 checkpoint) — `probe_agree` drift
> vs the frozen init went from M38's 0.966→0.8835 to 0.973 by the same iteration. All 5
> resulting candidates still failed the n=600 arena (pooled 0.511 [0.491, 0.531], CI
> touching 0.500). **M48 found why, arithmetically:** `Var(A)` for `A = R - V(s)` is
> **87.5% BETWEEN-game** even with M47's refit critic (AUC 0.79) — identical for all ~45
> MAIN decisions of a game, so the gradient direction is dominated by "which games did I
> win" and re-rolls every batch (mean consecutive-step cosine **-0.283**, worse than a
> random walk). Fixed with TD/GAE (`gae_lambda` in `rl_selfplay.py`, off by default):
> mechanically it worked perfectly (cosine **-0.283 → +0.234**, between-game variance
> **87.5% → 2.5%**) — but the resulting policy lost the arena **with confidence** (0.448
> [0.409, 0.488]), and its EXACT reflection through the champion also lost (0.427
> [0.388, 0.467]). Both signs losing means the champion sits at a local optimum along
> that weight-space axis — self-play's objective (beat 3 fixed clones) points somewhere
> that isn't ladder strength. **Self-play RL closed with 3 independent confirmations
> (flat / confidently worse / confidently worse reflected) and a mechanistic explanation,
> not just "didn't work".**
> **Search reopened, differently than M18 expected.** `v_alakazam.json` — M47's RL
> critic, AUC 0.65 on self-play states — scores **AUC 0.755 on imitation-final's own 135
> real ladder replays**, because it was fit on Yushin's full-field 2330-game corpus, not
> a 3-opponent self-play mix (M18's actual blocker). No new critic needed. Found and
> fixed a real bug instead: `ImitationSearchPolicy` (via the shared `SearchPolicy`) ran
> search on **every** decision context, not just MAIN — silently overriding the four
> specialized context heads M42/M43 built (the single largest measured gain in the
> project, +0.096 to +0.117 in this exact arena) with a 7-feature board evaluator never
> validated for "which card to fetch" or "which bench slot to promote". Fixed
> (`ImitationSearchPolicy.choose` now routes non-MAIN straight to the champion); 20 new
> tests. Score vs champion rose 0.280 → 0.330 at n=100 — real, but still confidently
> below 0.500, and greedy_bias sweeps plateau around 0.27-0.28 well short of parity. The
> n=600 Gate A/B protocol was not run given the time budget and this signal.
> **A user-proposed reframe — most ladder opponents are themselves clones, so
> hand-disruption should be a generally winning strategy because it exploits shared clone
> fragility, not "better TCG"** — was tested on 4,622 real games from Kaggle's daily
> dump (M46), zero new games, after building the disruption card list from actual card
> TEXT (catching a real error in project memory: Petrel/Spikemuth Gym do NOT attack hand
> size, they search cards into YOUR OWN hand). Paired comparison (exactly one side runs
> disruption tech, 960 mixed pairings): the disruption side **loses** with significance,
> **0.425 [0.394, 0.457]** — the hypothesis is refuted at ladder scale, not confirmed.
> An unplanned secondary finding from the same measurement: card-type LOCKS (a different
> mechanism — restricting what the opponent can *play*, not reducing hand *count*) show a
> real positive signal, 0.605 [0.577, 0.631] — correlational, not acted on.
> **Reusable:** `m47_step_coherence.py` (coherence-vs-noise gate for any future RL run,
> should be run BEFORE spending arena time, not after — M47 broke this rule once, at a
> cost of ~95 min); `m48_advantage_variance.py`; `rl_selfplay.py`'s `gae_lambda`/`--clip`/
> `--tag`; `m48_reflect.py` (reflect a measured-bad weight direction to test local-optimum
> vs wrong-sign); the search context-routing fix; `m48_fit_ladder_critic.py`;
> `m48_ladder_disruption_check.py` (paired-comparison template for any future
> card-mechanic-vs-ladder-outcome hypothesis).
>
> **⚠️ M34 UPDATE (2026-07-29) — read [m34_findings.md](m34_findings.md) FIRST. First POSITIVE
> gate since M28, and it came from the context nobody had ever measured.** M33 left the assembly
> delay unexplained; M34 found it in **TO_HAND** — the tutor-resolution context (Dawn ×4 / Hilda ×4 /
> Poké Pad ×4 / Poffin ×4 = "which combo piece do I pull"). Every upgrade since M28 went to MAIN;
> **TO_HAND was still LINEAR at 0.606** (greedy 0.512). **(1) Measured (`analyze_tohand_divergence.py`,
> 10,393 teacher decisions re-scored, prize picks dropped, card-id MULTISET matching, live
> lowest-index tie-break):** on turns 1-5 the teacher fetches a combo piece **59.9%** vs the clone's
> **41.9%**; the #1 disagreement is **"teacher fetched Kadabra, clone would fetch Dudunsparce" ×914**
> — it swaps combo piece for draw engine. Pre-registered gate **G-0 passed 3/3**: early non-trivial
> agreement 54.2%; on the 3,506-decision tradeoff subset **T=0.494 vs C=0.210, McNemar p=6.3e-141**;
> 2,128 early disagreements (1.73/game). *The script's written prior was that G-0b would FAIL — the
> prior lost, not the gate.* **(2) MECHANISM, not a guess:** splitting on a variable the model cannot
> see, the teacher's combo rate falls **59.1% → 30.6% → 12.1%** as Kadabra then Abra are already in
> hand ("don't fetch a piece you hold"); the clone is flat-to-inverted and misses the most common
> cell (n=2249) by **39 points**. In a TO_HAND decision every option is `OptionKind.CARD`, so blocks
> A/C/E/F/**S** and **A⊗S** are identical across options and **cancel in the softmax** — the only
> state-conditioned channel is **B⊗R** (card-id ⊗ `reduced`), and `reduced_alakazam`'s 9 scalars carry
> **no Abra/Kadabra hand count**. An OBSERVABILITY gap, not a capacity gap. **This also explains M31:
> `ALAKAZAM_V2` enriched only `snapshot` and left `reduced_len=9` — verified, 1269−(74+76+12·76)=207=23×9
> — so V2 never touched the one block that could help.** **(3) FIX — new additive profile
> `ALAKAZAM_FETCH` (dim 750):** `reduced` 9→13 with Abra/Kadabra/Alakazam-in-hand + Abra-in-play.
> **G-1 (linear A/B, same split seeds as every ALAKAZAM experiment): TEST 0.6074 → 0.7104, non-trivial
> 0.5256 → 0.6393 = +0.114, and the overfit gap SHRANK (−0.0048).** Bar was +0.02. Baseline reproduces
> the shipped 0.606. **(4) SHIP MECHANISM — `policy.py` now resolves a DeckProfile PER CONTEXT**
> (generalises what `mlp_switch` already did), so TO_HAND scores under ALAKAZAM_FETCH/750 while
> **MAIN's proven MLP ensemble stays ALAKAZAM/658 byte-for-byte** — no expensive MAIN retrain, and
> `build_fetch_weights.py` hashes MAIN before/after and refuses to write if it changed.
> **(5) G-3 BEHAVIOURAL GATE — FAILED, and this is the most valuable result of M34.**
> `assembly_turn_gauntlet.py`, 25 field decks × 8 games = **200 games/arm**: "Alakazam active with
> {P}" **5.43 (champion) vs 5.41 (candidate), delta −0.02, 90% CI [−0.50, +0.49]**. Bar was ≤ −0.25
> with the CI excluding 0. Instrument is calibrated (champion's simulated 5.43 ≈ its real-replay 5.46).
> **Crucially this is NOT an 8th "the proxy lied": the intervention verifiably WORKS.** Re-scoring the
> teacher's own states with the candidate (`analyze_tohand_divergence.py --weights`): early non-trivial
> agreement **54.2% → 68.5%**, combo fetch rate **41.9% → 52.1%**, tradeoff-subset C **0.210 → 0.371**
> — **57% of the behavioural gap closed**. The previous seven failures never checked whether the
> intervention changed behaviour at all; this one did, and the outcome still didn't move. **So the
> finding is causal, not statistical: fetching the right card does NOT control assembly speed in this
> deck** — Dudunsparce draws 3, so the clone's "wrong" route finds the piece anyway, just the long way
> round. **This falsifies M33 hypothesis (B) (sequencing/priority) via TO_HAND and leaves (A)
> piece-loss/resilience as the only one of the two still unmeasured.**
> **(5-bis) RE-RUN IN THE RIGHT FIELD — still fails, nominally WORSE.** The cached gauntlet field is
> the WRONG field for this question: only **6 of 120 decks (5%) are Marnie's Grimmsnarl** (and 1 is
> Team Rocket) because it is extracted from `Logs/Submission {greedy v5, imitation v1, v2}` — our
> opponents from the ~650-elo era — while M33's 0.9-turn gap is **Grimmsnarl-specific** (58.1% of the
> teacher's field, 27% of ours). Added `--marker` and re-ran on those 6 decks × 24 games = **144
> games/arm**: champion **5.86** vs candidate **6.09**, **delta +0.23** [−0.56, +1.05]. The instrument
> correctly reproduces the Grimmsnarl slowdown (5.86 vs 5.43 general), but **in the field where the
> defect lives the candidate is nominally SLOWER**. Neither delta is significant, but that is **344
> games/arm with no signal in favour** and the point estimate in the relevant field pointing the wrong
> way. **The "it was never tested where it matters" escape hatch is closed.**
> **(6) `head_to_head` — no veto, but NO real evidence either; the literal verdict misleads.**
> 120 decks × 10 games × 2 arms: fetch 0.969 vs mlp 0.959, **paired +0.010, 90% CI [+0.000, +0.020]**,
> 26 decks better / 16 worse / **78 tied**. It prints "BEATS" only because `lo > 0` in the 3rd decimal
> — **the CI touches zero**, both arms are **saturated at 0.96-0.97** (strong_gauntlet.py exists
> because this field saturates ~0.90), and M14 calibrated this exact instrument reading a real +0.13
> ladder gap as −0.003. The one solid read: **0 interventions in 1200 games** — robust, won't crash.
> **(7) MAIN A/B (3h42m): +0.0063**, far under the +0.02 bar — the 4 features are TO_HAND-specific and
> inert in MAIN, so leaving MAIN at 658 costs nothing and a dim-750 MAIN MLP retrain is NOT worth it.
> **DISPOSITION: do NOT promote on offline merit — the pre-registered gate failed and promoting after
> that is moving the goalposts. Recommended instead: a CONCURRENT ladder free-roll** (champion vs
> candidate in the 2 active slots, same field/epoch, ≥50 eps — per M29 a lone converged score carries
> ±60-100 elo of field noise), decided before the ~2026-08-10 new-agent cutoff. Bundle
> `build/imitation-fetch.tar.gz` built + validated + smoke-tested, **not uploaded**. 217 tests pass.
> **Reusable regardless of the outcome:** the per-context profile override (change ONE context without
> retraining the expensive MLP), `assembly_turn_gauntlet.py` (first behavioural-in-simulation
> instrument; also found that `BattleRunner` never calls `on_battle_start/end`), and
> `analyze_tohand_divergence.py --weights` (the "did the intervention actually change behaviour?"
> check that was missing from all seven earlier mispredictions). New:
> `analyze_tohand_divergence.py`, `triage_tohand_profile.py`, `assembly_turn_gauntlet.py`,
> `build_fetch_weights.py`.
>
> **✅ M42 (2026-08-09) — read [m42_findings.md](m42_findings.md) FIRST. The first MEASURED
> improvement to the clone since M28, and it came from a surface nobody had audited: WHICH
> DECISION CONTEXTS THE CLONE LEARNS AT ALL.**
> M31–M41 all attacked the same surface (MAIN ranking fidelity, or the deck). An audit of Yushin's
> 2,330 games (**190,731 decisions**, vetted `iter_player_decisions`) against the shipped payload's
> 10 contexts found **12.2% of all decisions (23,354) have NO model** — they fall to `GreedyPolicy`,
> and the most frequent to `_safe_default` ([greedy.py:339-345](../src/ptcg_ai/decision/greedy.py#L339)),
> which returns `list(range(minCount))` = option 0, **with no look at the state**.
> **(1) WHY THEY WERE NEVER TRAINED — a threshold artifact, not a judgment.**
> [`train.py:284`](../src/ptcg_ai/imitation/train.py#L284) drops a context unless it beats greedy by
> `_MIN_LIFT = 0.05`. Greedy sits at 0.934 on ACTIVATE, so the bar demands 0.984 — **unreachable by
> arithmetic**. It had been hiding the game's 3rd most frequent decision since M7.
> **(2) TWO 100%-SATURATED DIVERGENCES, verified on our OWN ladder replays:** ACTIVATE (6.55/game) —
> teacher declines 6.6%, **clone 0 of 760**; SETUP_BENCH_POKEMON (0.45/game) — teacher declines
> 37.4%, **clone 0 of 45**. P ≈ 3e-23 and 1e-9. Structural impossibility, not noise.
> **(3) ARM A (ACTIVATE head, ZERO `src/` changes):** TEST **0.9408 → 0.9742 (+0.0335)**; behavioural
> gate G-A2 — candidate declines 4.88% (teacher 5.92%), **recall 0.695**, false-decline 0.8%.
> **(4) ARM B (count head) — the sharpest finding.** `_top_k` takes `min(maxCount, n)`; no RANKING
> model can express "bench fewer" at any capacity. Mechanism measured BEFORE building: Yushin's setup
> rule is deterministic — **Abra 673/673 benched, Dunsparce 146/146 benched, Fezandipiti ex 0/261,
> Shaymin 1/211**. `Fezandipiti ex` is an **ex = 2 prize cards**; the champion benched it **64 of 64**
> times on held-out data, the candidate **0 of 64**, in a deck whose losses are 88.5% prize races.
> TEST **0.4706 → 0.9804 (+0.5098)**. Only `src/` change: `count_heads`, **absent by default ⇒ every
> existing agent byte-identical** (M27 `recover_rule` pattern), prediction **clamped** to the legal
> band. 8 new tests; **250 pass**.
> **(5) ARM C (TO_HAND MLP ensemble):** **0.7517 → 0.7949 (+0.0432)**, and TEST > val — the OPPOSITE
> of the M11/M14 overfit signature. Caveat: the shipped baseline trained on 80% of games vs the
> candidate's 60%, so the comparison HANDICAPS the candidate (no leakage — same held-out 20%).
> **(6) THE DIRECT ARENA (n=600) — the instrument that discriminates, with its own controls:**
> candidate(A+B) vs champion **0.535**, champion vs champion control **0.492** (theoretical 0.500 —
> at n=200 M41 read 0.435, so n=600 is the minimum usable), champion vs Grimmsnarl **0.655** here vs
> M41's independent **0.658**. The +0.035 is z=1.71, **p≈0.043 one-sided — suggestive at 90%, NOT
> conclusive at 95%.** **Against it: candidate vs Grimmsnarl 0.640 vs champion's 0.655 (−0.015, not
> significant), the same direction as A+B+C's frequency-weighted gauntlet slice. That is where this
> would fail on the real ladder if it fails.**
> **(7) THREE BUNDLES BUILT + VERIFIED EXTRACTED, NONE UPLOADED** — `imitation-ctx-a` (A),
> **`imitation-ctx` (A+B, RECOMMENDED)**, `imitation-ctx-abc` (A+B+C: gauntlet macro +0.016 with a
> clean positive CI, but frequency-weighted −0.006 and no direct arena). MAIN is **byte-identical**
> in all three (sha `a206d7ed9c8e2181`).
> **REUSABLE:** `train_context_heads.py` (trains ANY context, linear+MLP, single/multi-pick; baseline
> is the SHIPPED artefact per the M37 rule; measures with the LIVE `_rank_order` tie-break, which
> differs materially from `np.argsort` — TO_HAND 0.7172 vs 0.6835); `count_heads` (general mechanism
> for "how many", not "which"); `build_fetch_weights.py` now grafts dict specs with per-context
> profiles + count heads and its sha guard covers **every** context (verified: the M34 path still
> rebuilds the shipped champion **byte-for-byte**); `verify_ctx_bundle.py` (drives the SHIPPED
> entrypoint on real observations and fails if the new head never fires — exists because M37's
> wrong-deck bundle passed both `validate_submission` and `smoke_test_entrypoint`).
> **⚠️ METHOD TRAP that nearly cost the milestone:** a first raw-replay sweep read `cell['action']`
> from the SAME step and reported the OPPOSITE result (75.8% declines). The real extractor
> ([kaggle_replay.py:97](../src/ptcg_ai/imitation/kaggle_replay.py#L97)) takes the action from
> `nxt[seat]`, **the NEXT step**. `scratchpad/_ctx_probe*.log` hold the defective version — do not
> cite them.
>
> **⚠️ M41 (2026-08-08) — read [m41_findings.md](m41_findings.md). Deck composition: NEGATIVE.**
> Swapping Enhanced Hammer for Night Stretcher on frozen weights looked like a win at n=200 (+0.115
> in the mirror, POOLED +0.012) and **evaporated at n=600** (−0.023, POOLED −0.013). The calibrator:
> the OLD deck against ITSELF read **0.435 at n=200** and **0.503 at n=600** — a mirror must be
> 0.500, so n=200 in this engine measures nothing. Nothing adopted; `decks/yushinito_v*stretcher.csv`
> kept as a record, NOT shipped. M42 later confirmed Yushin ran **ONE decklist across all 2,330
> games**, byte-identical to ours — there is no revised list of his to copy.
>
> **⚠️ M40 (2026-08-08) — read [m40_findings.md](m40_findings.md). The last open hypothesis from
> M33 is measured and falsified: NO open causal hypothesis remains on the decision policy.**
> M33 left two candidate causes for the clone's slower Alakazam assembly (Powerful-Hand-legal turn
> 5.46 vs Yushin's 4.56): (B) bad play priority — falsified by M34 (TO_HAND rebuild verified a real
> behavior change, closed 57% of the choice gap, still zero assembly/win effect); (A) piece loss to
> disruption — tested here. Measured on 2,330 master replays + 133 clone replays: losing an Abra/
> Kadabra copy pre-assembly costs the master **+1.71 turns** (real, confirmed mechanism), but the
> clone does **not** recover worse when it happens — recovery rate 28.4% [19.0-40.1%] vs the
> master's 21.8% [19.6-24.1%], overlapping, if anything nominally higher for the clone. The
> pre-registered 3-gate check (recovery gap > 0.30 / delay ≥ 0.5 turns / ≥20 opportunities each)
> fails on the central condition, so nothing was built — a forced-recovery rule (the M27 mechanism,
> already wired in `policy.py:255-284`, never applied to Alakazam) would have made the clone LESS
> like Yushin, not more: he only plays the recovery item 22% of the time it's legal, not on sight.
> **That closes both of M33's hypotheses with direct measurement.** Combined with the five null/
> negative fidelity levers (M31/M36/M37/M38/M39), that's seven intervention attempts plus two
> root-cause tests, all null or negative. **Recommendation: no further policy-level lever is known
> to try — consolidate the final submission instead of searching for an eighth.**
>
> **⚠️ M39 (2026-08-05) — read [m39_findings.md](m39_findings.md). Decision-importance
> reweighting is ALSO a wash, and M38's RL closes on a confirmed-negative reweight.**
> **(1) RL CLOSED FOR REAL.** Reweighting the M38 self-play opponent mix (25%→40%
> Kangaskhan, to fix its collapse) made everything worse with a CI that finally excludes
> the tie: head-to-head vs the frozen init went from 0.550 [0.481-0.619, tie] to **0.438
> [0.382-0.494, WORSE]** at n=300, and Kangaskhan itself barely moved (19.2%→21.0%).
> Across all 4 evaluations taken (it4/6/8/10) the candidate was NEVER confidently better
> than the frozen init — twice a tie, twice confidently worse. Do not reopen self-play RL
> on this base without a fundamentally different lever (much longer training, different
> objective shaping); repeating M16/M19/M38's setup is a closed question now three times
> over, across three different base-policy strengths.
> **(2) M39 — weight MAIN decisions by importance: NEGATIVE.** Hypothesis: M33 measured
> Yushin fires Powerful Hand on 99.6% of legal turns (no "when" to learn), so the real gap
> is ASSEMBLY SPEED (PH-legal turn 4.56 vs 5.46) — upweight (×3) the ~17.8% of decisions
> before PH first becomes legal each game, leave the rest alone. Same trainer/split as the
> champion (`train_mlp_alakazam.py`), val/test never weighted. Result: TEST **0.7670
> (control) vs 0.7659 (boosted)** — a −0.0011 difference, smaller than the ~0.008 seed-to-
> seed spread of the SAME config. Known gap: model weights were never saved to disk, so
> whether the boost helped specifically ON the assembly-phase slice (while being a wash
> elsewhere) was never checked — would need a full ~4h retrain to verify, not judged worth
> it at 5-for-5 negative.
> **(3) The full scoreboard, five independent levers against MAIN fidelity, all null or
> negative:** M31 width +0.006 (noise) · M36 +42% data → −0.005 · M37 transformer +0.017
> (below its own +0.030 bar) · M38 self-play RL never confidently better, twice confidently
> worse · M39 importance-weighting −0.0011 (noise). None of these touch the root cause M38
> measured directly: the clone leaves Yushin's line every ~4.7 decisions and then plays
> states with zero training signal. The only lever that attacks that (DAgger / interactive
> correction) needs a queryable oracle we don't have — only recorded replays, not a live
> Yushin. **Recommendation: stop searching for a policy/fidelity win and consolidate** —
> `imitation-setxf2-fixed` and `imitation-mlp-fetch` are both converging around 885-890
> after a full day of ladder games and are statistically indistinguishable from each
> other; spend remaining runway on a clean final submission, not a sixth lever on the same
> diagnosed bottleneck.
> **(4) Operational: a laptop sleep/hibernate silently paused a ~2h-estimated run for
> ~4.5 hours** (low-battery hibernate while the machine was unattended; the process
> survived and resumed on wake, but don't assume `powercfg standby-timeout-ac 0` prevents
> this — it only blocks IDLE sleep, not lid-close or critical-battery hibernation). Keep
> the machine plugged in and the lid open for any multi-hour background run.
>
> **🚨 M37 ADDENDUM (2026-08-04) — submission 55203764 (~500 elo) IS VOID: WRONG DECK SHIPPED.**
> The bundle carried the SDK's sample deck (Snover/Mega Abomasnow) next to ALAKAZAM-trained
> weights: **0 of 9 distinct cards in the profile vocabulary**. Everything hashed to the OOV bucket,
> so the scorer saw identical features for every option — a blind agent. Bench at turn 3/5/8 was
> **0.70/0.91/1.00** vs the champion's 3.77/4.43/4.39; **41% of losses had ≤1 Pokémon in play**;
> five losses took **zero prizes**, dead by turn 4. **That elo says nothing about the transformer.**
> Cause: `build_submission` reads `config.paths.deck_path`, which defaults to
> `sample_submission/deck.csv`; the build never overrode it. **Nothing caught it** —
> `validate_submission` passes, `smoke_test_entrypoint` passes with `bc_failures: 0` (an unknown
> card scores as OOV, it does not raise), and the extracted-tarball check verified profile and
> `feature_dim`, both of which were CORRECT. Fixed by `builder._assert_deck_matches_profile`
> (aborts under 80% deck/vocabulary overlap) + `tests/unit/test_submission_deck_profile_match.py`.
> Correct bundle rebuilt: **`build/imitation-setxf2-fixed.tar.gz`**, deck 22/22 = 100%, not uploaded.
> **Also settled: there is NO timeout problem** — 333 ms/decision, max **32.8 s/game** against the
> 600 s budget, guard never fired. The cost worry below was real in projection, not in practice.
>
> **⚠️ M37 (2026-08-03) — read [m37_findings.md](m37_findings.md). The set transformer was BUILT
> and WORKS, but misses its pre-registered bar; and the field gauntlet is SATURATED.**
> **(1) THE MODEL SHIPS TECHNICALLY.** `src/ptcg_ai/imitation/setnet.py` — hand-written stdlib
> multi-head attention/layernorm/softmax — matches torch at **8.9e-16** (bar was 1e-9). 235 tests
> pass; bundle `build/imitation-setxf2.tar.gz` (11.1 MiB, **k=3**) verified EXTRACTED with
> `_IMITATION_READY: True`, `set_contexts: ["MAIN"]`, `bc_failures: 0`. **NOT uploaded.**
> **(2) IT MISSES THE BAR, for a real reason.** Strict (card-identity) fidelity on the held-out 20%
> vs the SHIPPED MAIN scorer: mlp **0.768**; setxf2 k=1 **0.763 (−0.006)**, k=3 0.777 (+0.009),
> k=5 0.783 (+0.015), k=7 **0.786 (+0.017)**. Bar was **+0.030**. Colab's "+0.048" compared against
> its OWN retrained `mlp` arm (0.7561), **not** the shipped `bc_alakazam_fetch.json` MAIN — which is
> also 3×h=48 but better trained. **Rule: a sweep's baseline is the SHIPPED artefact measured with
> the same instrument, never an internal arm.** Cost is also real: p99 **243 s/game at k=3**, 567 s
> at k=7, vs a 600 s per-agent-per-episode budget.
> **(3) THE GAUNTLET CANNOT VETO ANYTHING — it is saturated.** Baseline macro **0.965**, **67/72
> decks at 100%**, 0 below 50%, **Grimmsnarl 9/9 at 1.000** — while the real ladder loses that
> matchup at ~37%. The field was rebuilt from `imitation-fetch`'s REAL replays (submission
> **55145833**, 134 downloaded, 72 distinct decks) and is *still* saturated: the decks are right, the
> **pilots** are wrong (greedy-piloted Grimmsnarl is trivial). Second, unfixed defect: the macro is
> **unweighted over distinct decks**, so Grimmsnarl gets 12.5% of the veto's weight against 30.8% of
> real episodes. `LADDER_FOLDERS` now points at the fresh replays.
> **(4) REUSABLE:** `setnet.py`; `policy._score_options` (the DECISION-level entry point — `_score`
> was per-option BY SIGNATURE, which made "best of what I hold" inexpressible at any depth; both
> call sites now route through it); a per-episode **time guard** with staged degradation, hooked to
> the only episode boundary Kaggle exposes (`select is None`, the deck request — the M36 note that
> no such channel exists is **wrong**); `head_to_head_par.py` (multiprocess gauntlet, verified
> **bit-identical** to serial; a numpy scorer was rejected — 7× but differs 1.33e-15, enough to flip
> a near-tie); `pareto_k.py` (many scorers in ONE pass over the held-out set).
> **(5) OPEN:** the Grimmsnarl-slice measurement was cut off mid-run — it is the only check that
> could still justify shipping, since ~half the corpus is that matchup. Re-run
> `PYTHONPATH=src python -u scratchpad/pareto_k.py`. Offline instruments are otherwise exhausted, so
> the remaining honest test is a **concurrent ladder free-roll** (champion + candidate in the two
> active slots, same field/epoch, ≥50 eps — M29). New-agent cutoff ~2026-08-10.
>
> **⚠️ M35+M36 (2026-07-31) — read [m36_findings.md](m36_findings.md) then [m35_findings.md](m35_findings.md).
> Two negatives, one instrument bug found, and the champion's decline explained.**
> **(1) DIAGNOSIS OF THE DECLINE — it is DECK COMPOSITION, not a harder ladder.** Champion's own 197
> episodes by quartile: opponent mean rating **887.9 → 855.3 (DOWN)** while our WR went
> **47.9% → 41.2% (DOWN)**. We face *lower-rated* opponents and win *less*. Arithmetic fit: we score
> 71.4% vs Alakazam and 37.0% vs Grimmsnarl, so shifting ~18 points of field between them predicts
> −6.2; observed −6.7. Grimmsnarl went 17%→51.3% of the top band (Jul 17-26) and is diffusing down.
> **(2) INSTRUMENT BUG (R21) — `train.py::_bc_accuracy` compares OPTION INDEX sets, not card
> identities**, so picking a different copy of an identical card scores as an error. Decks with many
> duplicates are systematically understated: Luca's TO_HAND read 0.543 but is **0.736**; MAIN 0.599 →
> **0.626**. **Yushin's MAIN correction is EXACTLY 0.000** — it is a deck-specific distortion, not a
> universal flatterer. New `scratchpad/semantic_fidelity.py` (strict key = card id + the target's
> unique `serial`, so two same-species Pokemon with different damage stay distinct; a "loose" key
> inflates MAIN to 0.674 and is WRONG). Handles linear + `mlp_ensemble`; **auto-validated** by
> reproducing M28's documented 0.780 on the champion. **Training was never affected** — duplicate
> options have identical feature vectors so the softmax is symmetric; only measurement broke.
> `train.py` deliberately NOT patched (it is also the L2 selection criterion).
> **(3) M35 — clone Luca (top-5 Grimmsnarl, ~1194): CLOSED NEGATIVE at G-2.** Hand-authored
> `GRIMMSNARL` profile built, registered, dim 706 pinned, `wants_fn` trap verified fixed (Munkidori/
> Froslass/Snorunt read a permanent 1.0 under the generic fallback). G-1 passed by 0.006 (MAIN 0.626
> strict vs 0.62 bar) but **below Yushin's 0.645**. G-2 failed: MLP lift +0.056 (M28 got +0.224) with
> the overfit gap **34× worse** (+0.0415 vs linear +0.0012) — the M11/M14 small-data signature.
> Root cause is a hard **data ceiling: 539 episodes total**, all downloaded. Not tunable.
> **(4) Team Rot-Weiß (Alakazam, peaked 1148, 480 eps) SCREENED AND REJECTED** — plays the
> byte-identical deck, but **48.4% vs Grimmsnarl where Yushin gets 51.9%**, sits at 1084 ≈ Yushin's
> 1096, and has less data. No advantage on any axis.
> **(5) M36 — more data for the Yushin clone: NEGATIVE.** +278 new replays → 1829 games, MAIN
> **92.231 (+83% over the 50.323 the champion ever saw)**. Champion re-scored on the enlarged
> held-out set = **0.767** (split is stable by game_id hash, so no leakage). New MLP **seed 0 =
> 0.7622 — below it**; stopped after one seed because the ~0.006-0.010 seed spread made the 0.777
> gate arithmetically unreachable. **But the LINEAR rose 0.556 → 0.588 (+0.032)**: the new data has
> signal the h=48 MLP cannot extract. **This is an ARCHITECTURE ceiling, not a data ceiling.**
> **STILL UNTESTED (the live lead):** the shipped net is `658 → 48 → 1` — **one hidden layer**, ~32k
> params. M31 closed *width* (h=48→96 = +0.0056, inside seed noise), never *depth*, and nobody has
> tried **cross-option context**: every option is scored in total isolation before the softmax, so
> "best of what is available" is inexpressible at any depth. Ladder: **DeepSets first** (pooled
> option-set summary as extra features — trains with the existing trainer, ships with the existing
> `_mlp_forward`, hours), then a set transformer only if that fires (compute fine at ~1.1M MAC/decision
> vs the 600 s budget, but needs hand-written stdlib attention/layernorm/softmax — days).
> **Caveat on all of it:** Yushin takes **401 ms/decision** vs 123-184 ms for the reactive top-8, so
> he may carry internal state/search — in which case a fidelity ceiling exists that no architecture
> crosses. Consistent with three independent levers each yielding ~+0.005.
>
> **⚠️ M33 UPDATE (2026-07-29) — read [m33_yushin_deep_dive.md](m33_yushin_deep_dive.md) FIRST.
> It invalidates the premise of the whole M32 line.** First dedicated study of the TEACHER
> (1551 eps on disk → **1313 non-mirror games, 747-566 = 56.9%**, 59,136 MAIN decisions).
> **(1) There is no "when to attack" decision to learn: Yushin fires Powerful Hand on 99.6% of
> the turns it is LEGAL (6343/6368), with a legal→fired lag of +0.00 turns — identical whether
> lethal (99.6%) or not (99.6%).** The 18-20% figure everyone (including M32) quoted is a
> per-DECISION dilution artifact — M20's lesson, re-learned. **(2) On the teacher's OWN states the
> clone is ALREADY more aggressive than him (fires PH 22.7% vs 21.3%, n=8124).** **(3) So the gap
> is ASSEMBLY SPEED, not decision-making:** clone gets PH legal at turn 5.46 vs teacher 4.56 vs
> Grimmsnarl, and it is already behind at the first milestone (Alakazam in play 5.16 vs 4.42).
> The teacher's own win/loss line sits exactly there — **PH legal turn 4.21 in his wins vs 4.95 in
> his losses.** **(4) The clone's hand-building is NOT the problem** (hand-size-per-turn curves are
> indistinguishable). **(5) Divergence analysis (17,709 teacher decisions, 78.6% agreement):** the
> dominant disagreement is **PLAY→PLAY (29.2%)** — same intent, different card; agreement is worst
> on PLAY (68.4%) and ATTACH (70.4%), near-solved on EVOLVE (92.1%). **(6) His worst matchup is NOT
> Grimmsnarl (51.9%) but Team Rocket (33-93 = 26.2%)**, never examined; his 566 losses are 88.5%
> prize-race, 7.8% bench-out, 3.7% deck-out. **DEAD HYPOTHESES — do not retry:** "the clone
> hesitates / fires PH too late", "the teacher reacts to Munkidori", "teach it a firing criterion",
> "the clone draws/manages its hand worse". New read-only tools `scratchpad/analyze_yushin.py`,
> `analyze_ph_availability.py`, `analyze_yushin_divergence.py`. No `src/` change, nothing uploaded.
>
> **⚠️ M32 UPDATE (2026-07-28) — read [m32_findings.md](m32_findings.md) first.**
> First loss diagnosis of the CHAMPION `imitation-mlp` (Yushin/Alakazam MLP clone) on its own
> ladder replays (`replays/55011997`, 100 eps, **53W-46L**). **The gap is a MATCHUP WALL, not a
> pilot defect:** three decks take **27 of 46 losses** — **Marnie's Grimmsnarl ex 10W-17L (37%,
> n=27)**, **Dragapult ex 0-6**, **Mega Kangaskhan 0-4 (all deck-out)** — while the champion is
> **43W-19L (69%)** against everything else. This was invisible because the inherited archetype
> marker table had no marker for any of them (all fell into `other`); markers 648/756/121/1191/104
> added. The Grimmsnarl deck is close to a designed counter (Munkidori *moves* the damage counters
> Powerful Hand places; Unfair Stamp/Petrel/Spikemuth Gym attack hand size = our damage dial;
> Froslass snipes the Abra line). Everything else is CLEAN — **parity 100.0% (5999/5999) in wins
> and losses, 0 SafePolicy failures, lethal 96.2% vs the teacher's own 96.3%**, behaviour faithful
> on every axis. One real lead: **combo assembly is a median 1 turn slower than the teacher, and
> ~15% of losses never assemble Alakazam (teacher 4%)** — but reverse causality is not excluded.
> **DISCRIMINATOR (same day, `scratchpad/diagnose_mlp_teacher_matchup.py`) — measures Yushin's OWN
> record per archetype over his 1063 non-mirror games and CORRECTS the above:** teacher **57.6%**
> vs clone 53.5%. **"Three hard counters" is RETRACTED** — the teacher goes **71% vs Mega
> Kangaskhan** and **57% vs Dragapult**, so our 0-4/0-6 are small-sample noise, not counters.
> **Grimmsnarl is the one real deck-level wall AND it is the meta:** it holds the world #1 to
> **53% over 603 games** and is **57% of HIS field vs 27% of ours** (he sits ~1230, we sit ~889) —
> it owns the TOP of the ladder, so we meet more of it as we climb. The clone's deficit is
> **broad, not matchup-specific** (~4 points uniform). **Ceiling: a PERFECT clone of Yushin on
> this deck is worth 57.6%** — all the headroom the imitation line has left here, and M31 closed
> every lever to claim it. **Suggested next regime change: clone a GRIMMSNARL pilot** (must clear
> greedy-pilotability M6 + clonability + the head_to_head veto).
> **RESIDUAL VERIFIED (same day, `scratchpad/bootstrap_ph_turn.py`) — the ~12-point gap unexplained
> by the deck's own difficulty + the clone's general deficit is REAL and MATCHUP-SPECIFIC.**
> Powerful-Hand first-use median turn, isolated to the 27 Grimmsnarl games: **6 (clone) vs 4
> (teacher)**. Bootstrap (20k resamples from the teacher's own 591-game distribution): **P(median
> ≥6 by pure sampling noise) = 0.1%** — not noise. Control in the clone's BEST matchup (Mega
> Lucario, 89% WR): clone median 4 vs teacher 4.5, **P=93.3%, indistinguishable** — the delay does
> NOT appear in a matchup the clone wins, so it's Grimmsnarl-triggered, not a generic clone trait.
> **This makes a Grimmsnarl/Munkidori-CONDITIONED feature (unlike M31's untargeted V2) a cheap,
> well-targeted hypothesis to try BEFORE the costlier clone-a-new-teacher path** — ceiling ~53%
> (the teacher's own matchup cap), not a full fix.
> **FIX ATTEMPT #1 CLOSED NEGATIVE (same day) — read [m32_findings.md](m32_findings.md) §1-quater.**
> Built `ALAKAZAM_MATCHUP` (dim 658→684, 2 additive flags for facing Munkidori/Grimmsnarl-line or
> hand-attack tech; 203 tests pass). The offline proxy (M31 Colab pipeline, extended to slice TEST
> by archetype, new `--emit-raw`) was weakly positive twice independently (Grimmsnarl slice
> +0.0095/+0.0066 then +0.0103/+0.0083 vs the rest). **The decisive check refuted it:** re-scoring
> the real 27 clone-vs-Grimmsnarl games (`scratchpad/counterfactual_ph_timing.py`, sanity-checked
> by exactly reproducing V1's bootstrap numbers) showed Powerful Hand fires **LATER**, not earlier
> (mean turn 5.46→6.21; of 24 paired games, 20 unchanged, 4 worse, 0 improved). **7th instance of
> "the offline proxy lied"** (M11/M14/M31) — the sharpest yet, since the proxy signal was targeted
> at the right slice and still didn't predict real behaviour. Working theory: both flags are
> "sticky" (stay 1.0 all game once triggered), carrying no per-turn timing signal. **No bundle
> built, no `head_to_head`/ladder slot spent.** `ALAKAZAM_MATCHUP` stays in the codebase (safe,
> tested, unused). **Remaining live path: clone a Grimmsnarl pilot** — candidates already mined
> from our own match history, no blind search needed: submissionIds 55011432 (1072 elo, 183 eps),
> 54785331 (977 elo, 669 eps), 54900560 (1013 elo, 443 eps), 54981508 (1036 elo, 231 eps).
> New tools `scratchpad/diagnose_mlp.py`, `scratchpad/diagnose_mlp_other.py`,
> `scratchpad/diagnose_mlp_teacher_matchup.py`, `scratchpad/bootstrap_ph_turn.py`,
> `scratchpad/linear_triage_matchup.py`, `scratchpad/counterfactual_ph_timing.py` (reusable for any
> future MAIN-scoring change, arguably the most valuable artifact of this attempt). Only `src/`
> change: the additive, tested, currently-unused `ALAKAZAM_MATCHUP` profile.
>
> **⚠️ M30/M31 (2026-07-26→28) — no banner here; see [m31_plan.md](m31_plan.md) for the full
> record.** Summary: all four levers for improving the Alakazam clone (data volume, MLP capacity,
> data weighting `alpha=0.5`, feature enrichment `ALAKAZAM_V2` dim 1269) measured **negative or
> unshippable**. V2 was built end-to-end and **lost the `head_to_head` veto −0.033 [−0.056,−0.011]
> despite +0.009 better MAIN fidelity** — the sixth confirmation that higher fidelity ≠ better
> play. **Keep the V1 `imitation-mlp` champion.** Colab GPU training is built and validated
> (`scratchpad/export_for_colab.py`, `colab_train_mlp.py`, ~80-90× speedup).
>
> **⚠️ M29 UPDATE (2026-07-26) — read [m29_findings.md](m29_findings.md) first.**
> The SAME `imitation-kanga` bundle posted 772.7 / 856.8 / 899.4 across uploads. Proved it's the
> IDENTICAL agent (byte-identical fixed deck == `decks/懒惰的金枪鱼.csv`; weights unchanged) — the
> ~127-elo swing is the OPPONENT DRAW, not the agent: the two runs faced NEARLY DISJOINT pools (2
> shared of 108 vs 69), and the 773-run hit a Mega-Lucario-heavy field (21× vs 3×, a known bad Kanga
> matchup) → WR 55.0% vs 59.7%. Chronology (EpisodeId order) is stable within-run, so it's between-run
> field, not a μ₀-streak. **A single converged score carries ~±60–100 elo of opponent-field noise →
> sub-±100 ladder comparisons are WEAK evidence** (this softens the M27 nsr[843]-vs-kanga[899] read:
> within noise). To compare agents, run them CONCURRENTLY (shared field/epoch) and judge the GAP after
> ~50 eps — do NOT re-roll one agent hoping for a lucky field. New read-only tool
> `scratchpad/compare_submissions.py`. No `src/` change, nothing uploaded.
>
> **⚠️ M28 UPDATE (2026-07-26) — read [m28_findings.md](m28_findings.md) first.**
> Answered the user's "can we clone with HIGHER fidelity?" by cloning the #1 (Yushin Ito, Stage-2
> Alakazam combo, 1000-episode archive) — the never-pushed TEACHER-STRENGTH axis (M23 had triaged
> Yushin out at the greedy gate without ever building a clone). **The 60–65% fidelity ceiling is real
> but BREAKABLE:** linear scales with data (generic 0.470 → ALAKAZAM profile 0.545 → full-1000g 0.603),
> and an **MLP on the big data lifts held-out MAIN 0.556 → 0.780 (+0.224) with a SMALLER overfit gap
> than linear (0.018 vs 0.038)** — the OPPOSITE of M11/M14 (big data + strong teacher ⇒ capacity
> captures skill, not noise). New additive `ALAKAZAM` DeckProfile (`deck_profiles.py`, dim 658; models
> Powerful Hand = 20×hand_size prose damage). The MLP clone PILOTS the combo (0 interventions) and
> **BEATS champion kanga on the gauntlet +0.074 [+0.052,+0.095]** (best gauntlet number ever). New:
> `scratchpad/train_mlp_alakazam.py`, `build_mlp_alakazam_weights.py`. Bundle
> `build/imitation-yushin-mlp.tar.gz` ready. **NOT uploaded** — offline fidelity has never predicted
> ladder (ITF beat kanga on gauntlet then lost); the free-roll is the judge, to be run CONCURRENTLY
> with kanga per M29. Tests 191 passed. **M27 nsr postmortem: the recovery rule HURT — converged plain
> kanga 899/905 > nsr 843; the early +37 was variance.**
>
> **⚠️ M27 UPDATE (2026-07-25) — read [m27_findings.md](m27_findings.md) first.**
> Acted on M26's root cause (55% bench-out losses = deck-structural, no plan B once Mega Kangaskhan
> falls) with the one aligned lever: a resilience card compatible with the existing deck. Chose
> **Night Stretcher (1097) ×2 for Hand Trimmer (1087) ×2** (recover the KO'd attacker; softest flex
> slot, matches the same-deck #9 pilot `whereismyorbit`). **A pure deck swap is a WEAK lever**:
> head_to_head is a dead-center tie, and a new de-risk probe (`scratchpad/probe_nightstretcher.py`)
> explains why — recovery OPPORTUNITY is abundant (95% of losses have a Kangaskhan/Crustle in discard)
> but the frozen clone PLAYS the card only ~7–9%, because a bolt-on card has no learned weight and
> rides an untunable generic PLAY-prior (the PASO 2 ceiling, quantified). **Fix (shipped in code, not
> uploaded): a DEFAULT-OFF forced-recovery pre-emption** in `ImitationPolicy._decide`, carried in the
> weights payload as `recover_rule={recover_id,clock_id}` (absent field = every existing agent
> byte-identical; no entrypoint/builder change). Fires only in MAIN when the clock (756) is gone from
> play+hand but in discard and the recover Item (1097) is legal → guarantees the recovery play. It
> fires 10×/150 games, ~doubles NS play (→15% games), passes the head_to_head veto with a **slight
> POSITIVE lean (+0.012, 40/26 decks)**, 0 interventions; 5 new unit tests, full suite **190 passed /
> 1 skipped**. Bundle `build/imitation-kanga-nsr.tar.gz` ready (dim 596, weights
> `bc_kangaskhan_1052_ns.json`). **Reusable lesson:** a small deterministic pre-empt in the rung-6
> policy is the clean way to make a bolt-on card actually fire despite the frozen prior — dim-safe
> (routing, not feature, change), ablatable, weights-payload-gated. **NEXT (M28, open): upload
> `imitation-kanga-nsr` as a ladder free-roll (only real judge of the resilience benefit vs
> Alakazam/Lucario/Archaludon), or generalize the pre-empt / return to the M23/M25 clone hunt.**
>
> **⚠️ M26 UPDATE (2026-07-23) — read [m26_findings.md](m26_findings.md) first.**
> Ran the M10-Phase-0 loss diagnosis on Kanga's real ladder logs (139 games, `replays/54911514`
> + `54893233`) for the first time, plus a clone-vs-teacher per-turn comparison. **Clean NEGATIVE for
> a decision fix — M10 repeating for Kanga:** bundle parity **100.0% (5981/5981), 0 SafePolicy
> interventions** (the ladder agent IS the trained clone), lethal discipline **98.8% with 0
> losing-game misses** (cleaner than the teacher's own 92.8%), routing normal (94% learned, no
> loss-concentration). The clone **matches/exceeds the teacher on every behavioral axis** — same
> first-attack turn, slightly more aggressive, and it benches DEEPER (3.09 vs 2.88) with identical
> bench-insurance (~89%), which **refutes** the "clone under-develops its board" hypothesis. The one
> new fact — **55% of Kanga's losses are bench-outs** (vs v1's prize-races) — is **deck-structural**
> (all-in Mega Kangaskhan, no secondary threat once answered), not a pilot defect. New read-only tool
> `scratchpad/diagnose_kanga.py`. **Two addenda, both NEGATIVE:** (1) the "more faithful deck than
> Kanga" hunt is closed — v1 has 0.862 fidelity yet is weaker (~686), every higher-fidelity candidate
> we dueled lost, and the lone lateral lead (Eduardo) resolved to unpilotable/loses-to-v1 (its
> promising 0.610 number was a stale mismatched artifact). (2) **`backstona` & `vvs` (the M25 "beat
> kanga at n=6" pair) run the BYTE-IDENTICAL deck to ITF** (18/18 cards) — three pilots of the same
> Mega Starmie/Crushing-Hammer disruption deck, and ITF is losing on the ladder, so they re-run a
> failing experiment; **do NOT spend a slot on them.** Of the tie-kanga candidates only **THIRD**
> (Team Rocket variant, a different clonable archetype) is a sensible ladder bet. **Nothing built/
> shipped; no `src/` change; 184–185 tests pass.** **NEXT MILESTONE (M27, decided by the user): try to
> improve Kanga directly — strategy search (domain-knowledge → search leaf-V, the M17 channel that
> worked) or pure self-play RL (e.g. extend REINFORCE to TO_HAND). The BC-clone-hunt lever is
> exhausted; the bottleneck is ladder slots + deck resilience, not a fixable pilot defect.**
>
> **⚠️ M25 UPDATE (2026-07-23) — read [m25_findings.md](m25_findings.md) first.**
> Third screening round: the next **20 candidates by leaderboard score** (ranks #125–#423), through a
> new self-tracking, two-tier-download pipeline. **18/20 confidently rejected** (8 fail greedy-
> pilotability — M6's wall; 4 lose the head-to-head by −0.08 to −0.38; 6 DISCARD_CLONE). Two
> (`backstona`, `vvs`) **beat kanga at n=6 with a clean positive CI — then COLLAPSED to a statistical
> TIE at n=12** on full-data weights (+0.018 [−0.001,+0.035] and +0.013 [−0.006,+0.031]), while beating
> v1 (+0.025/+0.032). **That is the exact signature of THIRD (ties kanga, beats v1) — the agent the
> user reports LOSING on the ladder.** The load-bearing result is not a new candidate; it is a live,
> **within-round** demonstration that the offline gauntlet is a reliable **VETO but not a RANKER in the
> ±0.05 band** (the same candidate flipped verdict just by doubling n). The gauntlet has now mis-ordered
> the ladder in every direction (kanga tie→+174 real; ITF beats-kanga→worse; THIRD tie→losing), and an
> offline Δ **cannot be translated to elo, not even in sign**. **NO candidate qualifies; nothing built/
> shipped; no `src/` change; 184 tests pass.** The real bottleneck is **ladder-test slots** (only the 2
> most-recent submissions are final-tracked, μ₀=600 restart, ~50 eps), not candidates — so generating
> more offline-indistinguishable candidates does not help. Recommendation: let the current ladder bet
> (THIRD/kanga) converge ~50 eps and judge on real data; if screening continues, act only on a candidate
> that FAILS cleanly or WINS by a large, sample-stable margin (+0.02–0.03 does not). New reusable:
> **`tools/next_batch.py`** (fixes the broken `screened` flag in `data/candidates_joined.json` by
> reconciling against disk — `decks/{slug}.csv` exists / sid in `candidates_m22*.txt` — then `--backfill`
> / `--mark`; now **42 screened / 247 remaining**) + the **two-tier download** (Tier-1 `--limit 40` for
> all, top up only survivors to 150 — ~1170 downloads this round instead of 3000).
>
> **⚠️ M24 UPDATE (2026-07-23) — read [m24_findings.md](m24_findings.md) first.**
> First-ever **pooling** attempt: concatenate the decisions of THREE independent top-20 pilots who run
> the EXACT same THIRD PTCG Club deck (THIRD #12, {{ team_name }} #14, Dries @ Tufa Labs #19) into one
> BC dataset, same linear scorer + `THIRD_PTCG` profile — the capacity-free alternative to the MLP
> (which lost the ladder in M11/M14). **NEGATIVE: pooling DEGRADES fidelity to the shipped teacher.**
> On a fixed held-out of THIRD's own MAIN decisions (same 1334 rows for every variant), pooled-3 scores
> 0.570 vs solo-THIRD 0.594 — paired **−0.024, 90%CI [−0.041,−0.006]**, entirely negative; the pre-
> registered escalation gate FAILS. **Mechanism (the real finding):** the three "same deck" pilots make
> materially DIFFERENT MAIN decisions — a Dries-trained clone scores 0.684 on Dries but only 0.508 on
> THIRD; team_name 0.622→0.478 — so pooling dilutes the target policy rather than reinforcing it.
> Deck-level analogue of the M8/M23 lesson: **clonability of the exact PILOT > volume of same-deck
> data.** The leak concern was verified clean (`game_id`=global episode id → `split_by_game` co-locates
> identical ids; THIRD-val games in pooled TRAIN = 0). **Nothing shipped/changed; no `.tar.gz`; no
> upload; no `src/` change** (the `THIRD_PTCG` profile already existed). New read-only tool
> `scratchpad/eval_fidelity_pooled.py` scores several weight files on one fixed held-out set. Pooling is
> closed as a fidelity lever for a specific-teacher clone.
>
> **⚠️ M21 UPDATE (2026-07-21) — read [m21_findings.md](m21_findings.md) first.**
> **The teacher search is UNBLOCKED — M15's "no new candidate available" was wrong.** Every episode
> our agent played records BOTH agents' `submissionId` + live rating, so our own match history is a
> ranked directory of opponents; joining it to the public leaderboard by `teamId` matched **289/289**,
> including ranks #11/#24/#32. New `tools/find_candidates.py` + `tools/make_batch.py` complete the
> pipeline (`find_candidates` → `download_competitors --limit` → `make_batch` → `quick_screen --batch`).
> Screen is now ~40 s for a failing candidate (field cache 15.6 s→0.01 s; progressive early-stop on
> the 0.40 gate; batch mode). **Screened 8 candidates, ranks #11–#285: four top decks FAIL the
> greedy-pilotability gate (0.24–0.39) — M6's wall on fresh evidence — and three of the four that pass
> fail clonability (0.60–0.62).** One survivor: **懒惰的金枪鱼 (rank #32, ~1052 elo)**, deck strength
> 0.736, clonability 0.646 (below the 0.75 bar). Measured it anyway with new
> `scratchpad/head_to_head.py`: **TIES v1** (+0.010, CI spans 0). Then took the designed path
> (option B): hand-authored a real **`KANGASKHAN_1052`** DeckProfile (Rapid-Fire Combo modelled at its
> GUARANTEED 200 not expected 250; {G} scarcity in `wants_fn`; opp-is-`ex` snapshot signal because
> Crustle's Ability blanks ex attackers). Fidelity only 0.646→**0.652** (M15's predicted "small win"),
> head-to-head still a tie (+0.011) — but it made the clone **shippable**, which a GENERIC_* profile
> never was (`get_profile` raises → `_build_imitation` silently degrades to greedy).
> **`build/imitation-kanga.tar.gz` built + explicitly verified to run the imitation tier**
> (`_IMITATION_READY: True`, profile resolves, dim 596). NOT uploaded — free-roll candidate, user's call.
> **Methodological correction:** `--screen-games` subsampling was justified as "can only understate
> accuracy, so a pass is safe" — **that is false** (0.686 on 60 games vs 0.646 on 461: it shrinks the
> validation set too, so it OVERstates). Only the BORDERLINE escalation band caught it. Also: the
> infamous 1.7 h featurization was machine contention, not cost — the same dataset featurized in 17 s.
> Two-sided read on the teacher-elo question: elo does NOT transfer proportionally, but a 0.65-fidelity
> clone of a 1052 player is worth about as much as a high-fidelity clone of a 661 player. The field
> gauntlet saturates ~0.90 (M11) so offline cannot settle it. **imitation-v1 (~686) remains champion.**
>
> **⚠️ M23 UPDATE (2026-07-22) — read [m23_findings.md](m23_findings.md) first.**
> The user manually filled in the `submission_id` of the **entire top-20 of the global leaderboard**
> (1093–1231 elo) — data M21/M22 could never reach (only opponents already faced). All 20 IDs
> validated 20/20 vs team names, all with ample replays. **Result: NEGATIVE.** (1) **M6's wall at the
> summit: 13/20 of the world's best decks are unpilotable by greedy** (strength 0.167–0.375),
> including #1 Majkel (0.319). (2) The 7 pilotable ones all clone to agents that **LOSE to kanga** on
> the field gauntlet (Δ −0.054 to −0.151, every CI entirely negative). (3) Cleanest finding —
> **"Where is my orbit" (#9, 1122) runs the SAME Mega Kangaskhan+Crustle deck as kanga's teacher
> (#32, 1052), 17/18 cards identical, piloted +70 elo — and STILL loses by −0.112**, because its build
> is less pilotable (0.662 vs 0.736) and clones worse (0.546 vs 0.646). This is the M8 lesson
> (clonability > teacher elo) proven on identical decks, and it **refutes the teacher-elo hypothesis
> that motivated M23**: above ~1050 elo the decks stop being clone-friendly, so more teacher elo buys
> nothing. **kanga's teacher (懒惰的金枪鱼) sits at/near the clonability-vs-strength sweet spot; the
> top-20 sweep found nothing better.** NO ship candidate; no slot spent; no `src/` change. Caveat: the
> gauntlet has mis-called the ladder 3× so a negative Δ isn't proof of ladder-worse — but no candidate
> shows a POSITIVE signal worth a scarce slot, and the motivating hypothesis failed its own control.
> **Recommendation: let kanga-new reconverge to ~860; if pushing further, hunt a clone-friendly deck
> in the ~900–1050 band (where 懒惰的金枪鱼 and ITF came from; ~270 unscreened M22 candidates remain),
> NOT a higher-elo teacher.**
> **M23 ADDENDUM — SHIPPED `build/imitation-third.tar.gz` at the user's decision (pending upload).**
> The user (correctly noting the gauntlet is unreliable — it read ITF as beating kanga, ITF is losing
> on ladder) chose to ship the one pilotable top-20 deck in our best archetype: **THIRD PTCG Club
> (#12, 1117)** = v1's exact Team Rocket Rush swarm (Rocket Rush 30×TR, `damage_650` verbatim) UPGRADED
> with **Mewtwo ex** (Erasure Ball 160→280) + **Articuno**. Hand-authored `THIRD_PTCG` profile
> (additive, dim 567; v1/kanga/ITF weights all still load). Abundant data (160 games, 5488 MAIN rows):
> MAIN fidelity **0.594** (vs greedy 0.356), learned 5 contexts, and the hand-authored profile **beat
> the generic +0.029** (0.594 vs 0.565 — biggest such gain yet; the Rocket Rush/Mewtwo prose
> corrections help). **Head-to-head n=12: THIRD(hand) TIES kanga (−0.006, CI [−0.031,+0.020])** — the
> hand-authoring closed the generic's −0.054 gap to a tie, and it's now offline-indistinguishable from
> the proven-860 kanga but with a higher-elo teacher (#12 vs #32). Bundle verified: `_IMITATION_READY
> True`, `THIRD_PTCG` dim 567, scorer ran 11/11 MAIN fixtures 0 failures; 186 tests pass. **Upload it
> to the ITF slot (keeps kanga-new reconverging); judge only after ~50 episodes vs kanga at equal age.**
>
> **⚠️ M22 UPDATE (2026-07-22) — read [m22_findings.md](m22_findings.md) first.**
> Second screening round through the M21 pipeline: 12 candidates, ranks **#71–#866**, chosen for
> rank spread (M21: top decks are disproportionately unpilotable) and for a high `seen` score.
> 4/12 fail greedy-pilotability at 0.236–0.310 (**M6's wall, four fresh data points**); the other 8
> all fail the 0.75 clonability bar (0.530–0.667). Per the user's standing instruction not to treat
> 0.75 as eliminatory, three were duelled against v1 anyway with `head_to_head.py`.
> **`ITF_Esys_Kasu` (rank #826, seen 863.8, strength 0.731, clonability 0.530 — the LOWEST fidelity
> of the round) BEATS imitation-v1 on the 120-deck field gauntlet: +0.043 90%CI [+0.017,+0.068]
> (n=6), replicated at n=12 as +0.036 [+0.015,+0.059], 38 decks better / 24 worse.** First candidate
> ever with a paired CI entirely above v1 (M21's Kangaskhan tied at +0.010; M8 v2 and M9 v3 lost).
> **NOT shippable as measured** — it used the generic profile, which never ships; a hand-authored
> DeckProfile is required and was NOT started (user's call pending). Counterexample that sharpens
> the picture: `f` has the round's best deck strength (0.764 > the M21 candidate's 0.736) and
> **loses** to v1 (CI entirely negative). **Neither screen gate predicts the head-to-head** (strength
> order f>ITF>Fakble, clonability order Fakble>f>ITF, actual ITF≫Fakble>f) → demote both gates to
> triage and spend the time on the ~20-min head-to-head, which is the only instrument that
> discriminated. Also fixed: M21's subsample-overstates-accuracy correction existed only in the doc,
> never in `quick_screen.py` (it still printed "lower bound" and "more data can only raise this");
> labels corrected, no threshold moved. ~276 of 289 mined candidates remain unscreened.
> **ITF also BEATS the current best clone** (M21's hand-authored Kangaskhan, its shippable form):
> **+0.037 90%CI [+0.019,+0.055]**, 48 decks better / 21 worse, n=12, 0 interventions. The three
> deltas are mutually consistent (ITF−v1 +0.036, kanga−v1 +0.011, ITF−kanga +0.037 vs +0.025
> expected) — an internal check that none is an artifact. `head_to_head.py` gained `--baseline`
> so any two clones can be compared. Mechanism note: ITF's deck (Mega Starmie ex + Staryu,
> 18 unique cards) has **no prose-scaling damage**, so the generic profile's damage model is
> already correct for it — unlike Team Rocket — which is a plausible reason a 0.530-fidelity clone
> plays this well, and more evidence MAIN-accuracy is the wrong proxy.
> **BUILT (2026-07-22): `build/imitation-itf.tar.gz`, verified, pending the user's upload.**
> Hand-authored `ITF_ESYS_KASU` profile added to `deck_profiles.py` (additive, dim 539; v1's 386
> and kanga's 596 re-verified as still loading). Fidelity 0.530→**0.523** (no gain — the M21
> pattern; the profile's value is shippability). **The head-to-head transfers to the shippable
> artifact**, which had to be re-measured since all earlier numbers used the *generic* profile:
> **ITF(hand) − v1 = +0.034 [+0.017,+0.049]** and **ITF(hand) − kanga = +0.069 [+0.045,+0.092]**
> (n=12, both CIs entirely positive; the kanga margin is *larger* than the generic's +0.037).
> Bundle verified past the structural check — `_IMITATION_READY: True`, profile resolves at dim
> 539, imitation scorer ran **11/11 MAIN fixtures, 0 failures** (it is not degrading to greedy).
> **Card-data reading error found & fixed:** items/tools/special-energy effect text lives in
> `CardInfo.skills[].text`, NOT `CardInfo.text` (which is `None` for them) — earlier passes read
> the wrong attribute and concluded there was no text. Reading it correctly proved **Nebula Beam
> ignores Weakness/Resistance** (so `damage_itf` returns a flat 210, skipping the W/R adjustment
> — the generic profile models this wrong) and that **Ignition Energy gives {C}{C}{C} on an
> Evolution Pokémon**, so one attach can pay Nebula Beam outright.
> **Upload caveat — the user's "re-uploads play worse" observation is a rating artifact, not a
> regression:** every upload is a NEW submission_id starting at **μ₀ = 600** with wide σ (it does
> not inherit the old rating, even with identical weights) and needs ~50+ episodes to converge,
> and **only the most recent 2 submissions are tracked for final evaluation** — so uploading ITF
> may push an older agent out of that window. Do not judge ITF against kanga's stable ~850 until
> it has ~50+ episodes. ITF's replays could NOT be topped up (63 is all that submission has), so
> the small-sample caveat on its 0.523 fidelity stands.
>
> **⚠️ M20 UPDATE (2026-07-21) — read [m20_findings.md](m20_findings.md) first.**
> Tested a NEW class of idea (not another RL/BC variation): instead of cloning strong players'
> decisions wholesale or importing rules from external guides, **mine discrete if-then patterns
> out of the replays already on disk** and audit each against hard engine facts (the way M17's one
> surviving rule, T3, was derived). New harness `scratchpad/mine_patterns.py`, 13 players spanning
> **586–1223 elo**, ~21k player-turns. **Result: NEGATIVE, K-D fired as pre-registered.** No cell
> clears |gap|≥0.25 with ≥3 players agreeing, a held-out player confirming, and a weak-pool deck
> control moving the same way. The a-priori favourite (retreat-instead-of-attack when the Active
> dies) is **not supported** — retreat is rare for everyone (0.03–0.21) and the holdout/weak
> control contradict it. Two candidates that survived a weaker gate were **instrument artifacts**,
> both caught only because the pre-registered sanity check (must rediscover M10's 0/256 lethal
> discipline) refused to pass: (1) wrong unit — MAIN is a multi-decision phase, so per-decision
> rates read known-perfect lethal discipline as 0.10–0.40; fixed by making the unit the
> player-turn (then 0.92–1.00); (2) the END-rate gap was "could not attack" (deck energy curve),
> not "chose not to attack". The ONE thing that replicates is diffuse, not discrete: strong
> players convert far fewer play opportunities into cards played (intensity 0.32–0.54 vs the
> teacher's 0.58–0.67, held-out player agrees) — real resource conservation, but a scorer-level
> stylistic bias, i.e. exactly the M11/M14 class that no offline gate can validate and that made
> real play worse twice. **Line CLOSED; Fase B/C never ran, no slot spent, champion untouched.**
> Sixth confirmation (M11/M14/M16/M17/M19/M20), and the first to show the gap is not stored in
> discrete situation-triggered decisions at all. Useful side-product: live ladder scores for the
> whole downloaded candidate field. **imitation-v1 (~686) remains champion.**
>
> **⚠️ M19 UPDATE (2026-07-20) — read [m19_findings.md](m19_findings.md) first.**
> Isolated ONE variable (the λ schedule) to test M16's own follow-up idea: a MUCH more gradual,
> CONDITIONAL anneal (descend a level only when K2/K3 clear with margin; HOLD + re-stabilize
> otherwise) vs M17's fast calendar-fixed one. **It does NOT break the plateau.** Local run (GPU
> confirmed irrelevant — the loop is native-engine/CPU bound, verified), same M16 init (v1.2) +
> critic (`v_rl.json` base-7, NOT v_rl_v2): iter30 hit **+0.027 at λ=0.10 immediately** (= M16
> iter8 ceiling, because it used the good critic from iter1), never rose as λ fell (+0.011 @0.085,
> +0.018 @0.07), and the novel HOLD mechanism FAILED to stabilize — one more iter at λ=0.07 dropped
> bc_acc 0.937→0.904 into a **K2 drift kill (iter33)**. No checkpoint cleared the +0.04 filter →
> n=300 confirm skipped. **FIFTH confirmation (M11/M14/M16/M17/M19)** that this class of refinement
> doesn't move the ~+0.027 ceiling / ~686 ladder — and the FIRST with the compute excuse removed.
> Per the user's pre-registration, this line is **CLOSED with confidence, not reopened with another
> variation.** Only untried RL lever: extend REINFORCE to the TO_HAND context (different variable,
> but low EV after five confirmations, multi-day build — flagged, not recommended without a
> decision). Code: `rl_selfplay.py iterate()` gained `--lam-levels/--lam-max-holds` (conditional
> anneal, backward compatible). **imitation-v1 (~686) remains champion.** The user's stated goal is
> purely the Kaggle ranking; the honest standing recommendation is to stop grinding marginal RL
> variations and let v1 ride unless a genuinely new class of idea appears.
>
> **⚠️ M18 UPDATE (2026-07-18) — read [m18_findings.md](m18_findings.md) first.**
> Followed M17 Fase 3's "search reopened" lead to its conclusion. Fase A (mirror sweep, n=60):
> a better self-play leaf V (`v_rl_v2`, AUC 0.836) + lower greedy_bias reached **0.600 vs v1 in
> the mirror** (Gate A pass; search cost normally ~35 s/game, well under Kaggle's ~600 s budget).
> Fase B (generalization, n=60 vs lucario/kenn/cinderace): **FAIL, K-B — pooled −0.012 vs v1.**
> Search is strongly **matchup-dependent**: beats v1 vs Bellibolt (+0.084, clears 0.5) but loses vs
> Lucario (−0.053) and Cinderace (−0.067). The leaf V (trained on TR self-play) is well-calibrated
> for TR-like tempo (mirror + Bellibolt) and mis-calibrated elsewhere; the low greedy_bias that won
> the mirror amplifies the bad out-of-distribution overrides. **Ship path CLOSED** — Fase C (bundle
> wiring + 1-slot Kaggle sandbox probe) did NOT run; no slot spent. Confirmed-but-moot feasibility
> finding: search CAN physically ship (the engine `cg/`+`libcg.so` is already a mandatory bundle
> member; `builder.py:43` `_AGENT_MODULES` would just need search modules + a leaf-V weight added).
> Remaining search lever (low EV, not recommended): a leaf V calibrated ACROSS matchups, not TR-only.
> New: `docs/m18_findings.md`; `scratchpad/search_v2_pilot.py` is now a general search-arena harness
> (`ab`/`sweepA`/`oppB`); `data/models/v_650_v2.json`. **imitation-v1 (~686) remains champion.**
>
> **⚠️ M17 UPDATE (2026-07-18) — read [m17_findings.md](m17_findings.md) first.**
> Executed option 5 (domain-expertise heuristics). The user sourced competitive TCG strategy
> guides; fact-checking every concrete number against the engine's own card data
> (`pokemon-tcg-ai-battle/EN_Card_Data.csv`) **killed most specific claims** (several were from
> Pokémon TCG *Pocket* — a different game — or cards absent here: the "Bellibolt 70→140 at 4
> energy" threshold does not exist, our Iono's Bellibolt ex is Thunderous Bolt LLLC flat 230;
> Rocky Energy is absent; the shipped TR_650 deck is Tarountula/Spidops Rocket Rush, NOT
> Koffing/Weezing). **Routing correction:** `decision/evaluator.py` does NOT ship (only the
> unshipped `SearchPolicy` uses it), and editing a shipped `DeckProfile.wants_fn/damage_fn`
> silently corrupts trained BC weights — so the work was retargeted to the **RL critic** (same 7
> features, the M16-identified lever). **Fase 1 (positive):** an offline AUC gate over the 222k
> cached self-play states cleanly isolated ONE real gap — **T3 prose-aware threat** (fixing
> `attack_damage`'s structured-only approximation that underestimates Rocket Rush/Voltaic Chain),
> +0.014/+0.017 AUC; the two weighting terms (prize-weighted threat, board prize pool) added ~0 /
> fit a confound → rejected. New critic `data/models/v_rl_v2.json` (base-7+T3, AUC 0.836).
> **Fase 2 (NEGATIVE):** RL rerun from the v1.2 init with the v2 critic + λ-anneal (0.1→0.03,
> `rl_ckpt_iter20-22`) did NOT break M16's plateau — iter20 +0.010 ≈ M16 iter2 +0.011 (sharper
> critic didn't move the RL outcome), the delta decayed as λ dropped, and iter22 hit a K2 drift
> kill. **No M17 checkpoint beats M16's iter8 (+0.027); n=300 confirm skipped (nothing to
> confirm).** The THIRD independent proof (M11, M14, now M17) that offline gains on a subtle
> component don't move the ladder objective. New/changed files:
> `scratchpad/{value_features_v2,critic_v2_gate,build_critic_v2}.py`, backward-compatible edits to
> `scratchpad/rl_selfplay.py` (payload-routed critic + λ-anneal).
> **Fase 3 (POSITIVE) — search REOPENED.** The SAME T3 fix, applied to the M10 determinized-search
> leaf V instead of the RL critic, is significant: controlled A/B (`scratchpad/search_v2_pilot.py`,
> only the leaf V differs — teacher-fit `v_650` vs `v_650_v2`), n=100 vs imitation-v1 mirror,
> search-v1 **0.340** (replicates M10's loss) → search-v2 **0.500** (+0.16, two-prop z=2.32, p≈0.02).
> A better leaf V closes M10's ENTIRE −0.16 gap to the champion — the M10 ceiling was the evaluator,
> which is improvable, so determinized search is reopened (opposite of M10's close). Caveats: ties BC,
> doesn't beat it (0.50); ~2× wall cost; single opponent/gb/mirror — broaden before real investment.
> Next to push search PAST BC: richer leaf features, self-play critic distribution, stronger rollout.
> Other untried RL lever: extend RL to the TO_HAND context. **Ladder note:** the M16 free-roll
> `imitation-v2rl` (iter8) read 561.4 vs imitation-v1.2's 635.9 at ~5h — discouraging, consistent with
> the offline↛ladder gap.
>
> **⚠️ M16 UPDATE (2026-07-17) — read [m16_findings.md](m16_findings.md) then [m12_findings.md](m12_findings.md).**
> Self-play policy-gradient RL from the v1.2 init (`scratchpad/rl_selfplay.py`) produced the
> FIRST agent to move the win objective the RIGHT way: **iter8 (`data/models/rl_ckpt_iter8.json`)
> beats BOTH v1 and v1.2 on the n=300 strong gauntlet** (+0.027 vs v1.2, +0.024 vs v1, 7/8
> opponents; held-out cinderace +0.110) — the opposite of v1.2, where more BC fidelity made play
> worse. The edge is below the +0.05 offline ship bar and (per M14) below the gauntlet's ladder-
> resolution, so it is a **ladder free-roll bet**, not a gated ship. `build/imitation-v2rl.tar.gz`
> built + verified, pending the user's upload decision. The M14 note below (offline verification
> of a SUBTLE pilot change is impossible) still holds — the ladder remains the only real judge.
> RL plateaued at +0.02-0.027 (trust-region anchor equilibrium, MAIN-only scope — see
> m16_findings.md's mechanism section); **new direction being explored: sourcing real TCG
> strategy/archetype guides to encode domain expertise into `DeckProfile.wants_fn/damage_fn`
> and `decision/evaluator.py`** (M6's wall was piloting sophistication, not deck/algorithm;
> neither user nor assistant has deep TCG expertise — likely the actual gap vs. >1000-elo
> agents on the ladder). Not yet started; see m12_findings.md open option 5.
>
> **⚠️ M14 (2026-07-13) — read [m12_findings.md](m12_findings.md).**
> The "improve the pilot / verify a candidate offline" tactic is **CLOSED**.
> imitation-v1 (linear, ~686 elo) is the proven champion and the maximum *verifiable*
> result. Two decisive findings: (1) **imitation-v1.2 (the MLP MAIN scorer) is genuinely
> WORSE on the ladder** — converged to ~597 vs v1's ~686; the +9.3-pt offline accuracy
> gain was fidelity to a 650-elo teacher and made real play worse. (2) **No offline
> clone-vs-clone gauntlet resolves the true ladder gap** — a calibration with known ground
> truth (v1 − v1.2 ≈ +0.13) reads +0.003 (a coin flip), so a candidate improvement cannot
> be verified offline; the ladder is the only judge. Sections below predate M14 where they
> still say v1.2 is "pending" — treat the ladder verdict as final. Open options are listed
> at the end of m12_findings.md.

## 1. What this project is

Kaggle competition **"PTCG AI Battle Simulation"**: submit an agent that plays
Pokémon TCG battles against other submitted agents; ranked by ladder score.
One engineer + one AI assistant, **no GPU**, competition provides a vendored
game engine (native lib + Python wrapper) under `pokemon-tcg-ai-battle/`
(read-only, ADR-0002 — never edit it). Objective: leaderboard rank is the
priority, not architectural elegance. **~3 weeks left in the competition.**

Runtime contract: `agent(obs_dict: dict) -> list[int]`. First call (`obs["select"]
is None`) returns a 60 card-ID deck list; every subsequent call returns
`minCount..maxCount` unique indices into `obs["select"]["option"]`. Resources:
2 vCPU, 12.2 GiB RAM, no GPU, submission ≤197.7 MiB. **No per-move timeout**;
~600s compute budget per agent per episode. Reward is win/loss only (no
margin). Full detail: [competition_analysis.md](competition_analysis.md).

## 2. Repository map

```
src/ptcg_ai/
  agent/          PTCGAgent (Kaggle-shaped callable) + Deck loading/validation
  observation/    raw-dict -> ParsedObservation (tolerant parser, ADR-0005)
  cards/          CardDatabase (static card knowledge: HP/attacks/abilities/costs)
  state/          GameState — perspective-normalized derived state + KO/threat math
  decision/       Policy implementations, one per ladder rung: greedy (3), rule_based (4)
  planning/       rung-5 search plumbing (built, never shipped — see §5)
  imitation/      rung-6: behavior cloning of real ladder players — CURRENT
                  BEST AGENT. See §4. Only imitation/{deck_profiles,features,
                  policy}.py are shipped (pure stdlib); kaggle_replay.py,
                  dataset.py, train.py are dev-only (numpy, dev dep group).
  submission/     Kaggle bundle builder + entrypoint + validator (ADR-0001/0014)
  environment/    vendored-SDK loader + battle adapter/runner (ADR-0002)
  config/         frozen-dataclass config + YAML profiles (ADR-0003)
  bench/, analytics/, replay/, debug/, evaluation/, cli.py   supporting tooling
tests/            unit/ (SDK-free) + integration/ (@pytest.mark.sdk, auto-skip
                  if native engine missing) + fixtures/ (captured observations)
docs/             see §12 / docs/index.md for the full map
decks/            extracted real-player decks + experimental lists (see §9)
data/             imitation datasets (data/imitation/*.jsonl.gz) + trained
                  weights (data/models/bc_*.json) — versioned, NOT gitignored
                  for the models (weights are load-bearing artifacts)
build/            submission tarballs (gitignored; rebuild with scripts in §9)
scratchpad/       research scripts (arenas, extractors, the field gauntlet) —
                  NOT under the same review discipline as src/; verify a
                  script still matches its docstring before trusting it
Logs/             real Kaggle replay JSONs, versioned: our own submissions'
                  games (Submission {1,2,3,greedy v5,imitation v1,imitation v2})
                  + Higher ranking logs/{650 elo, 800 elo, Vibechu, Majkel}
pokemon-tcg-ai-battle/   vendored competition SDK — READ-ONLY (ADR-0002)
```

## 3. Architecture in one paragraph

Every decision — deck submission, mulligan, setup, main phase, mid-effect
targets — flows through one interface (ADR-0004):
`Policy.choose(DecisionContext) -> list[int]`. Shared pipeline (ADR-0013):
`ObservationParser -> DecisionContext{raw, observation, cards} ->
GameState.build -> per-context Handler -> Scorer -> SafePolicy`. `SafePolicy`
wraps every submitted policy: validates returned indices, catches exceptions,
degrades to uniform-random, logs interventions (should be 0 in production).
Canonical detail: [decision_system.md](decision_system.md).

### The ladder rungs

| Rung | Policy | Status |
|---|---|---|
| 1–2 | `random`, `safe-random` | done — floor |
| 3 | `greedy` | done, SHIPPED once (greedy-v5, score ≈497–536) — **superseded, see §4** |
| 4 | `rule-based` | built, tested, never shipped — neutral/negative vs greedy |
| 5 | `search` | built, tested, never shipped — underperformed greedy (0.44) |
| **6** | **`imitation` (behavior cloning)** | **SHIPPED — current best agent, see §4** |

## 4. Current submission state — THE critical section

**Best live agent = imitation-v1**: behavior clone of a real ~650-elo ladder
player (`greengreenpurple`), piloting their exact Team Rocket swarm deck
(`decks/greengreenpurple.csv`). **Confirmed live Kaggle score ≈688** (started
at ~657 with fewer games, rose as more accumulated — ratings need ~50+ games
to stabilize, don't judge a submission's true strength before that). Built via
`scratchpad/build_imitation.py decks/greengreenpurple.csv
data/models/bc_650_v1.json imitation-v1`. Weights: `data/models/bc_650_v1.json`
(profile `TR_650`, offline MAIN-context accuracy 86.2% vs greedy's 35.6%).

**A second agent (imitation-v2) was shipped and then found to UNDERPERFORM —
important cautionary tale, read §5.** It is not live; v1 is the standing best.

**M9 (imitation-v3) also did NOT beat v1 — but was correctly caught by the gate
and never shipped.** A clone of the ~940-elo Bellibolt pilot `kenN2439` cloned
well (MAIN 0.770, pilot-lift +0.30) yet lost the field gauntlet badly (macro WR
0.662 vs v1's 0.876; paired delta −0.214). Key finding: **v1's edge is an
exceptional clone-lift (+0.67), not deck strength** — Team Rocket is the weakest
deck under greedy (0.210) but clones to 0.876. See [m9_findings.md](m9_findings.md).

**M10 (exceed v1, don't re-clone) also did NOT beat v1.** Diagnosed v1's 86 real
ladder games: **perfect lethal discipline (0/256 misses)** and **no cheap tactical
leak** (alpha win-weighting was null too). Built BC-guided determinized search with
a learned value function (`v_650.json`, AUC 0.796) — it plays WORSE than v1 (arena
0.208 greedy-rollout → 0.320 BC-rollout, both lose), reconfirming M6's lesson that
cheap search degrades already-strong play. Nothing shipped. See
[m10_findings.md](m10_findings.md).

**M11 (imitation-v1.2): a neural-net MAIN scorer — big offline gain, flat gauntlet,
SHIPPED as a free-roll.** A 1-hidden-layer MLP replacing v1's linear MAIN scorer
lifts held-out MAIN accuracy +9.3 pts (0.858→0.952, clean 3-way split) — the first
real offline win in 4 milestones. But the field gauntlet is a dead tie (paired delta
−0.001) because it saturates at ~90% vs the greedy field and can't resolve a subtle
gain. Strict gate narrowly failed, but it's a TIE not a regression; shipped by user
choice as a low-downside free-roll (Kaggle keeps best score) to let the REAL ladder
measure it. `build/imitation-v1.2.tar.gz` verified, **pending upload**. The MLP
inference path (`policy.py` payload v2) is now reusable for any context. See
[m11_findings.md](m11_findings.md).

**Default `ptcg build-submission` still produces greedy-v5** (the
`_AGENT_MODULES` allowlist ships `imitation/{deck_profiles,features,policy}.py`
unconditionally — they're inert without a bundled `bc_weights.json` — but the
default build passes no `weights_path`, so `_IMITATION_READY` is False and the
entrypoint falls back to greedy). To ship an imitation agent you must call
`build_submission(config, weights_path=...)` directly — see
`scratchpad/build_imitation.py` for the pattern (deck/weights/out all argv).

**Entrypoint tiering** (`submission/_entrypoint.py`): imitation → greedy →
inline safe-random, checked per decision. Never raises.

## 5. What was tried, condensed (full detail in the linked findings docs)

| Milestone | What | Outcome |
|---|---|---|
| M1–M4 | safe-random → greedy rungs, Trainer whitelist tuning | greedy-v5 shipped, ladder ≈497–536 |
| M5 | Rung-4 `rule-based` + alternative decks | neutral/negative vs greedy — [m6→ still not the answer] |
| M6 | Rung-5 determinized search; extracted #1/#2 players' actual decks | search 0.44 vs greedy (worse); **top players' own decks, in OUR hands, score 19%** vs their 62% — proved the wall was piloting skill, not deck quality. [m6_findings.md](m6_findings.md) |
| **M7** | **Behavior cloning (BC) of the 650-elo player** — the lever M6 identified as unspent | **Beat greedy-v5 0.717 in arena; SHIPPED as imitation-v1; confirmed on ladder (≈688).** [m7_findings.md](m7_findings.md) |
| **M8** | Re-targeted BC at an 800-elo Mega Lucario pilot (imitation-v2) | **Beat imitation-v1 0.660 in a head-to-head arena → shipped → UNDERPERFORMED on the real ladder (≈610 vs v1's ≈688).** [m8_findings.md](m8_findings.md) |
| **M8.1** | Diagnosed the M8 regression + built a corrected promotion gate | **Root cause: the M8 gate measured ONE matchup (v2 vs v1); the ladder is a diverse field — non-transitivity.** Built `scratchpad/field_gauntlet.py` (97 real opponent decks from our own ladder replays) as the new gate. Attempted a fix (imitation-v2.1: fixed a real bug + added hand-composition features) — **still lost to v1 on the field gauntlet (macro WR 0.832 vs v1's 0.883; paired delta −0.050, 90% CI entirely negative) despite "winning" head-to-head 0.625.** NOT shipped. [m8_findings.md](m8_findings.md) (M8.1 addendum) + memory. |
| **M9** | Built the `tools/` replay downloader; screened 5 new ~850–940-elo teachers; cloned the best (imitation-v3, `kenN2439` Bellibolt) | **v3 cloned well (MAIN 0.770, pilot-lift +0.30) but LOST the field gauntlet (macro 0.662 vs v1 0.876; paired delta −0.214). All 4 screened candidates exhausted; two backups ruled out by cheap probes (Cinderace clones at 0.639, worst yet).** Finding: **v1's edge is clone-lift (+0.67), not deck strength.** NOT shipped — gate worked. [m9_findings.md](m9_findings.md) |
| **M10** | First attempt to EXCEED v1 (not re-clone): diagnosed v1's losses, tried cheap patches, built BC-guided search + learned V | **No cheap leak (lethal 0/256 perfect; alpha null). BC-guided search with a learned V (AUC 0.796) plays WORSE than v1 (arena 0.208→0.320, both lose) — reconfirms M6: cheap search degrades strong play.** v1's losses are ~structural (18/41 are 0-2-prize blowouts vs a diverse field). NOT shipped. [m10_findings.md](m10_findings.md) |
| **M11** | imitation-v1.2: replace v1's linear MAIN scorer with a 1-hidden-layer MLP (payload v2, stdlib inference) | **+9.3 pts held-out MAIN accuracy (0.858→0.952, clean split) — first real offline win in 4 milestones — but field gauntlet is a dead TIE (delta −0.001) because it saturates ~90% vs the greedy field.** SHIPPED as a low-risk free-roll (tie not regression; real ladder is the only test with resolution). [m11_findings.md](m11_findings.md) |
| **M12** | Diagnosed v1.2 (46→76 ladder games); built leave-archetype-out OOD CV + a residual "safety-net" scorer | **v1.2 behaves as built: 100% live-parity, perfect lethal, benign OOD divergence (1.16×); pure MLP is OOD-best (0.9524 vs linear 0.8732).** Residual architecture FAILED its gate (−0.0104 vs pure MLP). At the time read as "v1.2 healthy" — later shown to be instrument failure. [m12_findings.md](m12_findings.md) |
| **M13** | Built the strong-opponent gauntlet (clone foes, not greedy); prepped a TO_HAND-MLP extension (Phase B) | Phase A (4 foes, n=300): pooled v1.2−v1 = **+0.019** (inconclusive), head-to-head +0.090. Phase B (`train_mlp_v4.py`, `cv_archetype_tohand.py`) built + compile-checked but **PAUSED, never shipped** — premise undercut by M14. [m12_findings.md](m12_findings.md) |
| **M14** | Read v1.2's converged 76-game ladder record via the Kaggle API; ran a falsifiable calibration of the strong gauntlet | **DECISIVE. v1.2 is genuinely WORSE (converged ~597 vs v1 ~686; WR 0.492 vs 0.632 in the 500–700 band) — the MLP hurt real play. And the 8-clone gauntlet reads the known +0.13 gap as −0.003 (coin flip) = FALSIFIED.** Offline verification of a pilot gain is impossible; ladder is the only judge. Tactic CLOSED. [m12_findings.md](m12_findings.md) |

**Net result: imitation-v1 (linear, ~686) is the proven champion AND the maximum
*verifiable* result — the "improve the pilot / verify offline" tactic is CLOSED (M14).**
greedy-v5, rule-based, search, imitation-v2/v3, M10's search, and now **imitation-v1.2
(MLP scorer — ladder-confirmed worse)** are superseded/rejected. The hard lesson across
M11–M14: **no offline instrument we can build resolves a subtle pilot difference** (the
greedy gauntlet saturates; the clone gauntlet averages correlated matchups to zero;
higher BC fidelity to a 650 teacher makes play *worse*, not better). Any future candidate
can only be judged on the real ladder. Remaining open options: see the end of
[m12_findings.md](m12_findings.md).

## 6. The core strategic findings (read both — they built on each other)

**M6 finding (why heuristics/search hit a wall):** proven on **identical
decks** — the #1 ladder player pilots their own engine deck to 62%; our best
heuristic/search pilots pilot the *same extracted decklist* to 19%. The gap is
piloting sophistication (energy sequencing, ability timing), not deck choice.
→ led to M7's behavior-cloning approach, which worked.

**M8/M8.1 finding (why a stronger teacher isn't automatically a better clone):**
BC quality is capped by how *faithfully* the deck's decisions can be cloned,
not by the teacher's elo. The 650 swarm deck (simple: 1-step evolution,
structured attack damage, decisions concentrated in MAIN/TO_HAND) cloned to
86% MAIN-accuracy and transferred its ~650 elo almost exactly to the ladder.
The 800 Lucario deck (complex: ability-engine timing around hand energy,
opponent-targeting "gust" trainers, a conditional 0-damage attack) only cloned
to ~71% MAIN-accuracy — and that lower fidelity meant it piloted *worse* than
the more-faithful, lower-elo clone, despite the teacher being objectively
stronger. **Clonability > teacher elo.** A "clonability screen" (see §7) now
gates any future teacher choice.

**Process finding (just as important):** M8 shipped v2 on a promotion gate
that only checked "does the new agent beat the CURRENT champion head-to-head".
That gate is provably insufficient — v2 beat v1 0.660 head-to-head yet
underperformed it on the real, diverse ladder field. **Any future candidate
must clear the field gauntlet (§9), not just a head-to-head arena, before
shipping.**

## 7. Recommended next step (the user's chosen direction, not yet started)

**Pivot to a new teacher that passes a clonability screen BEFORE any build
work starts.** Criteria (derived from the 650-vs-800 comparison in §6):

1. Elo range ~700–900, stable for 15+ days (avoid noisy/recent climbers).
2. Runs **one fixed deck** across their games (required for a clean dataset).
3. Deck-complexity screen — reject candidates whose deck has:
   - more than 1 evolution line (Basic → Stage 1 only, no Stage 2),
   - "prose" attack damage (scales with board state, or is conditional/
     ignores weakness — like Rocket Rush or Cosmic Beam were),
   - many opponent-targeting "gust" trainers (Boss's Orders-style — these
     drove a large chunk of the M8 accuracy gap),
   - a decision distribution spread thin across many contexts instead of
     concentrated in MAIN/TO_HAND (a proxy for "simple to imitate").
4. Prefer a simple aggro/swarm archetype (like the 650 deck) over an
   engine/control deck (like the 800 Lucario deck), even at some elo cost.

**This needs the user to source 2–3 candidate players' replay logs** (same
format as `Logs/Higher ranking logs/650 elo/` — a folder of that player's
Kaggle replay JSONs). Once logs exist, the pipeline to evaluate + clone a new
teacher is now fully mechanical (loader → dataset → profile → train → arena →
**field gauntlet**, not just head-to-head) — see §9 for the reusable scripts.

**Longer-horizon stretch (unstarted, higher risk/reward):** BC's ceiling is
the teacher's own skill — pure imitation cannot exceed it. To actually surpass
a strong player (not just match one), the sanctioned next lever is reusing the
650-clone's learned logits as move *priors*, and/or fitting a learned value
function on the same win/loss-labeled dataset, inside the already-built rung-5
determinized search stack (`planning/`, `decision/search.py`,
`decision/evaluator.py`) — replacing M6's hand-tuned linear V (which is what
made search lose 0.44) with a data-driven one. Not started; discuss with the
user before committing days to it.

## 8. User context / working agreements

- User is **not a TCG domain expert** — relies on the assistant for
  game-strategy reasoning. Be direct about uncertainty.
- Prefers **stop-and-approve checkpoints** at milestone boundaries.
- Explicit priority: **ladder rank**, not code elegance. Every milestone
  should be justified by a measured score improvement — and now, explicitly,
  by a **field-gauntlet-validated** improvement, not just a head-to-head arena
  win (the M8 lesson).
- Reacted calmly and constructively to the M8 regression — wants the failure
  analyzed and fixed/learned-from, not hidden or minimized. Comfortable with
  "no ship" as a valid, honest outcome of an evaluation (M8.1 ended in no-ship
  and that was accepted as correct behavior, not a failure to deliver).
- Persistent memory entries exist at the assistant's memory store (if the
  running assistant has file-based memory) — check for
  `pokemon-tcg-m7-imitation-breakthrough.md` and `pokemon-tcg-m8-ladder-climb.md`
  (the latter has the full M8/M8.1 narrative including the ladder-verified
  outcome); `pokemon-tcg-pilot-wall.md` is the superseded pre-M7 state, kept
  for history only.

## 9. Reusable assets inventory

### Imitation-learning pipeline (`src/ptcg_ai/imitation/`) — the core asset

- `kaggle_replay.py` — player-agnostic Kaggle replay loader (`iter_player_decisions`,
  `load_replay_dir`): handles the verified **+1 action lag**
  (`steps[i].observation` answered by `steps[i+1].action`) and keeps only
  `ACTIVE` cells with a live `select`. Reusable for ANY future teacher.
- `dataset.py` — builds a gzip-JSONL dataset (`build_decision_dataset`) +
  leak-free by-game train/val split (`split_by_game`). CLI: `python -m
  ptcg_ai.imitation.dataset <log_dir> <player> <out.jsonl.gz>`.
- `deck_profiles.py` (SHIPPED, stdlib) — `DeckProfile` dataclass bundling a
  cloned deck's card/attack vocab + hand-computed damage corrections + state
  extractors. **Adding a new teacher = one new `DeckProfile` instance, not new
  code**, as long as the deck passes the §7 clonability screen. Current
  profiles: `TR_650` (dim 386, the shipped one), `LUCARIO_800` (dim 467),
  `LUCARIO_800_V2` (dim 633, unshipped — hand-composition features).
- `features.py` (SHIPPED, stdlib) — the per-option featurizer, parametric over
  a profile; identical code path offline and live (no train/serve skew).
- `train.py` (dev-only, numpy) — per-context linear conditional-logit trainer;
  data-driven learned-context selection (a context is learned only if it has
  enough rows AND greedy doesn't already predict it well AND its options
  aren't degenerate-by-identity; degenerate CARD/ENERGY selects get a
  zeros-vector "take-k" fallback, but COUNT/YES_NO selects route to greedy
  instead — a real M8.1 bug fix, do not regress it). CLI: `python -m
  ptcg_ai.imitation.train <dataset> <deck.csv> <out.json> <profile_name>
  [--seed N] [--alpha A]`.
- `policy.py` (SHIPPED, stdlib) — `ImitationPolicy(GreedyPolicy)`: scores
  learned contexts, defers to greedy elsewhere. Zero exceptions across ~2000+
  real ladder games analyzed to date.

### Evaluation scripts (`scratchpad/`)

- `arena_m7.py`, `arena_m8.py` — head-to-head arena harnesses (pattern:
  `run(a_kind, deck_a, b_kind, deck_b, label, n, weights...)`, Wilson CI,
  swapped sides). **Necessary but NOT sufficient** — see next entry.
- **`field_gauntlet.py` — the corrected promotion gate, use this before any
  ship decision.** Extracts every distinct legal opponent deck seen across our
  own ladder replay folders (currently 97 decks) + reference/meta decks;
  evaluates a candidate (piloting its own deck) against each under greedy, and
  reports a **paired per-deck delta vs the current champion** with a bootstrap
  CI. This is what caught the M8 regression that the head-to-head arena
  missed. `python scratchpad/field_gauntlet.py build` (extract + summarize)
  or `run <n>` (full evaluation of the `CANDIDATES` list defined in the file —
  update that list to add a new challenger).
- `build_imitation.py` — build + validate an imitation submission tarball:
  `python scratchpad/build_imitation.py <deck.csv> <weights.json> <out_name>`.
- `extract_top_decks.py` — reconstructs a player's exact 60-card deck from the
  first action of their replay JSONs (argv: `<log_dir> <player_name>
  <out.csv>`; deck submission is always fully recoverable, no inference).

### `decks/*.csv` (validated-legal 60-card lists; card IDs only)

| File | What it is |
|---|---|
| `greengreenpurple.csv` | **the shipped imitation-v1 deck** (650-elo Team Rocket swarm) |
| `lucario800.csv` | the unshipped imitation-v2 deck (800-elo Mega Lucario ex) |
| `meta_lucario.csv`, `meta_dragapult.csv` | real ladder meta decks (opponents/gauntlet reference) |
| `vibechu.csv`, `majkel.csv` | #1/#2 players' actual decks (M6 extraction; too complex to pilot with heuristics — pre-imitation-learning finding) |
| others (`psychic_*.csv`, `slaking_wall.csv`, `greedy_water_v2.csv`, ...) | pre-M7 heuristic-pilot deck experiments, all superseded |

### Real replay data (`Logs/`, versioned in git)

- `Submission {1,2,3,greedy v5,imitation v1,imitation v2}/` — our own ladder
  replays across every submission generation (imitation v2's replays are what
  M8.1's diagnosis was built on).
- `Higher ranking logs/{650 elo, 800 elo, Vibechu, Majkel}/` — real players'
  logs. 650 elo (325 games) and 800 elo (434 games) are the two teachers
  cloned so far; Vibechu/Majkel (M6) were extracted but never cloned (pre-BC).

## 10. Tooling quick-reference

- Package/venv manager: **uv**. Run everything as `uv run <cmd>` from the
  repo root. Dev-only deps (numpy, for the imitation trainer) need `uv run
  --group dev ...`.
- Tests: `uv run pytest -q` (179 passed / 1 skipped at snapshot; the skip is
  a flaky `@pytest.mark.sdk` search-warmup test, pre-existing, not a
  regression signal). `uv run --group dev pytest -q` if touching `train.py`.
- CLI (`ptcg_ai.cli`, installed as `ptcg`): `ptcg build-submission` (still
  greedy-v5 by default — see §4), `ptcg validate-submission`, `ptcg
  capture-fixtures`, `ptcg bench {parser,battle,search,all}`, `ptcg battle
  --trace`.
- SDK-gated tests/scripts need the vendored native engine loadable — it is,
  at snapshot.

## 11. Promotion gate (mandatory reading before shipping anything new)

> **M14 caveat (2026-07-13): for a SUBTLE pilot change (a scorer tweak on the same
> deck), NO offline gate below is sufficient.** M14 proved this by calibration: with
> known ground truth (v1 − v1.2 ≈ +0.13 on the ladder) the strong clone gauntlet — even
> at 8 diverse foes, n=300 — reads −0.003 (a coin flip). The greedy field gauntlet
> saturates; the clone gauntlet averages correlated matchups to zero. These gates only
> resolve LARGE gaps (e.g. v1 ≫ v3). A subtle candidate can only be judged on the real
> ladder. See [m12_findings.md](m12_findings.md). Do not ship a subtle candidate on an
> offline pass alone — that is exactly how v1.2 shipped and then underperformed.

**Two-stage gate, both required — this section was rewritten after the M8
lesson, do not revert to the single-stage version:**

1. **Head-to-head arena** vs the current champion (imitation-v1): swapped
   sides, 95% Wilson lower bound clearing 0.5. Necessary but **NOT sufficient**
   — M8 passed this stage (0.660) and still shipped a regression.
2. **Field gauntlet** (`scratchpad/field_gauntlet.py run <n>`): paired
   per-deck delta vs the champion across the ~97-deck field, bootstrap 90% CI.
   Ship only if the CI lower bound clears roughly break-even (the M8.1 bar
   used mean ≥ 0 AND CI-lo > −0.02; recalibrate if the field composition
   changes materially). **This is the stage that actually predicted the real
   ladder outcome for both v2 (field said worse, ladder confirmed) and v1
   (field said better, ladder confirmed).**

Offline accuracy (BC vs greedy-as-predictor vs random, held out by game) is a
cheap upstream filter — don't spend arena/gauntlet time on a candidate whose
weighted learned-context accuracy doesn't clearly beat greedy — but it is not
a promotion gate by itself either.

## 12. Documentation index

See [index.md](index.md) for the full table (competition rules, SDK
mechanics, architecture, benchmarking, decision system, etc.) and
[decisions/README.md](decisions/README.md) for the ADR list. Milestone
narratives, in order, each written to prevent re-running exhausted approaches:
[m6_findings.md](m6_findings.md) (heuristics/search wall + the piloting-skill
diagnosis that motivated imitation learning), [m7_findings.md](m7_findings.md)
(the first successful BC agent, imitation-v1), [m8_findings.md](m8_findings.md)
(re-targeting BC at a stronger teacher — includes both the M8 ship and the
M8.1 diagnosis/no-ship outcome). Read m8_findings.md end to end before
proposing another teacher pivot — it has the exact numbers behind §6/§7 above.
