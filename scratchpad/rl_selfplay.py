"""M16 — self-play policy-gradient RL from the BC init (Fase C).

The BC ceiling is the teacher's skill. This optimizes the REAL objective (winning)
instead of imitation: the agent samples from its own MAIN policy, plays thousands of
games vs strong clone opponents, and REINFORCE nudges up the moves from won games /
down the moves from lost games, anchored to the BC weights so cheap optimization can't
destroy strong play (the M10 lesson) and evaluated on win rate vs fixed opponents
(within the instrument's resolution for LARGE gaps, the M14 lesson).

Only the MAIN context (single-pick, densest) is fine-tuned; everything else stays BC.
Base model = v1.2's 3-seed MLP ensemble (bc_650_v2.json). Checkpoints are full v2
payloads, directly playable by ImitationPolicy / the gauntlets.

Subcommands:
  smoke                     C0 correctness gates (sampler parity, gradient check, collect)
  collect  <ckpt> <out> [n] one batch of self-play trajectories
  update   <ckpt> <traj> <out> [--lambda L] [--tau T] [--lr R] [--epochs E]
  eval     <ckpt> [n]       mini strong-gauntlet vs the opponent set
  iterate  [--iters K ...]  the full C2 loop with kill criteria

Run:  uv run --group dev python scratchpad/rl_selfplay.py smoke
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.runner import BattleRunner
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation import deck_profiles as dp
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.imitation.policy import ImitationPolicy, _score
from ptcg_ai.imitation.train import Decision, _pack, _seg_softmax
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.state.game_state import GameState

sys.path.insert(0, "scratchpad")
from train_mlp_main import _Adam, _mlp_scores  # noqa: E402
from train_value import features as value_features  # noqa: E402
import value_features_v2 as vf2  # noqa: E402  (M17: base-7 + T3 prose-aware critic)
import arena_m8  # noqa: E402

# Base-7 feature names, for critic payloads that predate the "features" key (v_650/v_rl).
_BASE7 = ("prize", "threat", "reserve", "survival", "energy", "hand", "deck_out")

# baselines from scratchpad/results/m14_calibration.txt (n=300), for delta reference.
V12_BASELINE = {"lucario": 0.315, "m093jp": 0.590, "budew": 0.680, "eduardo": 0.700,
                "tr_mirror": 0.547, "legend": 0.800, "kenn": 0.853, "cinderace": 0.290}
V1_BASELINE = {"lucario": 0.320, "m093jp": 0.640, "budew": 0.660, "eduardo": 0.683,
               "tr_mirror": 0.500, "legend": 0.843, "kenn": 0.783, "cinderace": 0.367}

MAIN_CTX = int(SelectContextKind.MAIN)

#: M38's measured `probe_agree` per iteration (data/rl_alakazam/loop_manifest.json), the
#: unclipped drift curve M47 is trying to beat. iter2's manifest entry was lost; the
#: trajectory file is on disk, the number is not.
_M38_DRIFT = {1: 0.9660, 3: 0.9367, 4: 0.9253, 5: 0.9113, 6: 0.9002,
              7: 0.9160, 8: 0.9207, 9: 0.9025, 10: 0.8835}

# --------------------------------------------------------------------------------------
# SETUPS. M16-M19 ran the `tr650` setup below; it stays the DEFAULT so those runs remain
# reproducible byte-for-byte. M38 adds `alakazam`, which changes exactly three things and
# nothing else -- the algorithm, λ, batch, epochs and all six kill criteria are untouched,
# so the experiment stays single-variable (the user's standing rule):
#
#   1. THE BASE. M16 fine-tuned `bc_650_v2` (~686 on the ladder). We now start from the
#      Alakazam clone at ~923. That is a ~250-elo better starting policy, and it is the
#      single biggest difference between this attempt and the five that failed.
#   2. THE OPPONENT MIX. The tr650 mix is the July ~650-bracket meta and contains NO
#      Grimmsnarl -- which is 30.8% of the field `imitation-fetch` actually faces. The
#      alakazam mix is rebuilt from that measured composition (submission 55145833).
#      `bc_luca_full` is the load-bearing piece: M35 rejected it as a CANDIDATE (data
#      ceiling, 539 episodes) but as an OPPONENT it is exactly what was missing, a
#      Grimmsnarl piloted properly instead of by greedy. The saturated gauntlet gives us
#      9/9 against greedy-piloted Grimmsnarl while the real ladder gives 43.9%; that gap
#      IS the pilot.
#   3. THE CRITIC. `v_650` was fit on TR-650 self-play. M18 already documented that this
#      value function is mis-calibrated out of distribution, so the Alakazam setup must
#      refit it (`refresh_critic`) before the advantages mean anything.
# --------------------------------------------------------------------------------------

SETUPS = {
    "tr650": {
        "cand_deck": "decks/greengreenpurple.csv",
        "base_ckpt": "data/models/bc_650_v2.json",
        "critic": "data/models/v_650.json",
        "dataset": "data/imitation/greengreenpurple.jsonl.gz",
        "profile": "TR_650",
        "opponents": {
            "lucario":  ("decks/lucario800.csv",           "data/models/bc_800_v2.json",     0.25, None),
            "m093jp":   ("decks/m093jp.csv",               "data/models/m093jp_screen.json", 0.15, "m093jp"),
            "budew":    ("decks/budew.csv",                "data/models/budew_screen.json",  0.15, "budew"),
            "eduardo":  ("decks/eduardorochadeandrade.csv", "data/models/eduardorochadeandrade_screen.json", 0.15, "eduardorochadeandrade"),
            "tr_mirror":("decks/greengreenpurple.csv",      "data/models/bc_650_v1.json",    0.15, None),
            "legend":   ("decks/legendbrothers.csv",        "data/models/legendbrothers_screen.json", 0.075, "legendbrothers"),
            "kenn":     ("decks/kenn2439.csv",              "data/models/bc_940_v1.json",    0.075, None),
        },
        "heldout": ("cinderace", "decks/yoshiki.csv", "data/models/bc_cinderace_probe.json"),
    },
    # M38. Weights follow the measured archetype shares of imitation-fetch's real field:
    # Grimmsnarl 30.8%, Alakazam mirror 24.8%, Kangaskhan 6.0%. Archaludon (13.5%) is the
    # HELD-OUT archetype -- we win it 83% on the ladder, so it is the cleanest read on
    # "did this overfit to the training mix?" without wasting training games on a matchup
    # that is already solved.
    "alakazam": {
        "cand_deck": "decks/yushinito.csv",
        "base_ckpt": "data/models/bc_alakazam_fetch.json",
        "critic": "data/models/v_alakazam.json",
        "dataset": "data/imitation/yushinito_full.jsonl.gz",
        "profile": "ALAKAZAM",
        # M38 REWEIGHT (2026-08-04): the original 0.40/0.35/0.25 split trained a model that
        # improved vs grimmsnarl (+1.7pt) and mirror (+10.0pt) while COLLAPSING vs
        # kangaskhan (init 0.275 -> stable ~0.18-0.19 across iters 4/6/8, mostly
        # deck-out losses -- 58/97). `rl_update` pools every decision from every opponent
        # into one gradient with no per-opponent reweighting beyond game COUNT, so the
        # kangaskhan gradient was simply outvoted. Raising its share from 0.25 to 0.40
        # gives it proportionally more decisions -> proportionally more gradient
        # influence, which is the direct lever for a matchup-specific regression (as
        # opposed to more iterations, which repeats the same imbalance).
        "opponents": {
            "grimmsnarl": ("decks/luca.csv",       "data/models/bc_luca_full.json",       0.30, None),
            "mirror":     ("decks/yushinito.csv",  "data/models/bc_alakazam_fetch.json",  0.30, None),
            "kangaskhan": ("decks/kanga_nightstretcher.csv", "data/models/bc_kangaskhan_1052.json", 0.40, None),
        },
        "heldout": ("archaludon", None, None),   # resolved from the fresh field at eval time
    },
    # M47. Same algorithm as `alakazam`; three inputs corrected, each for a reason that
    # was already MEASURED and written down in this repo but never acted on:
    #
    #   1. THE BASE is the SHIPPED champion, not `bc_alakazam_fetch`. Their MAIN blocks
    #      are byte-identical (sha 20bf903018d963d6, verified), so this does NOT change
    #      what is being optimised -- but `fetch` lacks the M42/M43 heads (ACTIVATE,
    #      SETUP_BENCH + count head, SWITCH, TO_HAND-MLP) worth ~1.26 corrected decisions
    #      per game, so M38's self-play generated the state distribution of a WEAKER
    #      agent than the one we deploy. `_write_ckpt` shallow-copies the base payload
    #      and replaces only contexts.MAIN, so `profile_overrides` and `count_heads`
    #      survive and every checkpoint is directly shippable.
    #   2. THE OPPONENT WEIGHTS revert to M38's 0.40/0.35/0.25. The 0.30/0.30/0.40 above
    #      is M39's reweight, which M39 itself measured as CONFIDENTLY WORSE (0.438
    #      [0.382-0.494] vs the frozen init at n=300) and left in the file.
    #   3. THE HELD-OUT ARCHETYPE is Mega Lucario, which is 11.5% of the real field
    #      (M44) and which we HAVE a clone of. `alakazam` named archaludon and then had
    #      to resolve it "at eval time" from a fresh field because no clone exists --
    #      i.e. it had no working overfit check at all.
    "alakazam_final": {
        "cand_deck": "decks/yushinito.csv",
        "base_ckpt": "data/models/bc_alakazam_final.json",
        "critic": "data/models/v_alakazam.json",     # override with --critic after M47 F0
        "dataset": "data/imitation/yushinito_full.jsonl.gz",
        "profile": "ALAKAZAM",
        "opponents": {
            "grimmsnarl": ("decks/luca.csv",      "data/models/bc_luca_full.json",      0.40, None),
            "mirror":     ("decks/yushinito.csv", "data/models/bc_alakazam_final.json", 0.35, None),
            "kangaskhan": ("decks/kanga_nightstretcher.csv", "data/models/bc_kangaskhan_1052.json", 0.25, None),
        },
        "heldout": ("lucario", "decks/lucario800.csv", "data/models/bc_800_v2.json"),
    },
}

SETUP = "tr650"
CAND_DECK = SETUPS[SETUP]["cand_deck"]
BASE_CKPT = SETUPS[SETUP]["base_ckpt"]
V650 = SETUPS[SETUP]["critic"]
DATASET = SETUPS[SETUP]["dataset"]
PROFILE = get_profile(SETUPS[SETUP]["profile"])
OPPONENTS = SETUPS[SETUP]["opponents"]
HELDOUT = SETUPS[SETUP]["heldout"]


def configure(setup: str) -> None:
    """Rebind the module globals to one of SETUPS. Must run before anything else.

    The globals are read all over this file (collect, _featurize_traj, evaluate, iterate),
    so a config object would mean touching every call site and re-verifying the M16 path.
    Rebinding keeps the tr650 code path byte-identical.
    """
    global SETUP, CAND_DECK, BASE_CKPT, V650, DATASET, PROFILE, OPPONENTS, HELDOUT
    if setup not in SETUPS:
        raise SystemExit(f"unknown setup {setup!r}; choose from {sorted(SETUPS)}")
    cfg = SETUPS[setup]
    SETUP = setup
    CAND_DECK = cfg["cand_deck"]
    BASE_CKPT = cfg["base_ckpt"]
    V650 = cfg["critic"]
    DATASET = cfg["dataset"]
    PROFILE = get_profile(cfg["profile"])
    OPPONENTS = cfg["opponents"]
    HELDOUT = cfg["heldout"]
    total = sum(w for _, _, w, _ in OPPONENTS.values())
    if abs(total - 1.0) > 0.02:
        raise SystemExit(f"setup {setup!r} opponent weights sum to {total:.3f}, expected ~1.0")
    for name, (deck, wts, _, _) in OPPONENTS.items():
        for p in (deck, wts):
            if p and not Path(p).is_file():
                raise SystemExit(f"setup {setup!r} opponent {name!r}: missing {p}")
    print(f"[setup] {setup}: base={BASE_CKPT} deck={CAND_DECK} profile={PROFILE.name} "
          f"opponents={list(OPPONENTS)}")

_SDK = None
_CARDS = None


def _sdk_cards():
    global _SDK, _CARDS
    if _SDK is None:
        cfg = load_config(profile="benchmark")
        _SDK = load_sdk(cfg.paths.sdk_dir)
        _CARDS = CardDatabase.from_sdk(_SDK)
    return _SDK, _CARDS


def _register_screen_profiles(cards):
    """Rebuild + register each screen opponent's GENERIC_<SLUG> profile in-process."""
    for _, (_, _, _, slug) in OPPONENTS.items():
        if slug is None:
            continue
        deck = [int(x) for x in Path(f"decks/{slug}.csv").read_text().split()]
        prof = dp.build_generic_profile(f"GENERIC_{slug.upper()}", tuple(deck), cards)
        dp.PROFILES[prof.name] = prof


# ---------- member <-> numpy conversion ----------

def _load_main_members(ckpt_path):
    payload = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
    spec = payload["contexts"]["MAIN"]
    assert spec.get("kind") == "mlp_ensemble", "base MAIN must be an mlp_ensemble"
    members = []
    for m in spec["members"]:
        members.append({
            "W1": np.asarray(m["W1"], dtype=np.float64),
            "b1": np.asarray(m["b1"], dtype=np.float64),
            "w2": np.asarray(m["w2"], dtype=np.float64),
            "b2": float(m["b2"]),
        })
    return payload, members


def _members_to_spec(members):
    return {"kind": "mlp_ensemble", "members": [
        {"kind": "mlp", "h": int(P["b1"].shape[0]),
         "W1": P["W1"].tolist(), "b1": P["b1"].tolist(),
         "w2": P["w2"].tolist(), "b2": float(P["b2"])}
        for P in members]}


def _write_ckpt(base_payload, members, out_path, meta=None):
    payload = dict(base_payload)
    payload["contexts"] = dict(payload["contexts"])
    payload["contexts"]["MAIN"] = _members_to_spec(members)
    payload["version"] = 2
    if meta:
        payload.setdefault("rl_meta", {}).update(meta)
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")


# ---------- sampling policy ----------

class SamplingImitationPolicy(ImitationPolicy):
    """Samples the MAIN action from softmax(scores/tau) instead of argmax; every other
    context and any k>1 defers to the deterministic parent. Exploration for RL."""

    def __init__(self, weights, deck=None, tau=1.0, seed=0, **kw):
        super().__init__(weights, deck=deck, **kw)
        self._tau = tau
        self._rng = random.Random(seed)
        self.n_sampled = 0
        self.n_nonargmax = 0

    def _decide(self, select, obs, gs, cards):
        ctx = select.context.name
        w = self._weights.get(ctx)
        if (ctx == "MAIN" and w is not None and cards is not None
                and len(select.option) > 0 and not F.is_prize_pick(select)):
            try:
                n = len(select.option)
                k = min(select.maxCount, n)
                k = max(k, min(select.minCount, n))
                if k == 1:
                    state = F.decision_state(self._profile, obs, gs, cards)
                    scores = [_score(w, F.featurize_option(self._profile, select, obs, gs, cards, i, state))
                              for i in range(n)]
                    m = max(scores)
                    exps = [math.exp((s - m) / self._tau) for s in scores]
                    tot = sum(exps)
                    r = self._rng.random() * tot
                    acc = 0.0
                    idx = n - 1
                    for i, e in enumerate(exps):
                        acc += e
                        if r <= acc:
                            idx = i
                            break
                    self._bc_used += 1
                    self.n_sampled += 1
                    if idx != max(range(n), key=lambda i: scores[i]):
                        self.n_nonargmax += 1
                    return [idx]
            except Exception:
                self._bc_failures += 1
        return super()._decide(select, obs, gs, cards)


# ---------- collection ----------

def _agent(kind, deck_ids, seed, weights, tau=None):
    if kind == "sampler":
        inner = SamplingImitationPolicy(weights, deck=deck_ids, tau=tau, seed=seed)
    else:
        inner = ImitationPolicy(weights, deck=deck_ids)
    return SafePolicy(inner, deck=deck_ids, seed=seed)


def _collect_slice(setup, ckpt_path, tau, n_games, shard, n_shards):
    """Play every `n_shards`-th game. Worker entry point for parallel collect.

    Sharding by GAME, not by opponent: the mix has only 3 opponents with unequal weights
    (0.40/0.35/0.25), so an opponent split leaves workers idle and measured just 1.86x on
    4 cores. Striding over the global game counter spreads all three evenly.

    `setup` MUST be passed explicitly. Windows spawns workers by re-importing this module,
    which re-runs the module body and resets SETUP to its default -- a worker that skipped
    this looked for tr650 opponents, matched none, and returned zero rows while the parent
    reported a 24x "speedup". That happened on the first run of this code.

    Games keep their GLOBAL index so `cand_side = game_no % 2` is unchanged: the
    alternation balances first/second-player advantage, and renumbering per shard would
    skew which side the candidate plays.
    """
    configure(setup)
    return _collect_impl(ckpt_path, tau, n_games, shard=(shard, n_shards))


def collect(ckpt_path, out_path, n_games, tau=1.0, seed=0, workers=1):
    """Self-play batch. `workers>1` splits BY OPPONENT across processes.

    The engine is STOCHASTIC (measured: two identical serial runs in the same process
    return different games -- it shuffles from entropy), so parallel output is NOT
    bit-identical to serial and does not need to be: collection is SAMPLING. What must
    hold is that the *distribution* is unchanged -- every opponent gets its exact game
    quota and nothing is silently dropped. `_check_collect_par.py` asserts that, and the
    `missing` guard below turns a lost worker into a hard error instead of a skewed mix.

    Windows uses spawn, so the caller must be an importable module -- a heredoc piped to
    stdin has no importable __main__ and the pool dies with BrokenProcessPool. Workers
    also receive the setup name explicitly; the re-import resets module globals.
    """
    t0 = time.perf_counter()
    if workers <= 1:
        merged = _collect_impl(ckpt_path, tau, n_games)
    else:
        from concurrent.futures import ProcessPoolExecutor
        names = list(OPPONENTS)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_collect_slice, SETUP, ckpt_path, tau, n_games, i, workers)
                    for i in range(workers)]
            parts = [f.result() for f in futs]
        merged = {"rows": [], "by_opp": {}, "n_sampled": 0, "n_nonargmax": 0, "bc_failures": 0}
        for name in names:                     # keep the serial opponent ordering
            games = decided = wins = 0
            for p in parts:
                merged["rows"].extend(p["rows_by_opp"].get(name, []))
                st = p["by_opp"].get(name)
                if st:
                    games += st["games"]; decided += st["decided"]; wins += st["wins"]
            if games:
                merged["by_opp"][name] = {"games": games, "decided": decided, "wins": wins,
                                          "cand_winrate": round(wins / max(decided, 1), 3)}
        missing = [n for n in names if n not in merged["by_opp"]]
        if missing:
            raise RuntimeError(
                f"parallel collect lost opponents {missing} — workers returned nothing for "
                f"them. Silent data loss here would train on a skewed opponent mix.")
        for p in parts:
            for k in ("n_sampled", "n_nonargmax", "bc_failures"):
                merged[k] += p[k]

    rows = merged["rows"]
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"))); fh.write("\n")
    manifest = {
        "n_games": sum(v["games"] for v in merged["by_opp"].values()),
        "by_opp": merged["by_opp"], "tau": tau, "workers": workers,
        "n_main_rows": len(rows),
        "pct_nonargmax": round(merged["n_nonargmax"] / max(merged["n_sampled"], 1), 4),
        "bc_failures": merged["bc_failures"],
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _collect_impl(ckpt_path, tau, n_games, shard=None):
    import dataclasses
    sdk, cards = _sdk_cards()
    _register_screen_profiles(cards)
    base_cfg = load_config(profile="benchmark")
    cand = load_deck(Path(CAND_DECK))

    counts = {name: max(1, round(w * n_games)) for name, (_, _, w, _) in
              [(k, v) for k, v in OPPONENTS.items()]}

    rows = []
    rows_by_opp = {}
    by_opp = {}
    n_nonargmax = n_sampled = n_bc_failures = 0
    game_no = 0

    for opp_name, (odeck, ow, _w, _slug) in OPPONENTS.items():
        want = counts[opp_name]
        opp_rows = []
        odk = load_deck(Path(odeck))
        cfg = dataclasses.replace(base_cfg, paths=dataclasses.replace(
            base_cfg.paths, deck_path=Path(CAND_DECK), opponent_deck_path=Path(odeck)))
        env = BattleEnvironment(cfg, sdk=sdk)
        runner = BattleRunner(env, max_decisions=cfg.battle.max_decisions)
        wins = decided = played = 0
        for g in range(want):
            game_no += 1
            # shard AFTER incrementing, so cand_side (game_no % 2) is the global value
            if shard is not None and (game_no - 1) % shard[1] != shard[0]:
                continue
            played += 1
            cand_side = game_no % 2
            gid = uuid4().hex
            pol_cand = _agent("sampler", cand.as_list(), 1, ckpt_path, tau=tau)
            pol_opp = _agent("imitation", odk.as_list(), 2, ow)
            pol_cand.on_battle_start(); pol_opp.on_battle_start()
            ag_cand = PTCGAgent(pol_cand, deck=cand, cards=cards)
            ag_opp = PTCGAgent(pol_opp, deck=odk, cards=cards)

            game_rows = []

            def hook(obs, action, player, elapsed_ms, _cs=cand_side, _gid=gid, _buf=game_rows):
                if player != _cs:
                    return
                sel = obs.get("select") if isinstance(obs, dict) else None
                if not sel or int(sel.get("context", -1)) != MAIN_CTX:
                    return
                opts = sel.get("option") or []
                if not isinstance(action, list) or len(action) != 1 or len(opts) < 2:
                    return
                raw = json.loads(json.dumps({"select": sel, "current": obs.get("current")}))
                _buf.append({
                    "game_id": _gid, "seat": _cs, "step_index": len(_buf),
                    "raw_observation": raw, "action": [int(action[0])], "won": None,
                    "context": MAIN_CTX, "select_type": int(sel.get("type", -1)),
                    "min_count": int(sel.get("minCount", 1)),
                    "max_count": int(sel.get("maxCount", 1)), "n_options": len(opts),
                })

            if cand_side == 0:
                rec = runner.run(ag_cand, ag_opp, cand.as_list(), odk.as_list(), on_decision=hook)
            else:
                rec = runner.run(ag_opp, ag_cand, odk.as_list(), cand.as_list(), on_decision=hook)
            ci = pol_cand._inner
            n_sampled += ci.n_sampled; n_nonargmax += ci.n_nonargmax
            n_bc_failures += ci._bc_failures
            if rec.outcome.winner is None:
                continue  # draw: discard this game's rows (false credit signal)
            decided += 1
            won = rec.outcome.winner == cand_side
            wins += int(won)
            for row in game_rows:
                row["won"] = won
                rows.append(row)
                opp_rows.append(row)
        rows_by_opp[opp_name] = opp_rows
        by_opp[opp_name] = {"games": played, "decided": decided, "wins": wins,
                            "cand_winrate": round(wins / max(decided, 1), 3)}
    return {"rows": rows, "rows_by_opp": rows_by_opp, "by_opp": by_opp,
            "n_sampled": n_sampled, "n_nonargmax": n_nonargmax,
            "bc_failures": n_bc_failures}


# ---------- value baseline ----------

def _load_v650(path=V650):
    v = json.loads(Path(path).read_text(encoding="utf-8"))
    return (np.asarray(v["mean"]), np.asarray(v["std"]),
            np.asarray(v["weights"]), float(v["bias"]),
            tuple(v.get("features", _BASE7)))


def _critic_vec(obs, cards, feat_names):
    """Feature vector for whichever critic is loaded. ``vf2.features`` is a superset
    whose base-7 columns are identical to ``train_value.features`` (verified), so a
    7-feature payload (v_650/v_rl) selects columns 0-6 unchanged, and the M17 v2 payload
    additionally selects ``prose_threat``. One path, backward compatible."""
    full = vf2.features(obs, cards)
    if full is None:
        return None
    return [full[vf2.FEATURE_NAMES.index(n)] for n in feat_names]


def _value(obs, cards, v650):
    mu, sd, w, b, feat_names = v650
    f = _critic_vec(obs, cards, feat_names)
    if f is None:
        return 0.5
    z = float(((np.asarray(f) - mu) / sd) @ w + b)
    return 1.0 / (1.0 + math.exp(-z))


# ---------- featurize trajectory with advantage ----------

def _featurize_traj(traj_path, cards, parser, v650, gae_lambda=None, gamma=1.0):
    """Featurize one self-play batch and attach an advantage to every decision.

    ``gae_lambda=None`` (the default) is the Monte-Carlo estimator M16-M47 all ran:
    ``A_t = R - V(s_t)``, with R the FINAL game outcome. M48 measured what that costs
    (`m48_advantage_variance.py`, on a real batch): with the refit critic **87.5% of the
    variance of A is BETWEEN games** -- identical for all ~45 MAIN decisions of a game,
    carrying no information about which decision was good. (With the pre-M47 critic it was
    92.5%; tripling the critic's AUC moved the discriminating share only 7.5% -> 12.5%.)
    That is the arithmetic behind M47's measured failure mode: the gradient direction is
    dominated by "which games did I happen to win", which re-rolls every batch, so
    consecutive updates anti-correlate (cosine -0.283) and cancel.

    ``gae_lambda`` in [0, 1] switches to generalized advantage estimation:

        delta_t = gamma*V(s_{t+1}) - V(s_t)        (last step of a game: R - V(s_T))
        A_t     = sum_k (gamma*gae_lambda)^k delta_{t+k}

    which is WITHIN-game by construction. lambda=1 recovers the Monte-Carlo estimator
    (up to the telescoping), lambda=0 is pure one-step TD -- lowest variance, highest bias.
    The trade is real and must be stated: GAE swaps outcome noise for CRITIC ERROR, so it
    is only worth doing with a critic worth bootstrapping from (ours: held-out AUC 0.79).

    NOTE the name. ``lambda_a`` in `rl_update` is the L2 anchor to the BC weights, a
    completely different quantity. Collapsing the two names would be exactly the class of
    inherited-constant confusion that produced four bugs in M38 alone.

    Row order in the returned list is the FILE order in both modes, so `_pack` builds a
    bit-identical bigX and only the weights differ -- the comparison stays single-variable.
    """
    return advantages(_prepare_traj(traj_path, cards, parser, v650),
                      gae_lambda=gae_lambda, gamma=gamma)


def _prepare_traj(traj_path, cards, parser, v650):
    """The EXPENSIVE half of `_featurize_traj`: parse, build state, featurize, value.

    Split out because none of it depends on the advantage estimator, so a sweep over
    `gae_lambda` (or over anything else that only re-weights decisions) can pay the
    featurisation once instead of once per arm. Returns
    ``(game_id, step_index, X, chosen, V(s), R)`` per usable MAIN decision, in file order.
    """
    prepared = []
    for r in read_decision_dataset(Path(traj_path)):
        if int(r.context) != MAIN_CTX:
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if len(chosen) != 1:
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(PROFILE, obs.select, obs, gs, cards), dtype=np.float64)
        prepared.append((r.game_id, int(r.step_index), X, chosen,
                         _value(obs, cards, v650), 1.0 if r.won else 0.0))
    return prepared


def advantages(prepared, gae_lambda=None, gamma=1.0):
    """Attach advantages to prepared rows. See `_featurize_traj` for the estimator."""
    if gae_lambda is None:
        return [Decision(X=X, chosen=c, weight=R - V)
                for _, _, X, c, V, R in prepared]

    by_game: dict[str, list] = {}
    for item in prepared:
        by_game.setdefault(item[0], []).append(item)
    adv_by_key: dict[tuple, float] = {}
    for gid, items in by_game.items():
        items.sort(key=lambda t: t[1])          # step_index, not file order
        acc = 0.0
        for t in range(len(items) - 1, -1, -1):
            V_t = items[t][4]
            if t == len(items) - 1:
                delta = items[t][5] - V_t       # bootstrap the tail on the true outcome
            else:
                delta = gamma * items[t + 1][4] - V_t
            acc = delta + gamma * gae_lambda * acc
            adv_by_key[(gid, items[t][1])] = acc
    return [Decision(X=X, chosen=c, weight=adv_by_key[(gid, si)])
            for gid, si, X, c, _, _ in prepared]


# ---------- REINFORCE update ----------

def rl_update(members, members_bc, decisions, lambda_a=0.05, tau=1.0, lr=1e-3, epochs=10,
              clip=None, adv_norm=False, entropy=0.0):
    """REINFORCE with a value baseline, optionally with PPO's ratio clip.

    M47. The three keyword args are ADDITIVE and OFF by default: with
    ``clip=None, adv_norm=False, entropy=0.0`` every line below reduces to the exact
    expression M16/M19/M38 ran, so those runs stay reproducible
    (``tests/unit/test_rl_update_regression.py`` asserts bit-equality).

    ``clip`` (PPO epsilon). The defect M47 exists to fix: this function reuses ONE
    collected batch for ``epochs`` gradient steps with no correction for the policy
    having moved between them. In the real M38 run `probe_agree` eroded 0.966 -> 0.8835
    over 10 iterations with no win-rate gain -- the instability PPO's clip exists to
    prevent. ``old_logp`` is snapshotted ONCE, before the first step, so it is the
    policy that actually COLLECTED the batch; that is PPO's contract, and reusing the
    running policy instead would make the ratio identically 1 and the clip a no-op.
    Validated on GPU against 3 already-collected M38 transitions before being ported
    here (build/colab_rl_extra/colab_rl_clip_test.py).

    ``adv_norm``. A = R - V(s) with R in {0,1}, so std(A) ~ 0.4-0.5 and the batch mean
    is NOT zero (the sampling policy wins ~45% of collected games). Normalising
    multiplies the effective step by ~2-2.5x, so lr must come down by the same factor
    -- do not turn this on without re-tuning lr.
    """
    bigX, starts, lens, chosen_global, adv = _pack(decisions)
    if adv_norm:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    onehot = np.zeros(bigX.shape[0]); onehot[chosen_global] = 1.0
    adv_rep = np.repeat(adv, lens)
    N = len(decisions); M = len(members)
    opts = [{k: _Adam(np.shape(P[k]), lr) for k in ("W1", "b1", "w2", "b2")} for P in members]
    old_logp = None
    for _ in range(epochs):
        fwd = [_mlp_scores(bigX, P) for P in members]     # (scores, Z1, A1) per member
        mean_scores = sum(f[0] for f in fwd) / M
        probs = _seg_softmax(mean_scores / tau, starts, lens)
        if clip is None:
            g = (probs - onehot) * adv_rep / (N * tau)     # dL/d(mean_score)
        else:
            logp = np.log(np.maximum(probs[chosen_global], 1e-12))
            if old_logp is None:
                old_logp = logp.copy()                     # the collecting policy, frozen
            ratio = np.exp(logp - old_logp)
            clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip)
            # d/ds of min(r*A, clip(r)*A) is the unclipped branch's gradient when that
            # branch is the smaller one (which includes r inside the band, where the two
            # are equal), and ZERO when the clip binds. `active` is that indicator.
            active = (ratio * adv <= clipped * adv).astype(np.float64)
            g = (probs - onehot) * np.repeat(adv * ratio * active, lens) / (N * tau)
        if entropy:
            # Maximise H = -sum p log p, i.e. subtract beta*H from the loss.
            logp_all = np.log(np.maximum(probs, 1e-12))
            H = -np.add.reduceat(probs * logp_all, starts)
            g = g + (entropy / (N * tau)) * probs * (logp_all + np.repeat(H, lens))
        for mi, P in enumerate(members):
            _, Z1, A1 = fwd[mi]
            dscore = g / M
            dw2 = A1.T @ dscore + lambda_a * (P["w2"] - members_bc[mi]["w2"])
            db2 = float(dscore.sum())
            dZ1 = np.outer(dscore, P["w2"]) * (Z1 > 0)
            dW1 = bigX.T @ dZ1 + lambda_a * (P["W1"] - members_bc[mi]["W1"])
            db1 = dZ1.sum(axis=0)
            P["W1"] = opts[mi]["W1"].step(P["W1"], dW1)
            P["b1"] = opts[mi]["b1"].step(P["b1"], db1)
            P["w2"] = opts[mi]["w2"].step(P["w2"], dw2)
            P["b2"] = float(opts[mi]["b2"].step(np.array(P["b2"]), db2))
    return members


# ---------- probe (M10 lesson) ----------

def _probe_decisions(cards, parser, limit=None):
    rows = list(read_decision_dataset(Path(DATASET)))
    _, val = split_by_game(rows, val_fraction=0.2, seed=0)
    out = []
    for r in val:
        if int(r.context) != MAIN_CTX:
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        n = len(obs.select.option)
        chosen = [a for a in r.action if 0 <= a < n]
        if len(chosen) != 1:
            continue
        gs = GameState.build(obs, cards)
        X = np.asarray(F.featurize_decision(PROFILE, obs.select, obs, gs, cards), dtype=np.float64)
        out.append((X, chosen[0]))
        if limit and len(out) >= limit:
            break
    return out


def _ensemble_argmax(members, X):
    return int(np.argmax(sum(_mlp_scores(X, P)[0] for P in members)))


def _probe(members, members_bc, probe):
    agree = teacher = 0
    for X, tc in probe:
        a = _ensemble_argmax(members, X)
        agree += (a == _ensemble_argmax(members_bc, X))
        teacher += (a == tc)
    n = max(len(probe), 1)
    return agree / n, teacher / n


# ---------- mini-eval (deterministic argmax, as deployed) ----------

def evaluate_vs_init(ckpt, n=200):
    """THE unsaturated measure: candidate vs the FROZEN INIT, same deck both sides.

    Why this replaces the gauntlet for M38. The field gauntlet is saturated -- the shipped
    clone wins 96.5% macro with 67/72 decks at 100%, so it has 3.5% of headroom and cannot
    rank two candidates. M16 read "+0.027 against a +0.05 bar" off exactly that instrument
    and the line was closed on it.

    A mirror against the init cannot saturate: it is 0.500 by construction at iteration 0
    and has no ceiling. It answers the only question that matters here -- "is this better
    than what it started from?" -- with no baseline table to go stale.
    """
    sdk, cards = _sdk_cards()
    arena_m8._SDK, arena_m8._CARDS = sdk, cards
    stats = arena_m8.run("imitation", CAND_DECK, "imitation", CAND_DECK,
                         "ckpt vs frozen init", n=n, wa=ckpt, wb=BASE_CKPT)
    wr = stats.score_rate
    se = math.sqrt(max(wr * (1 - wr), 1e-9) / n)
    lo, hi = wr - 1.96 * se, wr + 1.96 * se
    verdict = "BETTER" if lo > 0.5 else ("WORSE" if hi < 0.5 else "tie (within noise)")
    print(f"  vs frozen init: {wr:.3f}  [95% {lo:.3f}-{hi:.3f}]  n={n}  -> {verdict}")
    return wr, lo, hi


def evaluate(ckpt, n=120, names=None):
    sdk, cards = _sdk_cards()
    _register_screen_profiles(cards)
    arena_m8._SDK, arena_m8._CARDS = sdk, cards  # share cache; avoid a 2nd native SDK load

    if SETUP != "tr650":
        # No hardcoded baseline table: measure the candidate AND the frozen init against
        # the same opponents in the same run, so the reference can never go stale (the
        # M37 mistake -- a sweep compared itself to its own retrained arm instead of the
        # artifact actually shipped).
        wr_init, lo, hi = evaluate_vs_init(ckpt, n=max(n, 200))
        scores = {}
        for name in (names or list(OPPONENTS.keys())):
            odeck, ow = OPPONENTS[name][0], OPPONENTS[name][1]
            cand = arena_m8.run("imitation", CAND_DECK, "imitation", odeck,
                                f"ckpt vs {name}", n=n, wa=ckpt, wb=ow).score_rate
            base = arena_m8.run("imitation", CAND_DECK, "imitation", odeck,
                                f"init vs {name}", n=n, wa=BASE_CKPT, wb=ow).score_rate
            scores[name] = (cand, base)
        print("\n" + "=" * 62)
        print(f"{'opponent':<14}{'ckpt':>9}{'init':>9}{'delta':>9}")
        print("-" * 62)
        for k, (c, b) in scores.items():
            print(f"{k:<14}{c:>9.3f}{b:>9.3f}{c - b:>+9.3f}")
        pooled = sum(c - b for c, b in scores.values()) / max(len(scores), 1)
        print("-" * 62)
        print(f"pooled delta vs init: {pooled:+.3f}   |   head-to-head vs init: {wr_init:.3f}")
        print("SHIP BAR (pre-registered): head-to-head >= 0.55 with the 95% CI excluding 0.500")
        return {k: v[0] for k, v in scores.items()}, pooled, wr_init

    names = names or (list(OPPONENTS.keys()) + ["cinderace"])
    scores = {}
    for name in names:
        odeck, ow = (HELDOUT[1], HELDOUT[2]) if name == "cinderace" else (OPPONENTS[name][0], OPPONENTS[name][1])
        stats = arena_m8.run("imitation", CAND_DECK, "imitation", odeck,
                             f"ckpt vs {name}", n=n, wa=ckpt, wb=ow)
        scores[name] = stats.score_rate
    d12 = [scores[k] - V12_BASELINE[k] for k in scores]
    d1 = [scores[k] - V1_BASELINE[k] for k in scores]
    print("\n" + "=" * 68)
    print(f"{'opponent':<12}{'ckpt':>8}{'v1.2 base':>11}{'d(v1.2)':>9}{'v1 base':>9}{'d(v1)':>8}")
    print("-" * 68)
    for k in scores:
        print(f"{k:<12}{scores[k]:>8.3f}{V12_BASELINE[k]:>11.3f}{scores[k]-V12_BASELINE[k]:>+9.3f}"
              f"{V1_BASELINE[k]:>9.3f}{scores[k]-V1_BASELINE[k]:>+8.3f}")
    print("-" * 68)
    print(f"pooled delta vs v1.2: {sum(d12)/len(d12):+.3f}   vs v1: {sum(d1)/len(d1):+.3f}   "
          f"(held-out cinderace d(v1.2)={scores.get('cinderace',0)-V12_BASELINE['cinderace']:+.3f})")
    return scores, sum(d12) / len(d12), sum(d1) / len(d1)


def greedy_anchor(ckpt, n=60):
    """K3: candidate vs the greedy field slice (SAMPLE/meta_lucario/meta_dragapult).
    Catastrophic-forgetting detector; v1.2 baseline here is ~0.90 (saturated)."""
    sdk, cards = _sdk_cards()
    arena_m8._SDK, arena_m8._CARDS = sdk, cards
    scores = []
    for gd in (arena_m8.SAMPLE, arena_m8.LUC, arena_m8.DRAG):
        stats = arena_m8.run("imitation", CAND_DECK, "greedy", gd,
                             f"ckpt vs greedy({Path(gd).stem})", n=n, wa=ckpt)
        scores.append(round(stats.score_rate, 3))
    return sum(scores) / len(scores), scores


# ---------- C2.2 critic refresh (adopt only if it beats v_650 on held-out) ----------

def refresh_critic(train_trajs, val_traj, out_path="data/models/v_rl.json", features="base7"):
    """Refit the P(win|state) logreg on self-play states, temporal split: train on early
    batches, validate on a later one. Adopt only if held-out AUC beats the incumbent
    critic on the SAME val set.

    M47: `features` selects the feature set. "base7" is `train_value.features`, what
    M16-M39 used. "vf2" is `value_features_v2`, the M17 base-7 + prose-aware threat set;
    `_critic_vec` already selects critic columns BY NAME, so a payload written with
    either name list loads through the same path with no other change.

    The advantage A = R - V(s) multiplies every gradient in `rl_update`, so the critic's
    calibration bounds what the whole RL loop can do -- and `v_alakazam.json` was fit
    before a single Alakazam self-play game existed. There are ~350k labelled self-play
    decisions on disk that were never used for this.
    """
    from train_value import _fit_logreg, _auc, FEATURE_NAMES
    sdk, cards = _sdk_cards()
    parser = ObservationParser()
    # `features` may name one set or several. Building the matrix is the expensive part
    # (~350k decisions of GameState construction); the column selection is free, so a
    # multi-set request costs one pass, not one pass each.
    _SETS = {"base7": list(FEATURE_NAMES), "vf2": list(vf2.FEATURE_NAMES)}
    wanted = [features] if isinstance(features, str) else list(features)
    for name in wanted:
        if name not in _SETS:
            raise SystemExit(f"unknown feature set {name!r}; choose from {sorted(_SETS)}")

    # Always build in vf2's column order (a superset whose base-7 columns are identical,
    # the same invariant `_critic_vec` relies on). Each critic then SELECTS its own
    # columns by name, so the candidate and the incumbent are scored on exactly the same
    # rows even when they use different feature sets.
    full_names = list(vf2.FEATURE_NAMES)

    def build(paths):
        X, y = [], []
        for p in ([paths] if isinstance(paths, str) else paths):
            for r in read_decision_dataset(Path(p)):
                if int(r.context) != MAIN_CTX:
                    continue
                obs = parser.parse(r.raw_observation)
                f = vf2.features(obs, cards)
                if f is None:
                    continue
                X.append(f); y.append(1.0 if r.won else 0.0)
        return np.asarray(X), np.asarray(y)

    def cols(X, names):
        return X[:, [full_names.index(n) for n in names]]

    t0 = time.perf_counter()
    Xtr_all, ytr = build(train_trajs)
    Xva_all, yva = build(val_traj)
    print(f"built train={Xtr_all.shape} val={Xva_all.shape} in "
          f"{time.perf_counter() - t0:.0f}s  (train win rate {ytr.mean():.3f})", flush=True)

    # The bar: the incumbent, scored on the SAME held-out rows through its own columns.
    old = json.loads(Path(V650).read_text(encoding="utf-8"))
    old_names = list(old.get("features", _BASE7))
    omu, osd = np.asarray(old["mean"]), np.asarray(old["std"])
    ow, ob = np.asarray(old["weights"]), float(old["bias"])
    old_auc = _auc(((cols(Xva_all, old_names) - omu) / osd) @ ow + ob, yva)
    print(f"incumbent {Path(V650).name} ({len(old_names)} features): held-out AUC {old_auc:.4f}")

    best = None
    for name in wanted:
        feat_names = _SETS[name]
        Xtr, Xva = cols(Xtr_all, feat_names), cols(Xva_all, feat_names)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        w, b = _fit_logreg((Xtr - mu) / sd, ytr)
        auc = _auc(((Xva - mu) / sd) @ w + b, yva)
        print(f"  candidate [{name}] ({len(feat_names)} features): held-out AUC {auc:.4f} "
              f"({auc - old_auc:+.4f} vs incumbent)")
        if best is None or auc > best[0]:
            best = (auc, name, feat_names, mu, sd, w, b)

    auc, name, feat_names, mu, sd, w, b = best
    if auc > old_auc:
        Path(out_path).write_text(json.dumps({
            "version": 1, "features": feat_names,
            "mean": mu.tolist(), "std": sd.tolist(), "weights": w.tolist(), "bias": float(b),
            "val_auc": auc,
            "trained_on": f"{name} on {train_trajs}, val {val_traj}",
        }), encoding="utf-8")
        print(f"ADOPTED [{name}] -> {out_path}")
        return out_path
    print(f"NOT adopted ({Path(V650).name} still better on held-out); continue with it")
    return V650


# ---------- the C2 loop with coded kill criteria ----------

def iterate(start_ckpt, iters, start_iter=2, batch=2000, epochs=25, lam=0.1,
            eval_every=2, eval_n=160, tau0=1.0, critic=V650, lam_final=None,
            lam_levels=None, lam_max_holds=2, workers=1, lr=1e-3,
            clip=None, adv_norm=False, entropy=0.0, tag=None,
            gae_lambda=None, gamma=1.0):
    sdk, cards = _sdk_cards()
    parser = ObservationParser()
    _, members_bc = _load_main_members(BASE_CKPT)
    v650 = _load_v650(critic)
    # K2 only needs a representative sample to detect bc_acc DRIFT, not the whole split.
    # Uncapped on the Alakazam corpus that is ~23.6k decisions of pure-Python featurisation
    # -- a measured ~40 min tax before the first game is even played, paid again on every
    # relaunch. At n=4000 the s.e. on bc_acc is ~0.004, so the 0.93 kill floor is still
    # ~5 s.e. below the ~0.95 starting point. tr650 stays uncapped so M16/M19 reproduce.
    probe_limit = None if SETUP == "tr650" else 4000
    t_probe = time.perf_counter()
    probe = _probe_decisions(cards, parser, limit=probe_limit)
    # K2's floor must be RELATIVE to what the untouched base scores on this probe, not the
    # absolute 0.93 M16 hardcoded. That number came from the TR-650 model; the Alakazam
    # base scores 0.733, so the absolute floor would kill iteration 1 of a perfectly
    # healthy run -- it did, on the first attempt.
    base_agree, base_acc = _probe(members_bc, members_bc, probe)
    k2_acc_floor = 0.93 if SETUP == "tr650" else round(base_acc - 0.03, 4)
    print(f"[probe] {len(probe)} decisions in {time.perf_counter() - t_probe:.0f}s "
          f"(limit={probe_limit}) | base bc_acc={base_acc:.3f} -> K2 floor={k2_acc_floor:.3f}",
          flush=True)
    # Namespace every artefact by setup. The tr650 run wrote data/models/rl_ckpt_iter2..33
    # and data/rl/loop_manifest.json; reusing those names would silently overwrite the
    # M16/M19 history that this experiment is being compared against.
    # M47 `--tag` extends the same namespacing to ARMS of one setup. Two arms that differ
    # only by a CLI flag (e.g. lambda) share a SETUP, so without this the second run
    # overwrites the first's checkpoints AND appends to its manifest, silently merging two
    # experiments into one log.
    run_name = SETUP if not tag else f"{SETUP}_{tag}"
    rl_dir = Path("data/rl") if SETUP == "tr650" else Path(f"data/rl_{run_name}")
    rl_dir.mkdir(parents=True, exist_ok=True)
    man_path = rl_dir / "loop_manifest.json"
    log = json.loads(man_path.read_text()) if man_path.exists() else []

    ckpt = start_ckpt
    tau = tau0
    last = start_iter + iters - 1
    n_iters = last - start_iter + 1
    # M19 conditional-gradual λ-anneal state (active only when lam_levels is passed).
    conditional = lam_levels is not None
    lam_level_idx = 0
    holds = 0
    for t in range(start_iter, last + 1):
        if conditional:
            # λ for this iter = current schedule level; it descends only when a prior
            # step cleared the kill floors WITH margin (decided at the end of the loop).
            lam_t = lam_levels[lam_level_idx]
        elif lam_final is not None and n_iters > 1:
            # M17 λ-anneal: linear, calendar-fixed across surviving iterations.
            frac = (t - start_iter) / (n_iters - 1)
            lam_t = round(lam + (lam_final - lam) * frac, 4)
        else:
            lam_t = lam
        traj = str(rl_dir / f"iter{t}_traj.jsonl.gz")
        collected_tau = tau
        man = collect(ckpt, traj, batch, tau=collected_tau, seed=t, workers=workers)
        entry = {"iter": t, "tau": collected_tau, "lambda": lam_t, "critic": str(critic),
                 "start_ckpt": str(start_ckpt), "collect_wr": man["by_opp"],
                 "pct_nonargmax": man["pct_nonargmax"], "bc_failures": man["bc_failures"],
                 "n_main_rows": man["n_main_rows"]}
        if conditional:
            entry["lam_level_idx"] = lam_level_idx
            entry["holds_at_level"] = holds

        # K5 plumbing: any BC failure during collection is a hard stop
        if man["bc_failures"] > 0:
            entry["kill"] = "K5 bc_failures>0"; log.append(entry)
            man_path.write_text(json.dumps(log, indent=2)); print(f"[it{t}] KILL K5"); break

        base, members = _load_main_members(ckpt)
        decs = _featurize_traj(traj, cards, parser, v650,
                               gae_lambda=gae_lambda, gamma=gamma)
        # lr was hardcoded at 1e-3 -- the value tuned for TR-650, whose scores are 2.0-2.4x
        # wider. On the Alakazam base it collapses the model in ONE update (bc_acc
        # 0.733 -> 0.168, measured). scratchpad/_sweep_update.py swept it on a real
        # trajectory: 3e-5 with 10 epochs moves 3.5% of decisions and leaves bc_acc at
        # 0.732, which is the M16 profile scaled to this model.
        members = rl_update(members, members_bc, decs, lambda_a=lam_t, tau=collected_tau,
                            lr=lr, epochs=epochs, clip=clip, adv_norm=adv_norm,
                            entropy=entropy)
        out_ckpt = ("data/models/rl_ckpt_iter%d.json" % t if SETUP == "tr650"
                    else "data/models/rl_%s_iter%d.json" % (run_name, t))
        _write_ckpt(base, members, out_ckpt,
                    meta={"lambda": lam_t, "epochs": epochs, "tau": collected_tau,
                          "lr": lr, "clip": clip, "adv_norm": adv_norm, "entropy": entropy,
                          "n_decisions": len(decs)})
        agree, bc_acc = _probe(members, members_bc, probe)
        entry.update({"ckpt": out_ckpt, "n_decisions": len(decs),
                      "probe_agree": round(agree, 4), "bc_val_acc": round(bc_acc, 4)})
        # M47 kill signal. The whole premise is that the clip slows the drift M38
        # measured (probe_agree 0.966 -> 0.8835 over 10 iters). Print M38's value for
        # the SAME iteration next to ours so a run that is NOT beating that curve is
        # visible in the log at iteration 6, not at the post-mortem.
        ref = _M38_DRIFT.get(t)
        ref_txt = f" (M38 was {ref:.3f}, delta {agree - ref:+.3f})" if ref else ""
        print(f"[it{t}] lam={lam_t} rows={len(decs)} nonargmax={man['pct_nonargmax']:.1%} "
              f"probe_agree={agree:.3f}{ref_txt} bc_acc={bc_acc:.3f} "
              f"collect_wr={ {k: v['cand_winrate'] for k, v in man['by_opp'].items()} }")

        # tau monitor for NEXT iteration (this batch already used collected_tau)
        if man["pct_nonargmax"] < 0.03:
            tau = round(tau + 0.1, 2)

        # K2 drift
        if bc_acc < k2_acc_floor or agree < 0.85:
            entry["kill"] = f"K2 drift (agree={agree:.3f} bc_acc={bc_acc:.3f})"
            log.append(entry); man_path.write_text(json.dumps(log, indent=2))
            print(f"[it{t}] KILL K2"); break

        # periodic eval (always on the last iteration). g_mean stays None on non-eval
        # iters so the conditional-anneal margin check below treats it as "not clear".
        g_mean = None
        if t % eval_every == 0 or t == last:
            if SETUP == "tr650":
                scores, d12, d1 = evaluate(out_ckpt, n=eval_n)
                g_mean, g_scores = greedy_anchor(out_ckpt, n=60)
                entry["eval"] = {"scores": {k: round(v, 3) for k, v in scores.items()},
                                 "pooled_d12": round(d12, 4), "pooled_d1": round(d1, 4),
                                 "greedy_anchor": g_mean, "greedy_scores": g_scores}
                print(f"[it{t}] EVAL pooled_d12={d12:+.3f} pooled_d1={d1:+.3f} "
                      f"cinderace={scores['cinderace']:.3f} greedy_anchor={g_mean:.3f}")
                # K3 greedy forgetting. Floor calibrated to the measured baseline on THIS
                # 3-deck slice (v1.2=0.787, v1=0.81; meta_dragapult/meta_lucario are near-
                # mirrors, NOT the ~0.90 full field), floor = v1.2 base - 0.05.
                if g_mean < 0.74:
                    entry["kill"] = f"K3 greedy_anchor={g_mean:.3f}<0.74"
                    log.append(entry); man_path.write_text(json.dumps(log, indent=2))
                    print(f"[it{t}] KILL K3"); break
                # K4 held-out overfit
                if scores["cinderace"] < V12_BASELINE["cinderace"] - 0.05:
                    entry["kill"] = f"K4 cinderace={scores['cinderace']:.3f} collapsed"
                    log.append(entry); man_path.write_text(json.dumps(log, indent=2))
                    print(f"[it{t}] KILL K4"); break
            else:
                # M38: K3 (greedy_anchor) and K4 (cinderace) are TR-650 instruments --
                # meta_lucario/meta_dragapult and the cinderace probe don't exist in this
                # deck's vocabulary, and V12_BASELINE is a different model's number. Both
                # would be one more "constant calibrated for another context, inherited
                # unchecked" (the pattern that already caused four bugs this milestone).
                # `evaluate()` already runs the unsaturated head-to-head vs the frozen
                # init and prints its own report; only K2 (bc_acc drift, above) guards
                # this setup for now.
                scores, pooled, wr_init = evaluate(out_ckpt, n=eval_n)
                entry["eval"] = {"scores": {k: round(v, 3) for k, v in scores.items()},
                                 "pooled_vs_init": round(pooled, 4),
                                 "wr_vs_init": round(wr_init, 4)}
                print(f"[it{t}] EVAL pooled_vs_init={pooled:+.3f} wr_vs_init={wr_init:.3f}")

        # M19 conditional-gradual λ-anneal decision (only when lam_levels is active).
        # Descend ONE level for the next iter only if this step cleared the K2/K3 kill
        # floors WITH margin (bc_acc 0.93→0.94, agree 0.85→0.87, anchor 0.74→0.76);
        # otherwise HOLD at the current λ and retrain to stabilize. K1 futility if a
        # level can't stabilize within lam_max_holds. Requires eval_every=1 for a fresh
        # anchor each step (else g_mean is None and the margin never clears → all holds).
        if conditional:
            margin_ok = (bc_acc >= 0.94 and agree >= 0.87
                         and g_mean is not None and g_mean >= 0.76)
            if margin_ok:
                holds = 0
                if lam_level_idx < len(lam_levels) - 1:
                    lam_level_idx += 1
                    print(f"[it{t}] lam-anneal: margin clear -> descend to lam={lam_levels[lam_level_idx]}")
                else:
                    print(f"[it{t}] lam-anneal: stable at floor lam={lam_levels[-1]}")
            else:
                holds += 1
                print(f"[it{t}] lam-anneal: margin NOT met (bc_acc={bc_acc:.3f} agree={agree:.3f} "
                      f"anchor={g_mean}) -> HOLD {holds}/{lam_max_holds} at lam={lam_t}")
                if holds > lam_max_holds:
                    entry["kill"] = f"K1 futility (λ={lam_t} unstable after {lam_max_holds} holds)"
                    log.append(entry); man_path.write_text(json.dumps(log, indent=2))
                    print(f"[it{t}] KILL K1"); break

        log.append(entry)
        man_path.write_text(json.dumps(log, indent=2))
        ckpt = out_ckpt

    print(f"\nloop segment done (iters {start_iter}..{t}); manifest -> {man_path}")
    return log


# ---------- CLI ----------

def _cmd_smoke():
    import dataclasses
    sdk, cards = _sdk_cards()
    _register_screen_profiles(cards)
    parser = ObservationParser()

    # 1) sampler parity at tau->0 reproduces ImitationPolicy argmax on the probe set
    _, members = _load_main_members(BASE_CKPT)
    probe = _probe_decisions(cards, parser, limit=300)
    det = ImitationPolicy(BASE_CKPT, deck=load_deck(Path(CAND_DECK)).as_list())
    spec = det._weights["MAIN"]
    mism = 0
    for X, _ in probe:
        # emulate: argmax via _score on each option row equals ensemble argmax
        s = [_score(spec, X[i].tolist()) for i in range(X.shape[0])]
        if int(np.argmax(s)) != _ensemble_argmax(members, X):
            mism += 1
    print(f"[1] sampler/ensemble scoring parity on {len(probe)} states: {len(probe)-mism}/{len(probe)} match")

    # 2) gradient check: finite-diff vs analytic on a tiny synthetic batch
    rng = np.random.default_rng(0)
    dim = PROFILE.feature_dim
    small = [Decision(X=rng.standard_normal((3, dim)), chosen=[rng.integers(3)],
                      weight=float(rng.standard_normal())) for _ in range(6)]
    m0 = [{"W1": rng.standard_normal((dim, 4)) * 0.1, "b1": np.zeros(4),
           "w2": rng.standard_normal(4) * 0.1, "b2": 0.0}]
    mbc = [{k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in m0[0].items()}]

    def loss(members):
        bigX, starts, lens, cg, adv = _pack(small)
        onehot = np.zeros(bigX.shape[0]); onehot[cg] = 1.0
        mean_scores = sum(_mlp_scores(bigX, P)[0] for P in members) / len(members)
        probs = _seg_softmax(mean_scores, starts, lens)
        adv_rep = np.repeat(adv, lens)
        # -sum adv*log p(chosen) / N  (lambda=0 for the check)
        logp = np.log(probs[cg] + 1e-12)
        return -float((adv * logp).sum()) / len(small)

    # analytic grad on w2[0] of member 0 via one manual step decomposition
    eps = 1e-6
    P = m0[0]
    base = loss([{k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in P.items()}])
    Pp = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    Pp["w2"] = np.copy(P["w2"]); Pp["w2"][0] += eps
    num = (loss([Pp]) - base) / eps
    # analytic: run rl_update grad expression for one step, read dw2[0]
    bigX, starts, lens, cg, adv = _pack(small)
    onehot = np.zeros(bigX.shape[0]); onehot[cg] = 1.0
    adv_rep = np.repeat(adv, lens)
    fwd = _mlp_scores(bigX, P)
    probs = _seg_softmax(fwd[0], starts, lens)
    g = (probs - onehot) * adv_rep / len(small)
    dw2_ana = (fwd[2].T @ g)[0]
    print(f"[2] gradient check dL/dw2[0]: analytic {dw2_ana:+.6f}  numeric {num:+.6f}  "
          f"diff {abs(dw2_ana-num):.2e}  {'OK' if abs(dw2_ana-num) < 1e-4 else 'FAIL'}")

    # 3) collect smoke: 20 games, check rows parse + featurize + value
    out = "data/rl/smoke_traj.jsonl.gz"
    Path("data/rl").mkdir(exist_ok=True)
    man = collect(BASE_CKPT, out, n_games=20, tau=1.0, seed=0)
    v650 = _load_v650()
    decs = _featurize_traj(out, cards, parser, v650)
    print(f"[3] collect smoke: {man['n_main_rows']} MAIN rows from {man['n_games']} games, "
          f"pct_nonargmax={man['pct_nonargmax']:.1%}, featurized={len(decs)}, "
          f"wall={man['wall_s']}s")
    print(f"    advantages: mean {np.mean([d.weight for d in decs]):+.3f} "
          f"range [{min(d.weight for d in decs):+.3f},{max(d.weight for d in decs):+.3f}]")
    print("SMOKE DONE")


def main():
    ap = argparse.ArgumentParser()
    # Parsed and applied BEFORE the subparser defaults are built, because --critic
    # defaults to V650, which configure() rebinds. Without the pre-pass an --setup
    # alakazam run would silently take the TR-650 critic -- exactly the kind of
    # silent wrong-artifact failure that cost us submission 55203764.
    ap.add_argument("--setup", default="tr650", choices=sorted(SETUPS),
                    help="which base/deck/opponent configuration to run (default tr650, "
                         "the M16-M19 setup, kept reproducible)")
    known, _ = ap.parse_known_args()
    configure(known.setup)

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    pc = sub.add_parser("collect"); pc.add_argument("ckpt"); pc.add_argument("out")
    pc.add_argument("n", nargs="?", type=int, default=2000); pc.add_argument("--tau", type=float, default=1.0)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--workers", type=int, default=1)
    pu = sub.add_parser("update"); pu.add_argument("ckpt"); pu.add_argument("traj"); pu.add_argument("out")
    pu.add_argument("--lambda", dest="lam", type=float, default=0.05); pu.add_argument("--tau", type=float, default=1.0)
    pu.add_argument("--lr", type=float, default=1e-3); pu.add_argument("--epochs", type=int, default=10)
    pu.add_argument("--critic", default=V650)
    pe = sub.add_parser("eval"); pe.add_argument("ckpt"); pe.add_argument("n", nargs="?", type=int, default=120)
    pi = sub.add_parser("iterate")
    pi.add_argument("--start-ckpt", required=True); pi.add_argument("--start-iter", type=int, default=2)
    pi.add_argument("--iters", type=int, default=4); pi.add_argument("--batch", type=int, default=2000)
    pi.add_argument("--epochs", type=int, default=25); pi.add_argument("--lambda", dest="lam", type=float, default=0.1)
    pi.add_argument("--eval-every", type=int, default=2); pi.add_argument("--eval-n", type=int, default=160)
    pi.add_argument("--tau0", type=float, default=1.0); pi.add_argument("--critic", default=V650)
    pi.add_argument("--lambda-final", dest="lam_final", type=float, default=None)
    pi.add_argument("--lam-levels", dest="lam_levels", default=None,
                    help="comma-separated λ schedule for M19 conditional-gradual anneal, "
                         "e.g. '0.10,0.085,0.07,0.055,0.04,0.03,0.02' (overrides --lambda/--lambda-final)")
    pi.add_argument("--lam-max-holds", dest="lam_max_holds", type=int, default=2)
    pi.add_argument("--workers", type=int, default=1,
                    help="processes for self-play collection; 4 physical cores here")
    pi.add_argument("--tag", default=None,
                    help="namespace suffix for checkpoints and the manifest, so two ARMS "
                         "of the same setup (e.g. two lambdas) do not overwrite each other")
    pi.add_argument("--lr", type=float, default=1e-3,
                    help="Adam lr for the REINFORCE update. 1e-3 is the M16/TR-650 value "
                         "and DESTROYS the Alakazam base; use 3e-5 there (see _sweep_update.py)")
    # M47 additions. Absent => the M16/M19/M38 update path, bit-for-bit.
    for p in (pu, pi):
        p.add_argument("--clip", type=float, default=None,
                       help="PPO probability-ratio clip epsilon (0.2 is standard). Absent "
                            "= no clip, the update M38 actually ran.")
        p.add_argument("--adv-norm", dest="adv_norm", action="store_true",
                       help="normalise advantages to zero mean / unit std per batch. "
                            "Multiplies the effective step ~2-2.5x: lower --lr to match.")
        p.add_argument("--entropy", type=float, default=0.0,
                       help="entropy bonus coefficient (0 = off)")
        p.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=None,
                       help="GAE lambda in [0,1]. Absent = the Monte-Carlo advantage "
                            "A = R - V(s) that M16-M47 used, whose variance M48 measured "
                            "as 87.5%% BETWEEN games (i.e. carrying no per-decision "
                            "signal). 0 = pure one-step TD, 1 ~= Monte Carlo.")
        p.add_argument("--gamma", type=float, default=1.0,
                       help="discount for the GAE recursion (1.0 for these episodic games)")
    pr = sub.add_parser("refresh_critic")
    pr.add_argument("--train", nargs="+", required=True, help="trajectory .jsonl.gz to fit on")
    pr.add_argument("--val", nargs="+", required=True, help="held-out trajectory/ies")
    pr.add_argument("--out", default="data/models/v_rl.json")
    pr.add_argument("--features", nargs="+", default=["base7"], choices=("base7", "vf2"),
                    help="feature set(s) to fit. Several cost ONE featurisation pass; "
                         "the best on held-out is the one adopted, and only if it beats "
                         "the incumbent critic.")
    args = ap.parse_args()

    if args.cmd == "smoke":
        _cmd_smoke()
    elif args.cmd == "collect":
        man = collect(args.ckpt, args.out, args.n, tau=args.tau, seed=args.seed,
                      workers=args.workers)
        print(json.dumps(man, indent=2))
    elif args.cmd == "update":
        sdk, cards = _sdk_cards()
        parser = ObservationParser()
        base, members = _load_main_members(args.ckpt)
        _, members_bc = _load_main_members(BASE_CKPT)
        v650 = _load_v650(args.critic)
        decs = _featurize_traj(args.traj, cards, parser, v650,
                               gae_lambda=args.gae_lambda, gamma=args.gamma)
        members = rl_update(members, members_bc, decs, lambda_a=args.lam, tau=args.tau,
                            lr=args.lr, epochs=args.epochs, clip=args.clip,
                            adv_norm=args.adv_norm, entropy=args.entropy)
        probe = _probe_decisions(cards, parser)
        agree, teacher = _probe(members, members_bc, probe)
        _write_ckpt(base, members, args.out,
                    meta={"lambda": args.lam, "epochs": args.epochs, "n_decisions": len(decs)})
        print(f"updated {len(decs)} decisions -> {args.out}  probe agree={agree:.3f} bc_val_acc={teacher:.3f}")
    elif args.cmd == "eval":
        evaluate(args.ckpt, n=args.n)
    elif args.cmd == "iterate":
        levels = ([float(x) for x in args.lam_levels.split(",")]
                  if args.lam_levels else None)
        iterate(args.start_ckpt, args.iters, start_iter=args.start_iter, batch=args.batch,
                epochs=args.epochs, lam=args.lam, eval_every=args.eval_every,
                eval_n=args.eval_n, tau0=args.tau0, critic=args.critic, lam_final=args.lam_final,
                lam_levels=levels, lam_max_holds=args.lam_max_holds, workers=args.workers,
                lr=args.lr, clip=args.clip, adv_norm=args.adv_norm, entropy=args.entropy,
                tag=args.tag, gae_lambda=args.gae_lambda, gamma=args.gamma)
    elif args.cmd == "refresh_critic":
        refresh_critic(args.train, args.val, out_path=args.out, features=args.features)


if __name__ == "__main__":
    raise SystemExit(main())
