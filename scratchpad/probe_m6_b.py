"""M6-0 Probes B + C: native memory (Q8) and coin-node frequency (manual_coin).

B: run the v0 loop shape (D determinizations x per-option candidates x
   random rollout to actor-flip/terminal/40 steps) at ~24 real decisions of a
   greedy-vs-greedy game; sample process RSS after each decision (search_end
   between decisions, production hygiene) plus one 5-decision stress run
   WITHOUT search_end to see accumulation.

C: half the decisions run with manual_coin=True, half False; count COIN_HEAD
   decision nodes met inside my-turn rollouts, per mode. If coin nodes are
   rare, v0 ships manual_coin=False + replicates.
"""

from __future__ import annotations

import collections
import ctypes
import random
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

from ptcg_ai.agent.agent import PTCGAgent
from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.safety import SafePolicy
from ptcg_ai.environment.adapter import BattleEnvironment
from ptcg_ai.environment.sdk import load_sdk

SAMPLE = "pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"
D_WORLDS = 4
ROLLOUT_CAP = 40
CAND_CAP = 16


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.windll.kernel32
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_gpmi = _kernel32.K32GetProcessMemoryInfo
_gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
_gpmi.restype = wintypes.BOOL


def rss_mib() -> tuple[float, float]:
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not _gpmi(_kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("K32GetProcessMemoryInfo failed")
    return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20


def filler_hidden_info(raw: dict[str, Any], deck_ids: list[int], cards: CardDatabase):
    current = raw["current"]
    me = current["yourIndex"]
    mine = current["players"][me]
    opp = current["players"][1 - me]
    ids = deck_ids
    your_deck = ids[: max(mine["deckCount"], 1)]
    your_prize = ids[: len(mine["prize"])]
    opponent_deck = ids[: max(opp["deckCount"], 1)]
    opponent_prize = ids[: len(opp["prize"])]
    opponent_hand = ids[: opp["handCount"]]
    active = opp.get("active") or []
    opponent_active: list[int] = []
    if active and active[0] is None:
        basics = [c for c in ids if (i := cards.get_card(c)) and i.is_basic_pokemon]
        opponent_active = [basics[0]]
    return (your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)


def v0_search_shape(api, raw, deck_ids, cards, rng, manual_coin: bool):
    """One decision's worth of v0-shaped work. Returns stats dict."""
    root_index = raw["current"]["yourIndex"]
    agent_obs = api.to_observation_class(raw)
    fillers = filler_hidden_info(raw, deck_ids, cards)
    steps = 0
    begins = 0
    coin_nodes = 0
    contexts = collections.Counter()
    t0 = time.perf_counter()
    for _ in range(D_WORLDS):
        root = api.search_begin(agent_obs, *fillers, manual_coin=manual_coin)
        begins += 1
        sel = root.observation.select
        if sel is None:
            continue
        n_cand = min(len(sel.option), CAND_CAP)
        for c in range(n_cand):
            k = sel.minCount
            choice = [c] + [i for i in range(len(sel.option)) if i != c][: max(0, k - 1)]
            node = api.search_step(root.searchId, choice[:k] if k > 0 else [c])
            steps += 1
            # rollout: random until actor flip / terminal / cap
            for _ in range(ROLLOUT_CAP):
                s = node.observation.select
                cur = node.observation.current
                if s is None or (cur is not None and cur.result != -1):
                    break
                if cur is not None and cur.yourIndex != root_index:
                    break  # actor flipped -> leaf
                contexts[f"{s.type}/{s.context}"] += 1
                if s.context == 46:  # COIN_HEAD
                    coin_nodes += 1
                kk = rng.randint(s.minCount, s.maxCount)
                ch = rng.sample(range(len(s.option)), kk)
                node = api.search_step(node.searchId, ch)
                steps += 1
    dt = time.perf_counter() - t0
    return {"steps": steps, "begins": begins, "coin": coin_nodes,
            "secs": dt, "contexts": contexts}


def main() -> None:
    config = load_config(profile="benchmark")
    sdk = load_sdk(config.paths.sdk_dir)
    api = sdk.api
    cards = CardDatabase.from_sdk(sdk)
    deck = load_deck(Path(SAMPLE))
    deck_ids = deck.as_list()
    rng = random.Random(23)

    base_rss, _ = rss_mib()
    print(f"baseline RSS: {base_rss:.1f} MiB")

    pols = [SafePolicy(GreedyPolicy(deck=deck_ids), deck=deck_ids, seed=1),
            SafePolicy(GreedyPolicy(deck=deck_ids), deck=deck_ids, seed=2)]
    agents = [PTCGAgent(pols[0], deck=deck, cards=cards),
              PTCGAgent(pols[1], deck=deck, cards=cards)]
    env = BattleEnvironment(config, sdk=sdk)
    raw = env.start(deck_ids, deck_ids)

    per_dec = []
    coin_by_mode = collections.Counter()
    steps_by_mode = collections.Counter()
    all_contexts = collections.Counter()
    decision_i = 0
    max_rss_seen = 0.0
    # -- normal phase: search_end after every decision ----------------------
    while decision_i < 24 and env.result() is None:
        sel = raw.get("select")
        if sel is not None and len(sel.get("option", [])) > 1:
            manual = decision_i % 2 == 0
            stats = v0_search_shape(api, raw, deck_ids, cards, rng, manual_coin=manual)
            api.search_end()
            cur_rss, peak_rss = rss_mib()
            max_rss_seen = max(max_rss_seen, cur_rss)
            per_dec.append((decision_i, stats["steps"], stats["secs"], cur_rss))
            coin_by_mode["manual" if manual else "auto"] += stats["coin"]
            steps_by_mode["manual" if manual else "auto"] += stats["steps"]
            all_contexts.update(stats["contexts"])
            decision_i += 1
        raw = env.select(agents[env.acting_player()](raw))

    print("\n=== Probe B: per-decision v0 loop (search_end between) ===")
    tot_steps = sum(s for _, s, _, _ in per_dec)
    tot_secs = sum(t for _, _, t, _ in per_dec)
    print(f"decisions probed: {len(per_dec)}  total steps: {tot_steps}  "
          f"mean steps/dec: {tot_steps / max(1, len(per_dec)):.0f}  "
          f"mean secs/dec: {tot_secs / max(1, len(per_dec)):.3f}")
    print(f"slowest decisions: {sorted(per_dec, key=lambda x: -x[2])[:3]}")
    print(f"RSS after decisions: max {max_rss_seen:.1f} MiB (baseline {base_rss:.1f})")

    # -- stress phase: NO search_end across 5 decisions ---------------------
    stress_steps = 0
    stress_n = 0
    while stress_n < 5 and env.result() is None:
        sel = raw.get("select")
        if sel is not None and len(sel.get("option", [])) > 1:
            stats = v0_search_shape(api, raw, deck_ids, cards, rng, manual_coin=False)
            stress_steps += stats["steps"]
            stress_n += 1
        raw = env.select(agents[env.acting_player()](raw))
    no_end_rss, peak = rss_mib()
    api.search_end()
    after_end_rss, _ = rss_mib()
    env.close()

    print("\n=== Probe B2: 5 decisions WITHOUT search_end ===")
    print(f"steps accumulated: {stress_steps}  RSS: {no_end_rss:.1f} MiB  "
          f"peak: {peak:.1f} MiB  after search_end: {after_end_rss:.1f} MiB")
    if stress_steps:
        print(f"~KiB per node (allocated states / steps): "
              f"{(no_end_rss - base_rss) * 1024 / max(1, stress_steps + tot_steps):.1f} "
              f"(upper bound; pool never shrinks)")

    print("\n=== Probe C: coin nodes inside my-turn rollouts ===")
    print(f"coin nodes by mode: {dict(coin_by_mode)}  steps by mode: {dict(steps_by_mode)}")
    print(f"rollout contexts (top 12): {all_contexts.most_common(12)}")


if __name__ == "__main__":
    main()
