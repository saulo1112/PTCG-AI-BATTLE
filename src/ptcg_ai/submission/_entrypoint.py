# This file becomes `main.py` at the top level of the submission tarball.
#
# HARD CONSTRAINTS (ADR-0001, ADR-0014):
# - stdlib only; may import ONLY the bundled `ptcg_ai/` subtree (a pruned,
#   stdlib-only copy), never the dev research stack (no config/yaml/cg).
# - must never raise: a crashed agent forfeits the game. Every path is wrapped
#   and degrades to a uniform-random legal choice (inline SafePolicy).
#
# Kaggle contract: agent(obs_dict) -> list[int]
#   - initial call (obs["select"] is None): return the 60-card deck
#   - otherwise: minCount..maxCount unique indices into obs["select"]["option"]
#
# Strategy: prefer the imitation pilot (rung 6) when its weights are bundled;
# otherwise fall back to the greedy policy (rung 3); if ANYTHING fails to import
# or run, fall back to the safe-random baseline so play strength never costs
# submission safety. Tiers: imitation -> greedy -> random, checked per decision.
"""Kaggle submission entrypoint — imitation (rung 6) → greedy → safe-random."""

import json
import os
import random
import sys

_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent/"
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Kaggle loads this file via exec(code, env) with no `__file__` in env
    # (kaggle_environments.agent.get_last_callable). Module-level code must
    # never raise here — a crash at load time kills the whole agent before
    # even the safe-random fallback can run.
    _HERE = _KAGGLE_AGENT_DIR

#: True once the greedy policy + card data have loaded (checked by
#: `ptcg validate-submission` to confirm the smart agent is actually engaged,
#: not silently degraded to random).
_AGENT_READY = False
_run_greedy = None  # callable: (raw_obs_dict) -> list[int]

#: True once the imitation policy + weights have loaded (the preferred pilot).
_IMITATION_READY = False
_run_imitation = None  # callable: (raw_obs_dict) -> list[int]
#: M37: called once per episode (on the deck request) so the policy can reset its
#: per-episode compute budget. None for greedy builds and for older bundles.
_on_episode_start = None
#: M37: the live policy object, exposed ONLY so validation can read its counters.
#: `policy._decide` swallows every exception and plays greedy, so a subtly broken
#: scorer produces no error at all -- `_bc_failures`/`_bc_used` are the only evidence,
#: and the smoke test asserts on them before we ever spend a ladder slot.
_imitation_policy = None


def _agent_dir(name):
    """Locate a bundled file in the working dir or the Kaggle agent mount."""
    if os.path.exists(os.path.join(_HERE, name)):
        return os.path.join(_HERE, name)
    if os.path.exists(name):
        return name
    return _KAGGLE_AGENT_DIR + name


def _build_greedy():
    """Wire up the greedy policy; set the module globals. Never raises."""
    global _AGENT_READY, _run_greedy
    try:
        for path in (_HERE, _KAGGLE_AGENT_DIR, os.getcwd()):
            if path not in sys.path:
                sys.path.insert(0, path)
        from ptcg_ai.cards.database import CardDatabase
        from ptcg_ai.decision.base import DecisionContext
        from ptcg_ai.decision.greedy import GreedyPolicy
        from ptcg_ai.observation.parser import ObservationParser

        with open(_agent_dir("card_data.json"), "r", encoding="utf-8") as fh:
            cards = CardDatabase.from_records(json.load(fh))
        parser = ObservationParser()
        policy = GreedyPolicy()

        def run(raw):
            ctx = DecisionContext(raw=raw, observation=parser.parse(raw), cards=cards)
            return policy.choose(ctx)

        _run_greedy = run
        _AGENT_READY = True
    except Exception:
        _AGENT_READY = False
        _run_greedy = None


def _build_imitation():
    """Wire up the imitation policy IF weights are bundled. Never raises.

    Absence of ``bc_weights.json`` is the normal greedy-build case: we simply
    leave ``_IMITATION_READY`` False and let the greedy tier run.
    """
    global _IMITATION_READY, _run_imitation, _on_episode_start, _imitation_policy
    try:
        weights_file = _agent_dir("bc_weights.json")
        if not os.path.exists(weights_file):
            return  # greedy build — no imitation weights bundled
        for path in (_HERE, _KAGGLE_AGENT_DIR, os.getcwd()):
            if path not in sys.path:
                sys.path.insert(0, path)
        from ptcg_ai.cards.database import CardDatabase
        from ptcg_ai.decision.base import DecisionContext
        from ptcg_ai.imitation.policy import ImitationPolicy
        from ptcg_ai.observation.parser import ObservationParser

        with open(_agent_dir("card_data.json"), "r", encoding="utf-8") as fh:
            cards = CardDatabase.from_records(json.load(fh))
        with open(weights_file, "r", encoding="utf-8") as fh:
            weights = json.load(fh)
        parser = ObservationParser()
        policy = ImitationPolicy(weights)  # deck comes from deck.csv, not needed here

        def run(raw):
            ctx = DecisionContext(raw=raw, observation=parser.parse(raw), cards=cards)
            return policy.choose(ctx)

        def episode_start():
            # M37: the only per-episode signal the Kaggle harness gives us. The policy
            # is built ONCE at import and reused for every game, so its time budget has
            # to be told when a new one starts. getattr-guarded so a bundle carrying an
            # older policy.py still loads.
            hook = getattr(policy, "on_episode_start", None)
            if hook is not None:
                hook()

        _run_imitation = run
        _on_episode_start = episode_start
        _imitation_policy = policy
        _IMITATION_READY = True
    except Exception:
        _IMITATION_READY = False
        _run_imitation = None


_build_greedy()
_build_imitation()


_DECK = None


def _read_deck():
    """Read deck.csv from the working dir or the Kaggle agent mount."""
    with open(_agent_dir("deck.csv"), "r") as fh:
        lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    return [int(ln) for ln in lines[:60]]


def _random_legal(select):
    """A uniformly random legal selection; clamps terminal-obs quirks."""
    n = len(select["option"])
    min_count = min(int(select.get("minCount", 0)), n)
    max_count = min(int(select.get("maxCount", n)), n)
    k = random.randint(min_count, max_count)
    return random.sample(range(n), k)


def _is_legal(action, select):
    """True if `action` is a valid selection for `select`."""
    n = len(select["option"])
    if not isinstance(action, list) or len(set(action)) != len(action):
        return False
    if any(not isinstance(i, int) or not (0 <= i < n) for i in action):
        return False
    min_count = min(int(select.get("minCount", 0)), n)
    max_count = min(int(select.get("maxCount", n)), n)
    return min_count <= len(action) <= max_count


def agent(obs_dict):
    """Imitation when available and legal, then greedy, then a safe random choice."""
    global _DECK
    try:
        select = obs_dict.get("select")
        if select is None:
            # The deck request: the engine sends it exactly once, at the start of each
            # episode. It is the only per-episode boundary the harness exposes, so this
            # is where the policy's compute budget gets reset (M37).
            if _on_episode_start is not None:
                try:
                    _on_episode_start()
                except Exception:
                    pass  # a budget reset must never cost us the deck submission
            if _DECK is None:
                _DECK = _read_deck()
            return list(_DECK)

        if _IMITATION_READY and _run_imitation is not None:
            try:
                action = _run_imitation(obs_dict)
                if _is_legal(action, select):
                    return action
            except Exception:
                pass  # imitation hiccup → fall through to greedy
        if _AGENT_READY and _run_greedy is not None:
            try:
                action = _run_greedy(obs_dict)
                if _is_legal(action, select):
                    return action
            except Exception:
                pass  # greedy hiccup on one decision → fall through to random
        return _random_legal(select)
    except Exception:
        # Last-resort fallback: the most conservative legal-looking answer.
        try:
            min_count = int(obs_dict["select"].get("minCount", 0))
            return list(range(min_count))
        except Exception:
            return []
