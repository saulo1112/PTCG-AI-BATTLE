"""BC-guided determinized search (M10 Phase 2, dev/experimental — not yet shipped).

The M6 rung-5 search lost (0.44) with a hand-tuned linear leaf V and greedy as
both seed and rollout. This variant keeps the exact same determinized-search
plumbing (`SearchPolicy`) but swaps the two pieces the M6/M9 evidence says matter:

  - seed + fallback policy = the champion BC clone (`ImitationPolicy`), not greedy,
    so search only ever refines our best agent and falls back to it, never to
    greedy. `greedy_bias` (renamed role here: "trust BC unless clearly beaten")
    guards against a weak V regressing below BC — the M6 lesson.
  - leaf evaluator = the learned logistic V (`data/models/v_650.json`, val AUC
    0.796 vs the hand-tuned V's 0.770), scored from the FIXED root seat.

Rollouts stay greedy (fast) — they only need to finish developing our own turn to
reach the leaf board the V judges. Everything else (SearchSession hygiene,
determinizer, fast paths, per-decision cap, greedy-on-error) is inherited.

Pure-stdlib at inference (json weights + dot/sigmoid) so it can ship if it gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import ModuleType

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.evaluator import _board_energy, _deckout_risk
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.decision.search import SearchConfig, SearchPolicy
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import ParsedObservation, SelectContextKind
from ptcg_ai.state.game_state import GameState


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, x))


def value_features(obs: ParsedObservation, cards: CardDatabase | None, root_index: int) -> list[float] | None:
    """The 7 antisymmetric evaluator features, from a FIXED root seat.

    MUST stay identical to scratchpad/train_value.py's `features()` (which fits the
    v_650.json weights); training obs are always from the teacher's own seat so
    root_index there == yourIndex, and build_for(obs, root) == build(obs)."""
    state = obs.current
    if state is None:
        return None
    gs = GameState.build_for(obs, cards, root_index)
    me, opp = gs.me, gs.opponent
    if me is None or opp is None:
        return None
    prize = (gs.opp_prizes_left - gs.my_prizes_left) / 6.0
    reserve = _clip1((gs.reserve_attackers(me) - gs.reserve_attackers(opp)) / 5.0)
    survival = _clip1((gs.my_bench_count - len(opp.bench)) / 5.0)
    on_them = on_me = 0.0
    if gs.opp_active is not None and gs.opp_active.hp > 0:
        on_them = min(1.0, gs.max_threat(gs.my_active, gs.opp_active, extra_energy=1) / gs.opp_active.hp)
    if gs.my_active is not None and gs.my_active.hp > 0:
        on_me = min(1.0, gs.max_threat(gs.opp_active, gs.my_active, extra_energy=1) / gs.my_active.hp)
    threat = on_them - on_me
    energy = _clip1((_board_energy(me) - _board_energy(opp)) / 10.0)
    hand = _clip1((me.handCount - opp.handCount) / 10.0)
    deck_out = _deckout_risk(opp.deckCount) - _deckout_risk(me.deckCount)
    return [prize, threat, reserve, survival, energy, hand, deck_out]


class LearnedEvaluator:
    """Leaf V from the learned logistic weights, mapped to [-1, 1] like Evaluator.

    Terminal short-circuit (result field) is preserved so a won leaf scores +1."""

    #: How many features `value_features` produces. Subclasses that use a different
    #: feature set (M17's base-7 + prose_threat) override it.
    n_features = 7

    def __init__(self, model: dict, cards: CardDatabase | None = None) -> None:
        self._mean = model["mean"]
        self._std = model["std"]
        self._w = model["weights"]
        self._b = float(model["bias"])
        self._cards = cards
        # `value()` scores with zip(w, feats, mean, std), and zip STOPS AT THE SHORTEST
        # input without raising. A critic payload whose width does not match the feature
        # function would therefore be silently truncated -- the surviving weights would be
        # applied to the wrong features and the agent would play on a corrupted evaluator
        # with no error anywhere. That is the failure shape that cost submission 55203764
        # (M37 shipped the wrong deck; every structural check passed). The risk is live:
        # `v_alakazam_v2.json` carries 10 features, `value_features` returns 7, and M17's
        # `value_features_v2_rooted` returns 8.
        widths = {len(self._w), len(self._mean), len(self._std)}
        if len(widths) != 1:
            raise ValueError(
                f"critic payload is inconsistent: weights={len(self._w)}, "
                f"mean={len(self._mean)}, std={len(self._std)}")
        if widths.pop() != self.n_features:
            raise ValueError(
                f"{type(self).__name__} scores {self.n_features} features but the critic "
                f"payload has {len(self._w)} weights"
                + (f" (features={list(model['features'])})" if model.get("features") else "")
                + ". Refit the critic on the matching feature set, or use the evaluator "
                  "subclass that produces this width -- do NOT rely on zip() truncating.")

    @classmethod
    def from_path(cls, path: str | Path, cards: CardDatabase | None = None) -> "LearnedEvaluator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), cards)

    def value(self, obs: ParsedObservation, root_index: int) -> float:
        state = obs.current
        if state is None:
            return 0.0
        if state.result != -1:  # terminal short-circuit (matches Evaluator)
            if state.result == root_index:
                return 1.0
            if state.result == 1 - root_index:
                return -1.0
            return 0.0
        feats = value_features(obs, self._cards, root_index)
        if feats is None:
            return 0.0
        z = self._b + sum(
            w * (f - m) / s for w, f, m, s in zip(self._w, feats, self._mean, self._std)
        )
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return 2.0 * p - 1.0  # P(win)->[-1,1] value scale


class ImitationSearchPolicy(SearchPolicy):
    """SearchPolicy with a BC seed/fallback and a learned leaf V (greedy rollouts)."""

    name = "imitation_search"

    def __init__(
        self,
        weights,
        v_model: dict | str | Path,
        deck: list[int] | None = None,
        cards: CardDatabase | None = None,
        api: ModuleType | None = None,
        search_cfg: SearchConfig | None = None,
        rng_seed: int | None = None,
        bc_rollout: bool = True,
    ) -> None:
        super().__init__(deck=deck, cards=cards, api=api, search_cfg=search_cfg, rng_seed=rng_seed)
        # seed + fallback + legalize path = the champion BC clone
        self._greedy = ImitationPolicy(weights, deck=deck)
        # rollout policy: BC (reflects how WE actually finish the turn — greedy
        # pilots this deck badly, so a greedy rollout corrupts the leaf board the
        # V judges) or greedy (faster). BC is a separate instance from the seed.
        self._rollout_pol = ImitationPolicy(weights, deck=deck) if bc_rollout else GreedyPolicy(deck=deck)
        # leaf eval = learned V
        model = v_model if isinstance(v_model, dict) else json.loads(Path(v_model).read_text(encoding="utf-8"))
        self._evaluator = LearnedEvaluator(model, cards)

    def on_battle_start(self) -> None:
        super().on_battle_start()
        self._rollout_pol.on_battle_start()

    def choose(self, ctx: DecisionContext) -> list[int]:
        """M48: search engages ONLY on MAIN decisions; every other context defers
        straight to the champion (``self._greedy``), untouched.

        ``SearchPolicy.choose`` (the base) is context-agnostic by design -- it was built
        and tuned entirely against the M10-era TR-650 champion, a single MAIN-only
        linear/MLP scorer with no other context models at all, so there was nothing to
        override outside MAIN. That stopped being true after M42/M43: this project's
        deployed champion carries FOUR specialized, individually-verified context heads
        (ACTIVATE, SETUP_BENCH_POKEMON+count, SWITCH, TO_HAND's MLP ensemble) that are
        the single largest measured gain in the project's history (+0.096 to +0.117 in
        this exact n=600 mirror arena). Left unguarded, `_search_choose` generates
        alternative candidates around whatever `self._greedy` returns for ANY context and
        can override it whenever `self._evaluator` -- a 7-feature MAIN-shaped board
        evaluator, never fit or validated on "which card to fetch" or "which bench slot
        to promote" -- scores an alternative higher by more than `greedy_bias`. Measured
        directly (`m48_search_arena.py sweep`, n=100 per arm): score against the
        champion rose monotonically as less search override was allowed (gb=0.02: 0.175,
        gb=0.05: 0.280) -- exactly the signature of search silently discarding the
        specialized heads on every non-MAIN decision, not a MAIN-specific evaluator
        problem alone.
        """
        select = ctx.observation.select
        if select is None or select.context is not SelectContextKind.MAIN:
            return self._greedy.choose(ctx)
        return super().choose(ctx)

    def _greedy_action(self, node) -> list[int]:
        ctx = DecisionContext(raw=node.raw, observation=node.parsed, cards=self._cards)
        return self._rollout_pol.choose(ctx)
