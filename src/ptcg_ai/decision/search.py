"""Rung-5 determinized all-contexts search policy (ADR-0010).

Per decision (any context): sample ``D`` determinized worlds, and in each world
score every candidate selection by stepping it in the engine, rolling out the
rest of my turn with greedy, and evaluating the leaf with ``V`` from a fixed
root seat. Play the argmax over the world-averaged scores. Anything that can go
wrong — no SDK, low time budget, a determinization mismatch, any exception —
degrades to greedy for that decision; ``SafePolicy`` wraps the whole thing.

Composition: a :class:`SearchPolicy` *owns* a :class:`GreedyPolicy` used both as
the in-tree rollout policy and as the fallback, so search is strictly "greedy
plus lookahead" and never regresses below greedy's answer set (greedy's own
choice is always among the evaluated candidates).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from types import ModuleType

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import BasePolicy, DecisionContext
from ptcg_ai.decision.evaluator import EvalWeights, Evaluator
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.observation.models import OptionKind, ParsedObservation, ParsedSelect
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.planning.determinize import Determinizer, DeterminizeError
from ptcg_ai.planning.session import SearchSession, SimNode
from ptcg_ai.state.game_state import GameState


@dataclass(frozen=True)
class SearchConfig:
    """Search budget and shape (defaults are the M6-0-probed v0 prior).

    Lives in this shipped module (not the research-only config schema) so the
    Kaggle bundle is self-contained.
    """

    determinizations: int = 4
    max_candidates: int = 16          # single-choice options considered
    max_candidate_sets: int = 8       # multi-count selection sets considered
    rollout_step_cap: int = 40
    manual_coin: bool = False         # probed: ~0 coin nodes in my-turn rollouts
    #: Score bonus added to greedy's own choice: search overrides greedy only
    #: when another candidate beats it by more than this margin. 0 = pure V
    #: argmax; large = always greedy. Guards against a weak V regressing below
    #: greedy on decks where greedy's fixed priorities are already strong.
    greedy_bias: float = 0.0
    time_budget_s: float = 420.0      # of the ~600 s/agent pool
    min_reserve_s: float = 120.0      # below this -> pure greedy
    per_decision_cap_s: float = 6.0


#: Candidate-ordering prior for when the option cap binds. Unknown PLAY/ABILITY
#: rank high — the whole point of search is to evaluate cards the whitelist
#: cannot. END ranks last (it is never worth cutting a real action for).
_TYPE_ORDER = {
    OptionKind.ABILITY: 0,
    OptionKind.PLAY: 1,
    OptionKind.ATTACK: 2,
    OptionKind.EVOLVE: 3,
    OptionKind.ATTACH: 4,
    OptionKind.RETREAT: 5,
    OptionKind.DISCARD: 6,
    OptionKind.END: 99,
}


class SearchPolicy(BasePolicy):
    """Determinized search over the shared pipeline; greedy fallback + rollout."""

    name = "search"

    def __init__(
        self,
        deck: list[int] | None = None,
        cards: CardDatabase | None = None,
        api: ModuleType | None = None,
        evaluator: Evaluator | None = None,
        weights: EvalWeights | None = None,
        search_cfg: SearchConfig | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(deck)
        self._cards = cards
        self._api = api
        self._cfg = search_cfg or SearchConfig()
        self._evaluator = evaluator or Evaluator(weights or EvalWeights(), cards)
        self._greedy = GreedyPolicy(deck=deck)
        self._parser = ObservationParser()
        self._rng = random.Random(rng_seed)
        self._determinizer: Determinizer | None = None
        # per-battle observability + time ledger
        self._elapsed = 0.0
        self.searched = 0
        self.fallbacks = 0

    def on_battle_start(self) -> None:
        super().on_battle_start()
        self._greedy.on_battle_start()
        self._elapsed = 0.0
        self.searched = 0
        self.fallbacks = 0

    # -- entry ------------------------------------------------------------

    def choose(self, ctx: DecisionContext) -> list[int]:
        select = ctx.observation.select
        if select is None:
            raise ValueError("choose() called on a deck-submission observation")
        n = len(select.option)
        if n == 0:
            self.last_note = "empty option list; returning nothing"
            return []
        if n == 1:  # forced move — no search needed
            self.last_note = "single legal option"
            return GreedyPolicy._legalize([0], select)

        if self._api is None or self._budget_remaining(ctx) < self._cfg.min_reserve_s:
            return self._fallback(ctx, "no api / low budget")

        win = self._winning_lethal(select, ctx.observation, ctx.cards)
        if win is not None:
            self.last_note = "game-winning lethal (fast path)"
            return GreedyPolicy._legalize([win], select)

        try:
            t0 = time.perf_counter()
            chosen = self._search_choose(ctx, select)
            self._elapsed += time.perf_counter() - t0
            self.searched += 1
            return GreedyPolicy._legalize(chosen, select)
        except Exception as exc:  # noqa: BLE001 - never crash a live decision
            return self._fallback(ctx, f"search error: {type(exc).__name__}")

    def choose_deck(self, ctx: DecisionContext) -> list[int]:
        return self._greedy.choose_deck(ctx)

    # -- search -----------------------------------------------------------

    def _search_choose(self, ctx: DecisionContext, select: ParsedSelect) -> list[int]:
        obs = ctx.observation
        assert obs.current is not None
        root_index = obs.current.yourIndex
        det = self._get_determinizer()
        worlds = det.sample(obs, self._cfg.determinizations)  # DeterminizeError -> caught

        greedy_choice = self._greedy.choose(ctx)
        greedy_set = sorted(dict.fromkeys(greedy_choice))
        candidates = self._candidate_sets(select, obs, ctx.cards, greedy_choice)
        scores = [0.0] * len(candidates)

        with SearchSession(self._api, self._parser) as sess:
            for hidden in worlds:
                root = sess.begin(ctx.raw, hidden, manual_coin=self._cfg.manual_coin)
                if root.n_options != len(select.option):
                    # determinized root disagrees with the live options — cannot
                    # map candidate indices safely; abandon search for this world.
                    raise DeterminizeError("root option count mismatch")
                for ci, cand in enumerate(candidates):
                    scores[ci] += self._score_candidate(sess, root, cand, root_index)

        nworlds = max(1, len(worlds))
        avg = [s / nworlds for s in scores]
        # Trust greedy unless another candidate clearly beats it (greedy_bias).
        if self._cfg.greedy_bias:
            for ci, cand in enumerate(candidates):
                if cand == greedy_set:
                    avg[ci] += self._cfg.greedy_bias
        best = max(range(len(candidates)), key=lambda i: avg[i])
        self.last_note = (
            f"search: {len(candidates)} cands x {len(worlds)} worlds "
            f"(best {avg[best]:+.3f}, greedy={'yes' if candidates[best] == greedy_set else 'no'})"
        )
        return candidates[best]

    def _score_candidate(
        self, sess: SearchSession, root: SimNode, cand: list[int], root_index: int
    ) -> float:
        node = sess.step(root, cand)
        node = self._rollout(sess, node, root_index)
        return self._evaluator.value(node.parsed, root_index)

    def _rollout(self, sess: SearchSession, node: SimNode, root_index: int) -> SimNode:
        """Play the rest of the root player's turn with greedy; stop at the
        first opponent-owned node, terminal, or the step cap (the leaf)."""
        for _ in range(self._cfg.rollout_step_cap):
            if node.is_terminal or node.actor_index != root_index:
                break
            action = self._greedy_action(node)
            node = sess.step(node, action)
        return node

    def _greedy_action(self, node: SimNode) -> list[int]:
        ctx = DecisionContext(raw=node.raw, observation=node.parsed, cards=self._cards)
        return self._greedy.choose(ctx)

    # -- candidate generation ---------------------------------------------

    def _candidate_sets(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        cards: CardDatabase | None,
        greedy_choice: list[int],
    ) -> list[list[int]]:
        n = len(select.option)
        lo = min(max(select.minCount, 0), n)
        hi = min(select.maxCount, n)
        out: list[list[int]] = []

        def add(indices) -> None:
            s = sorted(dict.fromkeys(i for i in indices if 0 <= i < n))
            if lo <= len(s) <= hi and s not in out:
                out.append(s)

        add(greedy_choice)  # greedy's answer is always a candidate
        if hi <= 1:
            for i in self._prioritize(select):
                if len(out) >= self._cfg.max_candidates:
                    break
                add([i])
        else:
            add(list(range(lo)))          # min-count set
            add(list(range(hi)))          # max-count set
            base = greedy_choice or list(range(lo))
            for i in range(n):
                if len(out) >= self._cfg.max_candidate_sets:
                    break
                if i not in base:
                    add((base[1:] if len(base) > 1 else base) + [i])  # single swap/extend
        return out or [sorted(range(lo))]

    def _prioritize(self, select: ParsedSelect) -> list[int]:
        """Option indices ordered so the useful ones survive the candidate cap."""
        return sorted(
            range(len(select.option)),
            key=lambda i: (_TYPE_ORDER.get(select.option[i].type, 50), i),
        )

    # -- fast paths / fallback --------------------------------------------

    def _winning_lethal(
        self, select: ParsedSelect, obs: ParsedObservation, cards: CardDatabase | None
    ) -> int | None:
        """Index of a KO-math lethal that takes my LAST prize(s) — a
        game-ending move that cannot be wrong. Non-winning lethals are left to
        the search (their real value depends on what the opponent does next)."""
        if cards is None:
            return None
        gs = GameState.build(obs, cards)
        if gs.my_active is None or gs.opp_active is None:
            return None
        if gs.my_prizes_left > gs.opp_active_prize_value:
            return None
        for i, opt in enumerate(select.option):
            if opt.type is OptionKind.ATTACK and opt.attackId is not None:
                attack = cards.get_attack(opt.attackId)
                if attack is not None and gs.is_lethal(attack, gs.my_active, gs.opp_active):
                    return i
        return None

    def _fallback(self, ctx: DecisionContext, why: str) -> list[int]:
        self.fallbacks += 1
        out = self._greedy.choose(ctx)
        self.last_note = f"greedy fallback ({why}); {self._greedy.last_note}"
        return out

    def _budget_remaining(self, ctx: DecisionContext) -> float:
        rem = ctx.raw.get("remainingOverageTime")
        if isinstance(rem, (int, float)):
            return float(rem)
        return self._cfg.time_budget_s - self._elapsed

    def _get_determinizer(self) -> Determinizer:
        if self._determinizer is None:
            if self._deck is None:
                raise DeterminizeError("search policy has no deck to determinize")
            self._determinizer = Determinizer(list(self._deck), self._cards, self._rng)
        return self._determinizer
