"""Behavior-cloning policy (SHIPPED, pure stdlib).

Wraps the greedy pilot: for the contexts a scorer was trained on (the keys of
the weights payload) it ranks options by the learned linear score and takes the
top-k; for every other context — an unseen context, a face-down prize pick, or
any feature/scoring failure — it defers to the inherited
:class:`~ptcg_ai.decision.greedy.GreedyPolicy` handler, which the offline gate
showed already matches the teacher 88–100% there. ``SafePolicy`` remains the
crash backstop.

The deck-specific vocabularies + damage math live in the
:class:`~ptcg_ai.imitation.deck_profiles.DeckProfile` named by the payload's
``profile`` field (default ``TR_650`` so M7's ``bc_650_v1.json`` still loads).

No numpy, honoring the stdlib-only Kaggle submission contract. For linear and MLP
contexts that means a plain Python dot product, microseconds per option. A context may
instead declare ``kind: setxf2_ensemble`` (M37), a SET model scored by
:mod:`ptcg_ai.imitation.setnet` over all options at once — orders of magnitude dearer
(~1 s per decision), which is why those contexts run under the per-episode time guard
below and why ``on_episode_start`` exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.greedy import GreedyPolicy
from ptcg_ai.imitation import features as F
from ptcg_ai.imitation import setnet
from ptcg_ai.imitation.deck_profiles import DeckProfile, get_profile
from ptcg_ai.observation.models import (
    OptionKind,
    ParsedObservation,
    ParsedPlayer,
    ParsedSelect,
    SelectContextKind,
)
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState


#: Set-level scalars appended to the pooled option vector for a count head:
#: (n_options, maxCount), both scaled. They are NOT in the per-option features —
#: every option shares the state blocks, so "how many are on offer" is invisible
#: to the ranking featurizer and has to be supplied here.
_COUNT_EXTRA = 2


def load_weights(source: "str | Path | Mapping[str, Any]") -> dict[str, Any]:
    """Load a BC weights payload from a JSON file path or an in-memory mapping."""
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    profile = get_profile(payload.get("profile", "TR_650"))
    if int(payload.get("feature_dim", -1)) != profile.feature_dim:
        raise ValueError(
            f"weights feature_dim={payload.get('feature_dim')} != profile "
            f"{profile.name} dim {profile.feature_dim}; featurizer/weights out of sync"
        )
    for ctx, spec in payload.get("contexts", {}).items():
        _validate_spec(ctx, spec, profile.feature_dim)
    for ctx, head in (payload.get("count_heads") or {}).items():
        ctx_spec = payload.get("contexts", {}).get(ctx)
        ctx_profile = (
            get_profile(ctx_spec["profile"])
            if isinstance(ctx_spec, Mapping) and ctx_spec.get("profile")
            else profile
        )
        _validate_count_head(ctx, head, ctx_profile.feature_dim)
    return payload


def _validate_count_head(ctx: str, head: Any, dim: int) -> None:
    """A count head predicts HOW MANY options to take, not which ones (M42).

    Its input is the option matrix pooled to one vector plus two set-level scalars
    (see :meth:`ImitationPolicy._count_from_head`), so its width is the context's
    profile dim + ``_COUNT_EXTRA``.
    """
    if not isinstance(head, Mapping):
        raise ValueError(f"count_head {ctx!r} is not a mapping")
    classes = head.get("classes")
    W = head.get("W")
    b = head.get("b")
    want = dim + _COUNT_EXTRA
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"count_head {ctx!r} has no classes")
    if not isinstance(W, list) or len(W) != len(classes):
        raise ValueError(f"count_head {ctx!r} W has {len(W) if isinstance(W, list) else None} "
                         f"rows, expected {len(classes)}")
    if not isinstance(b, list) or len(b) != len(classes):
        raise ValueError(f"count_head {ctx!r} b length != n classes")
    for i, row in enumerate(W):
        if not isinstance(row, list) or len(row) != want:
            raise ValueError(
                f"count_head {ctx!r} row {i} width {len(row) if isinstance(row, list) else None}"
                f" != profile dim {dim} + {_COUNT_EXTRA}")


def _validate_spec(ctx: str, spec: Any, dim: int) -> None:
    """A context scorer is a flat linear vector, an MLP dict, an ensemble of MLPs,
    or an ``mlp_switch`` (M32 Strategy B: a Grimmsnarl-routed specialist head).

    Any dict spec may carry its own ``profile``, in which case it is validated
    (and later scored) against THAT profile's dim instead of the payload's — see
    ``_ctx_profiles`` in :class:`ImitationPolicy`.
    """
    if isinstance(spec, list):
        if len(spec) != dim:
            raise ValueError(f"context {ctx!r} linear length {len(spec)} != profile dim {dim}")
        return
    kind = spec.get("kind") if isinstance(spec, Mapping) else None
    if isinstance(spec, Mapping) and spec.get("profile"):
        dim = get_profile(spec["profile"]).feature_dim
    if kind == "linear":
        w = spec.get("w")
        if not isinstance(w, list) or len(w) != dim:
            raise ValueError(
                f"context {ctx!r} linear 'w' length {len(w) if isinstance(w, list) else None}"
                f" != profile dim {dim}")
        return
    if kind == "setxf2_ensemble":
        # M37: a SET model — every option's score depends on the others, so it is scored
        # through `_score_options`, never `_score`. `fallback` is the cheap per-option scorer
        # the time guard drops to; it rides inside the spec because `_decide` looks
        # contexts up by SelectContextKind name and a sibling key would never be found.
        setnet.validate_spec(ctx, spec, dim)
        if spec.get("fallback") is not None:
            _validate_spec(ctx, spec["fallback"], dim)
        return
    if kind == "mlp":
        h = len(spec["b1"])
        if len(spec["W1"]) != dim or len(spec["W1"][0]) != h or len(spec["w2"]) != h:
            raise ValueError(f"context {ctx!r} MLP shapes inconsistent with dim {dim}/h {h}")
    elif kind == "mlp_ensemble":
        for m in spec["members"]:
            _validate_spec(ctx, m, dim)
    elif kind == "mlp_switch":
        # ``general`` scores against the payload's own profile (dim, e.g. ALAKAZAM
        # 658 -- byte-identical to the un-switched model); ``specialist`` scores
        # against its OWN named profile (a different dim, e.g. ALAKAZAM_SPECIALIST
        # 697), never touching the general weights.
        specialist_profile_name = spec.get("specialist_profile")
        detector_ids = spec.get("detector_ids")
        if not specialist_profile_name or not detector_ids:
            raise ValueError(f"context {ctx!r} mlp_switch missing specialist_profile/detector_ids")
        specialist_dim = get_profile(specialist_profile_name).feature_dim
        _validate_spec(ctx, spec["general"], dim)
        _validate_spec(ctx, spec["specialist"], specialist_dim)
    else:
        raise ValueError(f"context {ctx!r} unknown scorer spec {kind!r}")


def _dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def _mlp_forward(spec: Mapping[str, Any], x: list[float]) -> float:
    """Pure-stdlib score of one option: w2·relu(W1·x + b1) + b2.

    ``W1`` is stored dim×h; the featurizer's option vectors are sparse (one-hot
    blocks are mostly zero), so we accumulate only over nonzero inputs — this is
    what keeps the forward cheap without numpy on the Kaggle contract."""
    W1, b1, w2 = spec["W1"], spec["b1"], spec["w2"]
    h = len(b1)
    hid = list(b1)
    for i, xi in enumerate(x):
        if xi == 0.0:
            continue
        row = W1[i]
        for j in range(h):
            hid[j] += row[j] * xi
    s = float(spec["b2"])
    for j in range(h):
        a = hid[j]
        if a > 0.0:
            s += w2[j] * a
    return s


def _score(spec: Any, x: list[float]) -> float:
    """Score one option under a linear / MLP / MLP-ensemble context spec."""
    if isinstance(spec, list):
        return _dot(spec, x)
    kind = spec.get("kind")
    if kind == "linear":
        return _dot(spec["w"], x)
    if kind == "mlp":
        return _mlp_forward(spec, x)
    if kind == "mlp_ensemble":
        members = spec["members"]
        return sum(_mlp_forward(m, x) for m in members) / len(members)
    raise ValueError(f"unknown score spec {kind!r}")


class ImitationPolicy(GreedyPolicy):
    """Rung 6: learned per-context scorer over the options, greedy fallback."""

    name = "imitation"

    def __init__(
        self,
        weights: "str | Path | Mapping[str, Any]",
        deck: list[int] | None = None,
        recover_rule: "Mapping[str, int] | None" = None,
        **greedy_kwargs: Any,
    ) -> None:
        super().__init__(deck=deck, **greedy_kwargs)
        payload = load_weights(weights)
        self._profile: DeckProfile = get_profile(payload.get("profile", "TR_650"))
        # A context scorer is a flat linear vector (v1) OR an MLP / MLP-ensemble
        # spec dict (v2). Kept as-is; `_score` dispatches on shape.
        self._weights: dict[str, Any] = dict(payload["contexts"])
        # Optional deterministic "recover the clock" pre-emption (M27). Default OFF:
        # the ctor arg wins, else the weights payload may carry it, else None — so
        # every existing agent (no field) is byte-for-byte unchanged. When active it
        # fires ONLY when the win-condition Pokémon is gone from play+hand yet sits in
        # discard and a recover Item is legal — a case the frozen scorer under-plays
        # because a bolt-on card has no learned weight (m27_findings.md).
        rule = recover_rule if recover_rule is not None else payload.get("recover_rule")
        self._recover_id: int | None = int(rule["recover_id"]) if rule else None
        self._clock_id: int | None = int(rule["clock_id"]) if rule else None
        self._recover_used = 0
        # M47: deterministic INDEX rules for contexts whose options carry no card
        # identity, so no ranking model can separate them at any capacity.
        # `DRAW_COUNT`'s options are bare `{"number": k, "type": 0}` -- the featurizer
        # produces the same vector for every one of them, which is the M34 bottleneck in
        # its most extreme form. But there is nothing to learn: audited over Yushin's
        # 2330 games, he picks the LAST option 488 of 488 times, and the shipped agent,
        # having no model for the context, falls to `GreedyPolicy._safe_default` and
        # picks option 0 every single time -- a 100% systematic divergence.
        # Absent key ⇒ every existing agent is byte-for-byte unchanged (the M27
        # `recover_rule` / M42 `count_heads` pattern).
        self._index_rules: dict[str, str] = {}
        for ctx_name, rule_name in (payload.get("index_rules") or {}).items():
            if rule_name not in ("first", "last"):
                raise ValueError(
                    f"index_rule {ctx_name!r}: {rule_name!r} is not 'first' or 'last'")
            self._index_rules[str(ctx_name)] = rule_name
        self._index_rule_used = 0
        self._bc_failures = 0
        self._bc_used = 0
        # M32 Strategy B: any context whose spec is an "mlp_switch" gets its
        # specialist DeckProfile resolved once here (never re-looked-up per
        # decision). Contexts without a switch never touch this dict, so every
        # existing agent is unaffected.
        self._switch_profiles: dict[str, DeckProfile] = {}
        self._route_used = 0
        # M34: a context may declare its own DeckProfile. Generalises the
        # per-context profile resolution mlp_switch already did, so ONE context
        # can be re-featurized (e.g. TO_HAND under ALAKAZAM_FETCH, dim 750) while
        # every other context — notably MAIN's proven MLP ensemble — keeps
        # scoring under the payload's profile, byte-identical to before.
        self._ctx_profiles: dict[str, DeckProfile] = {}
        # M42: contexts whose PICK COUNT is predicted instead of taken at the cap.
        # `_top_k` implements the teacher's verified rule "take min(maxCount, n)",
        # which is right for every context measured except SETUP_BENCH_POKEMON, where
        # Yushin declines on 37.4% of decisions and the champion declined 0 of 45 on
        # its own ladder replays — a divergence no RANKING model can express at any
        # capacity. Absent key ⇒ every existing agent is byte-for-byte unchanged.
        self._count_heads: dict[str, Mapping[str, Any]] = dict(payload.get("count_heads") or {})
        self._count_used = 0
        # M37: contexts scored by a SET model. `prepare_spec` splits every weight matrix
        # into rows once, here, so the per-decision hot path never re-slices (1.18x).
        self._set_specs: dict[str, dict[str, Any]] = {}
        for ctx, spec in self._weights.items():
            if isinstance(spec, Mapping) and spec.get("kind") == "mlp_switch":
                self._switch_profiles[ctx] = get_profile(spec["specialist_profile"])
            if isinstance(spec, Mapping) and spec.get("profile"):
                self._ctx_profiles[ctx] = get_profile(spec["profile"])
            if isinstance(spec, Mapping) and spec.get("kind") == "setxf2_ensemble":
                self._set_specs[ctx] = setnet.prepare_spec(spec)
        # --- M37 time guard ------------------------------------------------------
        # The set transformer costs ~2 s/decision against a 600 s per-agent-per-episode
        # budget. The MEAN is fine (~36 s/game at k=3) but the tail is not: ~0.2% of
        # Yushin's games run long enough to blow the budget, and that fraction is
        # measured on a dev laptop -- Kaggle is 2 vCPU, so it could be several times
        # worse. A timeout there would show up as unexplained ladder losses, never as a
        # visible error, so we degrade on a clock instead of hoping.
        self._budget_soft = float(payload.get("budget_soft_s", 240.0))
        self._budget_hard = float(payload.get("budget_hard_s", 400.0))
        self._elapsed = 0.0
        self._last_turn = -1
        self._budget_degraded = 0

    def on_episode_start(self) -> None:
        """Reset the per-episode time budget.

        REQUIRED, not cosmetic: the Kaggle entrypoint builds ONE ImitationPolicy at
        import and reuses it for every episode, so an un-reset accumulator would leak
        across games and degrade a fresh game because an earlier one ran long. Called
        from the entrypoint's ``select is None`` branch (the deck request, which the
        engine sends exactly once per episode); ``_decide`` also self-heals on a turn
        regression for harnesses that never issue that call, such as BattleRunner.
        """
        self._elapsed = 0.0
        self._last_turn = -1

    def on_battle_start(self) -> None:
        """Local-harness episode boundary. BattleRunner calls this; Kaggle never does.

        Wiring both hooks keeps the local gauntlet measuring the behaviour we actually
        ship, instead of leaning on the turn-regression backstop in one place and the
        explicit hook in the other. Mirrors SearchPolicy, which resets its clock here too.
        """
        super().on_battle_start()
        self.on_episode_start()

    @staticmethod
    def _count_id(player: ParsedPlayer, card_id: int, *, in_hand: bool) -> int:
        if in_hand:
            hand = player.hand or ()
            return sum(1 for c in hand if c.id == card_id)
        n = sum(1 for p in player.active if p is not None and p.id == card_id)
        return n + sum(1 for p in player.bench if p.id == card_id)

    def _forced_recovery(
        self, select: ParsedSelect, obs: ParsedObservation, gs: GameState
    ) -> int | None:
        """Index of a 'play the recover Item' option to force, or None.

        Fires only in MAIN, only when the clock Pokémon is absent from play AND hand
        yet present in discard (the win condition was answered and is recoverable), and
        the recover Item is a legal play right now. Narrow by design: when a clock is
        still available the learned policy runs untouched, so normal play is unchanged.
        """
        if self._recover_id is None or self._clock_id is None:
            return None
        if select.context is not SelectContextKind.MAIN:
            return None
        me = gs.me
        if me is None:
            return None
        if self._count_id(me, self._clock_id, in_hand=False) > 0:
            return None
        if self._count_id(me, self._clock_id, in_hand=True) > 0:
            return None
        if not any(c.id == self._clock_id for c in me.discard):
            return None
        for i, opt in enumerate(select.option):
            if opt.type not in (OptionKind.PLAY, OptionKind.CARD):
                continue
            try:
                if resolve_option(opt, obs).card_id == self._recover_id:
                    return i
            except Exception:  # noqa: BLE001 — never let the pre-empt crash a game
                continue
        return None

    def _score_options(self, ctx: str, spec: Any, X: list[list[float]]) -> list[float]:
        """Score ALL options of one decision.

        Set models cannot use ``_score``: its signature is one option in, one float out,
        and "this card is the best of the ones I hold" is inexpressible that way at any
        depth. Contexts without a set model keep the exact per-option loop, so their
        behaviour is byte-identical to before this method existed.
        """
        prepared = self._set_specs.get(ctx)
        if prepared is None:
            return [_score(spec, x) for x in X]
        if self._elapsed >= self._budget_hard:
            fallback = spec.get("fallback")
            if fallback is not None:
                self._budget_degraded += 1
                return [_score(fallback, x) for x in X]
        k = 1 if self._elapsed >= self._budget_soft else None
        if k is not None:
            self._budget_degraded += 1
        # perf_counter, not monotonic: it is the documented highest-resolution clock
        # (100 ns) and is monotonic too. `monotonic` is coarse enough on some platforms
        # that fast decisions accumulate as exactly 0.0, which would leave the guard
        # blind until a single decision happened to exceed a whole tick.
        started = time.perf_counter()
        scores = setnet.score_set(prepared, X, k=k)
        self._elapsed += time.perf_counter() - started
        return scores

    @staticmethod
    def _rank_order(scores: list[float]) -> list[int]:
        """All option indices ranked score-desc, deterministic lowest-index tie-break."""
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    @staticmethod
    def _top_k(select: ParsedSelect, order: list[int]) -> int:
        """Teacher's verified rule: min(maxCount, n), floored at minCount."""
        n = len(order)
        k = min(select.maxCount, n)
        return max(k, min(select.minCount, n))

    def _count_from_head(
        self, head: Mapping[str, Any], select: ParsedSelect, X: list[list[float]]
    ) -> int:
        """Predicted pick count, clamped to the legal band [minCount, min(maxCount, n)].

        Input is the MEAN of the option vectors (all options share the state blocks, so
        pooling preserves the state exactly and averages the card identities on offer)
        plus the two set-level scalars the per-option featurizer cannot see. X is already
        computed for the ranking, so this costs one pooling pass, not a re-featurize.

        The clamp is load-bearing: a mispredicting head must never be able to emit an
        illegal count. `_legalize` in GreedyPolicy is still the final net.
        """
        n = len(X)
        lo = min(max(select.minCount, 0), n)
        hi = min(select.maxCount, n)
        if hi <= lo or not X:
            return max(lo, 0)
        dim = len(X[0])
        pooled = [sum(row[j] for row in X) / n for j in range(dim)]
        pooled.append(n / 4.0)
        pooled.append(select.maxCount / 4.0)
        classes = head["classes"]
        best_i, best_s = 0, None
        for i, (row, bias) in enumerate(zip(head["W"], head["b"])):
            s = _dot(row, pooled) + bias
            if best_s is None or s > best_s:
                best_i, best_s = i, s
        k = int(classes[best_i])
        self._count_used += 1
        return max(lo, min(hi, k))

    @staticmethod
    def _switch_active(gs: GameState, detector_ids: set[int]) -> bool:
        """True if any of the opponent's in-play Pokémon (active or bench) match
        the switch's detector ids -- a transient, per-decision board check (never
        an "ever seen" flag), so routing turns back off the moment the trigger
        Pokémon leaves play."""
        opp = gs.opponent
        if opp is None:
            return False
        for p in list(opp.active) + list(opp.bench):
            if p is not None and p.id in detector_ids:
                return True
        return False

    def _decide_switch(
        self,
        spec: Mapping[str, Any],
        ctx: str,
        select: ParsedSelect,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase,
    ) -> tuple[list[int], bool]:
        """Score under an ``mlp_switch`` context: route to the specialist ensemble
        (its OWN profile/dim) when the detector fires, else the general ensemble
        under ``self._profile`` -- unchanged from the non-switch path, so the
        general model's play is byte-identical to before this context existed."""
        detector_ids = set(spec["detector_ids"])
        use_specialist = self._switch_active(gs, detector_ids)
        profile = self._switch_profiles[ctx] if use_specialist else self._profile
        active_spec = spec["specialist"] if use_specialist else spec["general"]
        X = F.featurize_decision(profile, select, obs, gs, cards)
        scores = self._score_options(ctx, active_spec, X)
        order = self._rank_order(scores)
        return order[: self._top_k(select, order)], use_specialist

    def _decide(
        self,
        select: ParsedSelect,
        obs: ParsedObservation,
        gs: GameState,
        cards: CardDatabase | None,
    ) -> list[int]:
        # M37 time-guard backstop: turn numbers only ever increase inside a game
        # (verified: 0 regressions across 2330 teacher games), so a drop means a new
        # episode began without the entrypoint's on_episode_start() -- which is exactly
        # what happens under BattleRunner in the local gauntlet.
        # Deliberately defensive: this runs BEFORE the try/except that guards scoring, on
        # every single decision, so an unexpected `turn` here would escape as a crash
        # rather than degrade to greedy.
        if self._set_specs and obs.current is not None:
            try:
                turn = int(obs.current.turn)
                if turn < self._last_turn:
                    self.on_episode_start()
                self._last_turn = turn
            except (TypeError, ValueError):
                pass
        forced = self._forced_recovery(select, obs, gs)
        if forced is not None:
            self._recover_used += 1
            self.last_note = f"forced-recovery play card={self._recover_id}"
            return [forced]
        ctx = select.context.name
        # M47 index rule. Applied only to SINGLE-pick decisions: the audited rule is
        # "which index", and extending it to multi-pick would be inventing behaviour
        # that was never measured. Falls through untouched for every other context.
        index_rule = self._index_rules.get(ctx)
        if index_rule is not None and len(select.option) > 0 and not F.is_prize_pick(select):
            n = len(select.option)
            if self._top_k(select, list(range(n))) == 1:
                idx = n - 1 if index_rule == "last" else 0
                self._index_rule_used += 1
                self.last_note = f"index-rule[{ctx}]={index_rule} -> {idx}"
                return [idx]
        w = self._weights.get(ctx)
        if (
            w is not None
            and cards is not None
            and len(select.option) > 0
            and not F.is_prize_pick(select)
        ):
            try:
                if isinstance(w, Mapping) and w.get("kind") == "mlp_switch":
                    chosen, routed = self._decide_switch(w, ctx, select, obs, gs, cards)
                    self._route_used += 1 if routed else 0
                    self._bc_used += 1
                    self.last_note = f"bc[{ctx}] switch=specialist k={len(chosen)}" if routed \
                        else f"bc[{ctx}] switch=general k={len(chosen)}"
                    return chosen
                profile = self._ctx_profiles.get(ctx, self._profile)
                X = F.featurize_decision(profile, select, obs, gs, cards)
                scores = self._score_options(ctx, w, X)
                order = self._rank_order(scores)
                head = self._count_heads.get(ctx)
                k = self._count_from_head(head, select, X) if head is not None \
                    else self._top_k(select, order)
                chosen = order[:k]
                self._bc_used += 1
                margin = scores[order[0]] - (scores[order[1]] if len(order) > 1 else scores[order[0]])
                self.last_note = f"bc[{ctx}] k={len(chosen)} margin={margin:.2f}"
                return chosen
            except Exception as exc:  # noqa: BLE001 — never let scoring crash a game
                self._bc_failures += 1
                self.last_note = f"bc[{ctx}] failed ({exc!r}); greedy fallback"
        return super()._decide(select, obs, gs, cards)
