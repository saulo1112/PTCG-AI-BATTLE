"""M37: the per-episode compute guard around the set-transformer MAIN scorer (SDK-free).

WHY IT EXISTS. The set transformer costs ~1.3 s per MAIN decision against a 600 s
per-agent-per-episode budget. The mean is comfortable (~36 s/game at k=3) but the tail
is not: measured on the teacher's 2330 games, ~0.2% run long enough to exhaust the
budget — and that figure comes from a dev laptop, while Kaggle gives 2 vCPU. A timeout
there produces unexplained ladder losses, never a visible error, so the policy watches a
clock and sheds cost instead of hoping.

These tests pin the three things that can silently break it:
  * degradation actually happens, in both stages (fewer members, then the cheap scorer);
  * the accumulator is RESET per episode — the Kaggle entrypoint builds one policy at
    import and reuses it for every game, so a leak would degrade a fresh game because an
    earlier one ran long;
  * both reset paths work: the explicit hook AND the turn-regression backstop that
    covers harnesses (BattleRunner) which never issue the deck request.
"""

from __future__ import annotations

from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.imitation import deck_profiles as DP
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import CardKind, EnergyKind, OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser

PSYCHIC = int(EnergyKind.PSYCHIC)
MY_ACTIVE = 743
OPP = 9999
ALAKAZAM = DP.ALAKAZAM
D, LAYERS, HEADS = 8, 1, 2       # tiny but structurally identical to the shipped model


def _card(cid, hp=120):
    return {
        "cardId": cid, "name": f"c{cid}", "cardType": int(CardKind.POKEMON), "retreatCost": 1,
        "hp": hp, "weakness": None, "resistance": None, "energyType": PSYCHIC,
        "basic": True, "stage1": False, "stage2": False, "ex": False, "megaEx": False,
        "tera": False, "aceSpec": False, "evolvesFrom": None, "skills": [], "attacks": [],
    }


def _cards() -> CardDatabase:
    return CardDatabase.from_records(
        {"cards": [_card(MY_ACTIVE, 130), _card(OPP, 150)], "attacks": []})


def _pokemon(cid, hp=150):
    return {"id": cid, "serial": cid * 10, "hp": hp, "maxHp": hp, "appearThisTurn": False,
            "energies": [PSYCHIC], "energyCards": [], "preEvolution": [], "tools": []}


def _obs(turn: int = 4) -> dict:
    me = {
        "active": [_pokemon(MY_ACTIVE, 130)], "bench": [], "benchMax": 5, "deckCount": 40,
        "discard": [], "prize": [None] * 6, "handCount": 5,
        "hand": [{"id": 1, "serial": 900 + i, "playerIndex": 0} for i in range(5)],
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False,
        "confused": False,
    }
    opp = {
        "active": [_pokemon(OPP)], "bench": [], "benchMax": 5, "deckCount": 40,
        "discard": [], "prize": [None] * 6, "handCount": 4, "hand": None,
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False,
        "confused": False,
    }
    options = [{"type": int(OptionKind.PLAY), "index": 0}, {"type": int(OptionKind.END)}]
    return {
        "select": {"type": 0, "context": int(SelectContextKind.MAIN), "minCount": 1,
                   "maxCount": 1, "remainDamageCounter": 0, "remainEnergyCost": 0,
                   "option": options, "deck": None, "contextCard": None, "effect": None},
        "logs": [], "search_begin_input": "AAAA",
        "current": {"turn": turn, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                    "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [], "looking": None,
                    "players": [me, opp]},
    }


def _member(bias: float) -> dict:
    """A constant-output member: every weight zero, so the score is exactly `bias`.

    That makes the ensemble mean trivially predictable, which is the point — these tests
    are about WHICH members ran, not about what the network computes.
    """
    dim, opt_end, st = ALAKAZAM.feature_dim, 69, 34
    z = lambda n: [0.0] * n  # noqa: E731
    return {
        "opt_w": z(D * opt_end), "opt_b": z(D),
        "state0_w": z(D * st), "state0_b": z(D),
        "state2_w": z(D * D), "state2_b": z(D),
        "film_w": z(2 * D * D), "film_b": z(2 * D),
        "lnout_w": z(D), "lnout_b": z(D), "head_w": z(D), "head_b": bias,
        "blocks": [{
            "ln1_w": z(D), "ln1_b": z(D), "ln2_w": z(D), "ln2_b": z(D),
            "in_proj_weight": z(3 * D * D), "in_proj_bias": z(3 * D),
            "out_proj_weight": z(D * D), "out_proj_bias": z(D),
            "ff0_w": z(4 * D * D), "ff0_b": z(4 * D),
            "ff2_w": z(D * 4 * D), "ff2_b": z(D),
        } for _ in range(LAYERS)],
    }
    # (dim is unused beyond validation, kept for readability)


def _payload(*, with_fallback: bool = True) -> dict:
    dim = ALAKAZAM.feature_dim
    spec = {
        "kind": "setxf2_ensemble", "d_model": D, "layers": LAYERS, "heads": HEADS,
        "opt_end": 69, "state_start": 69, "state_end": 103,
        # distinct biases so the ensemble mean reveals how many members were evaluated
        "members": [_member(1.0), _member(3.0), _member(5.0)],
    }
    if with_fallback:
        spec["fallback"] = [0.0] * dim        # a linear scorer: cheap, always scores 0
    return {"profile": ALAKAZAM.name, "feature_dim": dim,
            "budget_soft_s": 10.0, "budget_hard_s": 20.0,
            "contexts": {"MAIN": spec}}


def _decide(pol: ImitationPolicy, raw: dict, cards: CardDatabase):
    return pol.choose(DecisionContext(raw=raw, observation=ObservationParser().parse(raw),
                                      cards=cards))


def _scores(pol: ImitationPolicy, raw: dict, cards: CardDatabase) -> list[float]:
    """Score MAIN through the real policy path, so the guard is exercised."""
    from ptcg_ai.imitation import features as F
    from ptcg_ai.state.game_state import GameState
    obs = ObservationParser().parse(raw)
    gs = GameState.build(obs, cards)
    X = F.featurize_decision(ALAKAZAM, obs.select, obs, gs, cards)
    return pol._score_options("MAIN", pol._weights["MAIN"], X)


def test_full_ensemble_runs_when_under_budget() -> None:
    pol = ImitationPolicy(_payload())
    scores = _scores(pol, _obs(), _cards())
    assert scores[0] == 3.0            # mean of 1, 3, 5 -> all three members ran
    assert pol._budget_degraded == 0
    assert pol._elapsed > 0.0          # the clock is actually being fed


def test_soft_threshold_drops_to_one_member() -> None:
    pol = ImitationPolicy(_payload())
    pol._elapsed = 15.0                # past soft (10), under hard (20)
    scores = _scores(pol, _obs(), _cards())
    assert scores[0] == 1.0            # first member only
    assert pol._budget_degraded == 1


def test_hard_threshold_falls_back_to_cheap_scorer() -> None:
    pol = ImitationPolicy(_payload())
    pol._elapsed = 25.0                # past hard (20)
    before = pol._elapsed
    scores = _scores(pol, _obs(), _cards())
    assert scores[0] == 0.0            # the zero linear fallback, not the set model
    assert pol._budget_degraded == 1
    assert pol._elapsed == before      # the cheap path must not charge the budget


def test_hard_threshold_without_fallback_still_degrades_not_crashes() -> None:
    """A payload with no cheap scorer must not blow up; it clamps to one member."""
    pol = ImitationPolicy(_payload(with_fallback=False))
    pol._elapsed = 25.0
    scores = _scores(pol, _obs(), _cards())
    assert scores[0] == 1.0
    assert pol._budget_degraded == 1


def test_on_episode_start_resets_the_budget() -> None:
    pol = ImitationPolicy(_payload())
    pol._elapsed = 999.0
    pol.on_episode_start()
    assert pol._elapsed == 0.0
    assert _scores(pol, _obs(), _cards())[0] == 3.0      # full ensemble again


def test_turn_regression_resets_the_budget() -> None:
    """The backstop for harnesses that never send the deck request. Without it the
    singleton policy would carry one long game's clock into the next."""
    cards = _cards()
    pol = ImitationPolicy(_payload())
    _decide(pol, _obs(turn=30), cards)
    pol._elapsed = 999.0
    _decide(pol, _obs(turn=2), cards)                    # a new game began
    assert pol._elapsed < 999.0
    assert pol._budget_degraded == 0


def test_budget_accumulates_across_decisions_within_a_game() -> None:
    """Turns advancing must NOT reset — only a regression counts as a new episode."""
    cards = _cards()
    pol = ImitationPolicy(_payload())
    _decide(pol, _obs(turn=4), cards)
    first = pol._elapsed
    _decide(pol, _obs(turn=5), cards)
    assert pol._elapsed > first
