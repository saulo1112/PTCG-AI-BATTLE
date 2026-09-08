"""SearchPolicy logic (SDK-free: a fake api over scripted node graphs)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ptcg_ai.cards.database import AttackInfo, CardDatabase, CardInfo
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.decision.evaluator import Evaluator
from ptcg_ai.decision.search import SearchConfig, SearchPolicy
from ptcg_ai.observation.models import CardKind, EnergyKind
from ptcg_ai.observation.parser import ObservationParser
from tests.conftest import make_raw_obs

parser = ObservationParser()


def _card(cid: int, **kw) -> CardInfo:
    base = dict(
        cardId=cid, name=f"c{cid}", cardType=CardKind.POKEMON, retreatCost=1,
        hp=100, weakness=None, resistance=None, energyType=EnergyKind.WATER,
        basic=True, stage1=False, stage2=False, ex=False, megaEx=False,
        tera=False, aceSpec=False, evolvesFrom=None, skills=(), attacks=(),
    )
    base.update(kw)
    return CardInfo(**base)  # type: ignore[arg-type]


def _db() -> CardDatabase:
    return CardDatabase(
        {721: _card(721, hp=60, attacks=(900,)), 800: _card(800, hp=50)},
        {900: AttackInfo(attackId=900, name="A", text="", damage=200,
                         energies=(EnergyKind.WATER,))},
    )


# -- a fake search API over a trivial scripted graph ------------------------

@dataclass
class _State:
    searchId: int
    observation: object


@dataclass
class _FakeApi:
    """Every begin -> a root at a copy of the live obs; every step -> a
    terminal node whose winner is chosen by ``winner_for`` keyed on the first
    selected index. Lets us assert argmax without the engine."""

    root_raw: dict
    winner_for: dict  # first-selected-index -> result winner index
    begins: int = 0
    steps: int = 0
    ends: int = 0

    def to_observation_class(self, raw):
        return raw

    def search_begin(self, agent_obs, *hidden, manual_coin=False):
        self.begins += 1
        return _State(1, _RootObs(self.root_raw))

    def search_step(self, search_id, select):
        self.steps += 1
        # Every step ends the game (rollout breaks immediately); the default is
        # a loss for the root, so only the mapped index scores +1.
        winner = self.winner_for.get(select[0], 1)
        return _State(search_id + 1, _TermObs(winner))

    def search_end(self):
        self.ends += 1


# cg-shaped nested dataclasses (observation_to_mapping walks dataclass FIELDS).

@dataclass
class _FState:
    result: int
    yourIndex: int = 0
    turn: int = 5
    turnActionCount: int = 0
    firstPlayer: int = 0
    supporterPlayed: bool = False
    stadiumPlayed: bool = False
    energyAttached: bool = False
    retreated: bool = False
    stadium: list = field(default_factory=list)
    looking: object = None
    players: list = field(default_factory=list)


@dataclass
class _FSelect:
    option: list
    type: int = 0
    context: int = 0
    minCount: int = 1
    maxCount: int = 1
    remainDamageCounter: int = 0
    remainEnergyCost: int = 0
    deck: object = None
    contextCard: object = None
    effect: object = None


@dataclass
class _FObs:
    select: object
    current: object
    search_begin_input: object = None


def _RootObs(raw):
    c = raw["current"]
    return _FObs(select=_FSelect(option=list(raw["select"]["option"])),
                 current=_FState(result=-1, yourIndex=c["yourIndex"]))


def _TermObs(winner):
    return _FObs(select=None, current=_FState(result=winner, yourIndex=0))


def _ctx(raw):
    return DecisionContext(raw=raw, observation=parser.parse(raw), cards=_db())


def _policy(api, cfg=None):
    return SearchPolicy(deck=[721] * 60, cards=_db(), api=api,
                        evaluator=Evaluator(cards=_db()), search_cfg=cfg or SearchConfig(),
                        rng_seed=1)


def test_single_option_no_search() -> None:
    raw = make_raw_obs(n_options=1)
    api = _FakeApi(raw, {})
    pol = _policy(api)
    out = pol.choose(_ctx(raw))
    assert out == [0]
    assert api.begins == 0  # never searched a forced move


def test_search_picks_winning_candidate() -> None:
    # 3 options; option index 1 leads to a win for the root (yourIndex 0).
    raw = make_raw_obs(n_options=3)
    # my deck must reconcile with the obs: give a deck that determinizes cleanly
    api = _FakeApi(raw, {1: 0})  # selecting index 1 -> root (player 0) wins
    pol = SearchPolicy(deck=[721] * 60, cards=_db(), api=api,
                       evaluator=Evaluator(cards=_db()), rng_seed=1)
    # patch determinizer to a no-op single world so we don't need a legal deck
    pol._determinizer = _StubDet()
    out = pol.choose(_ctx(raw))
    assert out == [1]
    assert api.ends >= 1  # session cleaned up


def test_greedy_bias_overrides_v() -> None:
    # Under pure V, index 1 wins; a large greedy_bias must force greedy's own
    # choice instead (the "trust greedy unless clearly better" guard).
    from ptcg_ai.decision.greedy import GreedyPolicy

    raw = make_raw_obs(n_options=3)
    api = _FakeApi(raw, {1: 0})
    pol = SearchPolicy(deck=[721] * 60, cards=_db(), api=api,
                       evaluator=Evaluator(cards=_db()),
                       search_cfg=SearchConfig(greedy_bias=5.0), rng_seed=1)
    pol._determinizer = _StubDet()
    greedy_pick = GreedyPolicy(deck=[721] * 60).choose(_ctx(raw))
    assert pol.choose(_ctx(raw)) == greedy_pick


def test_search_falls_back_without_api() -> None:
    raw = make_raw_obs(n_options=3)
    pol = SearchPolicy(deck=[721] * 60, cards=_db(), api=None, rng_seed=1)
    out = pol.choose(_ctx(raw))
    assert out  # greedy answered
    assert pol.fallbacks == 1


def test_search_falls_back_on_determinize_error() -> None:
    raw = make_raw_obs(n_options=3)
    api = _FakeApi(raw, {})
    pol = _policy(api)
    pol._determinizer = _RaisingDet()
    out = pol.choose(_ctx(raw))
    assert out
    assert pol.fallbacks == 1
    assert api.begins == 0  # never got to search


def test_low_budget_uses_greedy() -> None:
    raw = make_raw_obs(n_options=3)
    raw["remainingOverageTime"] = 30.0  # below min_reserve_s (120)
    api = _FakeApi(raw, {})
    pol = _policy(api)
    out = pol.choose(_ctx(raw))
    assert pol.fallbacks == 1
    assert api.begins == 0


def test_winning_lethal_fast_path() -> None:
    # active 721 with a 200-dmg attack vs opp 800 (50hp); my last prize -> win.
    raw = make_raw_obs(n_options=3)
    raw["current"]["players"][0]["prize"] = [None]  # 1 prize left
    raw["current"]["players"][0]["active"][0]["id"] = 721
    raw["current"]["players"][1]["active"] = [{
        "id": 800, "serial": 9, "hp": 50, "maxHp": 50, "appearThisTurn": False,
        "energies": [], "energyCards": [], "tools": [], "preEvolution": [],
    }]
    raw["select"]["option"] = [
        {"type": 13, "attackId": 900},  # ATTACK (lethal)
        {"type": 14},                    # END
    ]
    raw["select"]["maxCount"] = 1
    api = _FakeApi(raw, {})
    pol = _policy(api)
    out = pol.choose(_ctx(raw))
    assert out == [0]        # took the lethal
    assert api.begins == 0   # fast path, no search


# -- stub determinizers -----------------------------------------------------

class _StubDet:
    def sample(self, obs, n):
        return [_H() for _ in range(n)]


class _RaisingDet:
    def sample(self, obs, n):
        from ptcg_ai.planning.determinize import DeterminizeError
        raise DeterminizeError("stub")


@dataclass
class _H:
    def as_args(self):
        return ([], [], [], [], [], [])
