"""Why can't we match Yushin? Split the fidelity ceiling into its three parts.

Every milestone so far asked "can a better model close the gap?" and answered by training a
better model. Nobody asked the prior question: **how much of the gap is closable at all?**

Three sources, and they need completely different fixes:

  (A) IRREDUCIBLE. The same RAW observation appears twice and Yushin plays differently. Then
      his action is not a function of what we can see -- he carries internal state, runs a
      search, or randomises. No model of any size fixes this, because the target is not a
      function of the input.

  (B) FEATURE LOSS. Two DIFFERENT raw observations collapse to the SAME feature vector X, and
      Yushin plays differently. Our featurizer threw away the distinguishing information. A
      bigger network cannot recover it -- `featurize_decision` already destroyed it. Only new
      features help.

  (C) LEARNABLE. Everything else: distinct inputs, distinct correct answers, model just gets
      them wrong. This is the ONLY part architecture or capacity can address.

The M37 set transformer bought +0.017. If (C) is small, that is the whole story and it was
never an architecture problem.

CAVEAT, stated up front: collisions concentrate in simpler, more repetitive positions, so
these rates are not a uniform sample of all decisions. The measurement bounds the problem;
it does not pin an exact number.

Run:  PYTHONPATH=src python -u scratchpad/ceiling_decomposition.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ptcg_ai.imitation.features as F
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset
from ptcg_ai.imitation.deck_profiles import get_profile
from ptcg_ai.observation.models import SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option
from ptcg_ai.state.game_state import GameState

DATASET = Path("data/imitation/yushinito_full.jsonl.gz")


def _h(obj) -> str:
    return hashlib.blake2b(
        json.dumps(obj, sort_keys=True, default=str).encode(), digest_size=16
    ).hexdigest()


def _action_key(row, obs, cards) -> str:
    """WHAT was played, not which index -- two copies of a card are the same move (R21)."""
    keys = []
    for a in row.action:
        if not (0 <= a < len(obs.select.option)):
            continue
        o = obs.select.option[a]
        try:
            r = resolve_option(o, obs)
            keys.append((o.type.name, r.card_id, r.attack_id, r.number))
        except Exception:
            keys.append((o.type.name, None, None, None))
    return _h(sorted(map(str, keys)))


def main() -> int:
    cards = CardDatabase.from_sdk(load_sdk(load_config(profile="benchmark").paths.sdk_dir))
    parser = ObservationParser()
    profile = get_profile("ALAKAZAM")
    MAIN = int(SelectContextKind.MAIN)

    by_raw: dict[str, set[str]] = defaultdict(set)     # raw obs  -> distinct actions
    by_x: dict[str, set[str]] = defaultdict(set)       # features -> distinct actions
    x_to_raw: dict[str, set[str]] = defaultdict(set)   # features -> distinct raw obs
    n_raw: dict[str, int] = defaultdict(int)
    n_x: dict[str, int] = defaultdict(int)
    total = 0

    for r in read_decision_dataset(DATASET):
        if r.context != MAIN:
            continue
        obs = parser.parse(r.raw_observation)
        if obs.select is None or obs.current is None or F.is_prize_pick(obs.select):
            continue
        chosen = [a for a in r.action if 0 <= a < len(obs.select.option)]
        if not chosen or len(set(chosen)) != len(chosen):
            continue
        gs = GameState.build(obs, cards)
        X = F.featurize_decision(profile, obs.select, obs, gs, cards)

        # the raw key excludes `logs` (free text that differs on every step and would make
        # every observation unique) and keeps the actual game state the agent can see
        raw = {k: v for k, v in r.raw_observation.items() if k not in ("logs", "search_begin_input")}
        rk, xk, ak = _h(raw), _h(X), _action_key(r, obs, cards)
        by_raw[rk].add(ak); n_raw[rk] += 1
        by_x[xk].add(ak);   n_x[xk] += 1
        x_to_raw[xk].add(rk)
        total += 1
        if total % 20000 == 0:
            print(f"  {total} decisiones...", flush=True)

    print(f"\ndecisiones MAIN analizadas: {total:,}\n")

    # --- (A) irreducible: identical RAW observation, different action -------------
    rep_raw = {k: v for k, v in n_raw.items() if v > 1}
    dec_rep_raw = sum(rep_raw.values())
    incons_raw = sum(n for k, n in rep_raw.items() if len(by_raw[k]) > 1)
    print("(A) IRREDUCIBLE -- misma observacion CRUDA, accion distinta")
    print(f"    observaciones que se repiten: {len(rep_raw):,} "
          f"(cubren {dec_rep_raw:,} decisiones = {dec_rep_raw/total:.1%})")
    if dec_rep_raw:
        print(f"    de esas, en estados con accion INCONSISTENTE: {incons_raw:,} "
              f"= {incons_raw/dec_rep_raw:.1%} de las repetidas")
        print(f"    -> techo por no-determinismo/estado oculto: ~{1 - incons_raw/dec_rep_raw:.3f}")

    # --- (B) feature loss: same X, different raw obs, different action ------------
    rep_x = {k: v for k, v in n_x.items() if v > 1}
    dec_rep_x = sum(rep_x.values())
    incons_x = sum(n for k, n in rep_x.items() if len(by_x[k]) > 1)
    collapsed = sum(n for k, n in rep_x.items() if len(x_to_raw[k]) > 1)
    collapsed_bad = sum(n for k, n in rep_x.items()
                        if len(x_to_raw[k]) > 1 and len(by_x[k]) > 1)
    print("\n(B) PERDIDA POR FEATURES -- mismo vector X, accion distinta")
    print(f"    vectores X que se repiten: {len(rep_x):,} "
          f"(cubren {dec_rep_x:,} decisiones = {dec_rep_x/total:.1%})")
    if dec_rep_x:
        print(f"    de esas, accion INCONSISTENTE: {incons_x:,} = {incons_x/dec_rep_x:.1%}")
        print(f"    -> techo de Bayes de ESTA representacion: ~{1 - incons_x/dec_rep_x:.3f}")
        print(f"    decisiones donde X colapsa observaciones distintas: {collapsed:,} "
              f"({collapsed/dec_rep_x:.1%} de las repetidas)")
        print(f"    de esas, ademas con accion distinta (perdida REAL): {collapsed_bad:,}")

    # --- (C) what compounding does to a per-decision number ----------------------
    per_game = total / len({r.game_id for r in read_decision_dataset(DATASET)
                            if r.context == MAIN})
    print(f"\n(C) EFECTO ACUMULATIVO -- {per_game:.0f} decisiones MAIN por partida")
    for acc, name in ((0.768, "MLP embarcada"), (0.786, "transformer k=7"), (0.95, "hipotetico")):
        print(f"    fidelidad {acc:.3f} ({name:<16}) -> P(partida identica) = "
              f"{acc ** per_game:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
