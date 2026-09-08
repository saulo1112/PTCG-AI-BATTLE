"""Stage-0 audit: does the v3 clone over-use Kilowattrel's Flashing Draw ability?

This is the direct check for M8.1's failure mode. imitation-v2.0 cloned an
800-elo deck but over-clicked its draw engine (Lunar Cycle: clone 88.7% take-rate
vs teacher 69.2%), discarding energy it needed for attacks and losing prize races.
Flashing Draw (Kilowattrel, id 271) is the same shape of trap: "discard a {L} from
this Pokémon → draw to 6". On the held-out val split, we compare the teacher's
take-rate vs the trained clone's take-rate on decisions where Flashing Draw is an
offered MAIN option. A large positive gap (clone >> teacher) is a red flag.

Run:  uv run --group dev python scratchpad/audit_flashing_draw.py \
          data/imitation/kenn2439.jsonl.gz decks/kenn2439.csv data/models/bc_940_v1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from ptcg_ai.agent.deck import load_deck
from ptcg_ai.cards.database import CardDatabase
from ptcg_ai.config import load_config
from ptcg_ai.decision.base import DecisionContext
from ptcg_ai.environment.sdk import load_sdk
from ptcg_ai.imitation.dataset import read_decision_dataset, split_by_game
from ptcg_ai.imitation.policy import ImitationPolicy
from ptcg_ai.observation.models import OptionKind, SelectContextKind
from ptcg_ai.observation.parser import ObservationParser
from ptcg_ai.observation.resolve import resolve_option

_KILOWATTREL_ID = 271


def _flashing_draw_options(obs) -> list[int]:
    """Option indices in this decision that fire Kilowattrel's Flashing Draw."""
    if obs.select is None:
        return []
    out = []
    for i, opt in enumerate(obs.select.option):
        if opt.type is OptionKind.ABILITY:
            try:
                if resolve_option(opt, obs).card_id == _KILOWATTREL_ID:
                    out.append(i)
            except Exception:
                continue
    return out


def main() -> int:
    dataset, deck_csv, weights = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = load_config(profile="benchmark")
    cards = CardDatabase.from_sdk(load_sdk(cfg.paths.sdk_dir))
    parser = ObservationParser()
    deck = load_deck(Path(deck_csv))
    clone = ImitationPolicy(weights, deck=deck.as_list())

    rows = list(read_decision_dataset(Path(dataset)))
    _, val_rows = split_by_game(rows, val_fraction=0.2, seed=0)

    offered = teacher_took = clone_took = 0
    for r in val_rows:
        if SelectContextKind(r.context) is not SelectContextKind.MAIN:
            continue
        obs = parser.parse(r.raw_observation)
        fd = _flashing_draw_options(obs)
        if not fd:
            continue
        offered += 1
        if any(a in fd for a in r.action):
            teacher_took += 1
        try:
            chosen = clone.choose(DecisionContext(raw=r.raw_observation, observation=obs, cards=cards))
        except Exception:
            chosen = []
        if any(a in fd for a in chosen):
            clone_took += 1

    if offered == 0:
        print("Flashing Draw was never an offered MAIN option in val — nothing to audit.")
        return 0
    t_rate = teacher_took / offered
    c_rate = clone_took / offered
    print(f"Flashing Draw offered in {offered} val MAIN decisions")
    print(f"  teacher take-rate: {t_rate:.3f} ({teacher_took}/{offered})")
    print(f"  clone   take-rate: {c_rate:.3f} ({clone_took}/{offered})")
    print(f"  gap (clone - teacher): {c_rate - t_rate:+.3f}")
    print(f"  verdict: {'RED FLAG (clone over-uses draw engine)' if c_rate - t_rate > 0.10 else 'ok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
