"""Find update hyperparameters that do not destroy the Alakazam model.

Iteration 1 collapsed: bc_acc 0.733 -> 0.168, agreement with the base 1.000 -> 0.172,
while the weights moved only ~5%. Root cause measured: the Alakazam ensemble's scores are
2.0-2.4x more compressed than the TR-650 one M16 tuned against (margin 1.98 vs 4.66, std
3.42 vs 6.99), so the same absolute step flips far more argmaxes.

This re-runs ONLY the update on the trajectory already collected -- no new games -- so a
whole hyperparameter sweep costs minutes instead of 33 min per setting.

`tau` stays at the value the data was SAMPLED with (1.0). The policy gradient is only
valid at the sampling temperature; changing tau here would silently make the update
off-policy. Lowering tau for exploration is a change for the NEXT collection.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rl_selfplay as R
from ptcg_ai.observation.parser import ObservationParser

TRAJ = "data/rl_alakazam/iter1_traj.jsonl.gz"
SAMPLED_TAU = 1.0


def main():
    R.configure("alakazam")
    sdk, cards = R._sdk_cards()
    parser = ObservationParser()
    probe = R._probe_decisions(cards, parser, limit=4000)
    _, base = R._load_main_members(R.BASE_CKPT)
    ag0, acc0 = R._probe(base, base, probe)
    print(f"BASE: agree={ag0:.3f} bc_acc={acc0:.3f}  (K2 debe medirse contra ESTO, no contra 0.93)\n")

    v = R._load_v650(R.V650)
    decs = R._featurize_traj(TRAJ, cards, parser, v)
    print(f"decisiones en la trayectoria: {len(decs)}\n")

    print(f"{'lr':>8}{'epochs':>8}{'lambda':>8}{'agree':>9}{'bc_acc':>9}{'d(bc_acc)':>11}")
    for lr in (1e-3, 3e-4, 1e-4, 3e-5):
        for epochs in (25, 10):
            _, mem = R._load_main_members(R.BASE_CKPT)
            mem = R.rl_update(mem, base, decs, lambda_a=0.10, tau=SAMPLED_TAU,
                              lr=lr, epochs=epochs)
            ag, acc = R._probe(mem, base, probe)
            print(f"{lr:>8.0e}{epochs:>8}{0.10:>8.2f}{ag:>9.3f}{acc:>9.3f}{acc - acc0:>+11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
