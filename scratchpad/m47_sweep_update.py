"""M47 — calibrate the update against the configuration we are ACTUALLY going to run.

WHY A SECOND SWEEP AT ALL. `_sweep_update.py` (M38) found lr=3e-5 / 10 epochs for the
Alakazam base. M47 changes two inputs that the M38 number was calibrated against, and
this project's most repeated failure is "a constant calibrated for another context,
inherited unchecked" (M38 §7 names four instances of it in one milestone):

  1. THE CRITIC. M47 F0 refit it: held-out AUC on self-play states 0.6526 -> 0.7902.
     A sharper V(s) explains more of the outcome, so A = R - V(s) SHRINKS. The M38 lr
     was tuned against the old, wider advantage distribution.
  2. THE TEMPERATURE. M38 collected at tau=1.0 (33.3% of MAIN decisions off-argmax,
     measured; M16's healthy TR-650 profile was ~16%). M47 F2 measured the curve and
     picks tau=0.3 -> 16.9%. The gradient carries a 1/tau factor, so the step size at
     tau=0.3 is not the step size at tau=1.0.

So the batch swept here is collected at tau=0.3 by the shipped champion, and the
advantages are computed with the NEW critic. That is the real configuration.

WHAT THE NUMBERS MEAN. `agree` is the fraction of held-out MAIN decisions where the
updated model still picks what the frozen BC init picks -- i.e. 1-agree is DISTANCE
TRAVELLED. `bc_acc` is agreement with the teacher -- QUALITY RETAINED. They trade off,
and the lr=0 row is the null control that makes them readable: an update that does
nothing scores agree=1.000 and bc_acc=base, which is why neither number alone can say
"this update is better". The target is M38's healthy profile -- ~3.5% of decisions
moved per update -- at the highest bc_acc available.

Run:  PYTHONPATH=src python -u scratchpad/m47_sweep_update.py [traj.jsonl.gz]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rl_selfplay as R  # noqa: E402
from ptcg_ai.observation.parser import ObservationParser  # noqa: E402

TRAJ = sys.argv[1] if len(sys.argv) > 1 else "data/rl_alakazam_final/calib_tau03.jsonl.gz"
SAMPLED_TAU = 0.3          # MUST match the tau the batch was collected at (see _sweep_update)
OLD_CRITIC = "data/models/v_alakazam.json"
NEW_CRITIC = "data/models/v_alakazam_v2.json"
TARGET_MOVED = 0.035       # M38's healthy per-update displacement


def _adv_stats(decs, label):
    a = np.array([d.weight for d in decs])
    print(f"  {label:<28} mean={a.mean():+.4f}  std={a.std():.4f}  "
          f"|A|median={np.median(np.abs(a)):.4f}  frac>0={float((a > 0).mean()):.3f}")
    return a


def main() -> int:
    R.configure("alakazam_final")
    sdk, cards = R._sdk_cards()
    parser = ObservationParser()

    probe = R._probe_decisions(cards, parser, limit=4000)
    _, base = R._load_main_members(R.BASE_CKPT)
    ag0, acc0 = R._probe(base, base, probe)
    print(f"\nBASE (champion): agree={ag0:.4f} bc_acc={acc0:.4f}  on {len(probe)} held-out "
          f"MAIN decisions\n")

    print("advantage distribution on the SAME trajectory, by critic:")
    decs_old = R._featurize_traj(TRAJ, cards, parser, R._load_v650(OLD_CRITIC))
    _adv_stats(decs_old, f"old ({Path(OLD_CRITIC).name})")
    decs = R._featurize_traj(TRAJ, cards, parser, R._load_v650(NEW_CRITIC))
    a_new = _adv_stats(decs, f"new ({Path(NEW_CRITIC).name})")
    print(f"  -> {len(decs)} decisions; advantage std shrank by "
          f"{1 - a_new.std() / np.array([d.weight for d in decs_old]).std():.1%}\n")

    rows = []
    print(f"{'clip':>7}{'advnorm':>9}{'lr':>10}{'agree':>9}{'moved':>9}{'bc_acc':>9}"
          f"{'d(bc)':>9}")
    print("-" * 62)
    configs = [(None, False, 0.0)]                       # <- the null control
    for lr in (1e-5, 3e-5, 1e-4, 3e-4):
        configs.append((None, False, lr))                # M38's update, new inputs
        configs.append((0.2, False, lr))                 # + PPO clip
    configs.append((0.2, True, 3e-5))                    # + advantage normalisation
    configs.append((0.2, True, 1e-4))

    for clip, advn, lr in configs:
        _, mem = R._load_main_members(R.BASE_CKPT)
        mem = R.rl_update(mem, base, decs, lambda_a=0.10, tau=SAMPLED_TAU,
                          lr=lr, epochs=10, clip=clip, adv_norm=advn)
        ag, acc = R._probe(mem, base, probe)
        rows.append((clip, advn, lr, ag, acc))
        print(f"{str(clip):>7}{str(advn):>9}{lr:>10.0e}{ag:>9.4f}{1 - ag:>9.4f}"
              f"{acc:>9.4f}{acc - acc0:>+9.4f}")

    print("-" * 62)
    # Pick the lr whose displacement lands nearest the M38 healthy profile, per arm.
    for clip in (None, 0.2):
        arm = [r for r in rows if r[0] == clip and not r[1] and r[2] > 0]
        if not arm:
            continue
        best = min(arm, key=lambda r: abs((1 - r[3]) - TARGET_MOVED))
        print(f"clip={str(clip):<5} nearest {TARGET_MOVED:.1%} displacement: "
              f"lr={best[2]:.0e}  (moved {1 - best[3]:.2%}, bc_acc {best[4]:.4f})")
    print("\nRead this as: same displacement, higher bc_acc = a better update. The lr=0 row")
    print("is the trivial ceiling on both columns -- a config that beats it on bc_acc while")
    print("moving nothing has learned nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
