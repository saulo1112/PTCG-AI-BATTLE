"""Probe top-player replay structure and try to extract their deck submission."""
from __future__ import annotations

import json
from pathlib import Path

DIR = Path("Logs/Higher ranking logs/Vibechu")


def main():
    f = sorted(DIR.glob("*.json"))[0]
    print("file:", f.name, "size(MB):", f.stat().st_size / 1e6)
    data = json.loads(f.read_text(encoding="utf-8"))
    print("top keys:", list(data.keys()))
    info = data.get("info", {})
    print("info keys:", list(info.keys()))
    # agent names
    for k in ("Agents", "TeamNames"):
        if k in info:
            print(k, "=", info[k])
    steps = data.get("steps", [])
    print("num steps:", len(steps))
    if steps:
        s0 = steps[0]
        print("step0 is list of", len(s0), "agents")
        for i, ag in enumerate(s0):
            akeys = list(ag.keys())
            action = ag.get("action")
            atype = type(action).__name__
            alen = len(action) if isinstance(action, list) else None
            print(f"  agent{i}: keys={akeys} status={ag.get('status')} "
                  f"action_type={atype} action_len={alen}")
            if isinstance(action, list) and len(action) == 60:
                from collections import Counter
                print(f"    *** DECK (60): {dict(Counter(action))}")
    # also check step1 actions (deck may be answered at step 0's obs -> action in step 1)
    if len(steps) > 1:
        for i, ag in enumerate(steps[1]):
            action = ag.get("action")
            if isinstance(action, list) and len(action) == 60:
                from collections import Counter
                print(f"  step1 agent{i} DECK(60): {dict(Counter(action))}")


if __name__ == "__main__":
    main()
