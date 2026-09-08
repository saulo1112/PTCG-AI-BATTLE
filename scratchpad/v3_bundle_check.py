"""Kaggle-faithful check of the greedy-v3 bundle: exec main.py with no
__file__ and a clean sys.path (dev src stripped), confirm greedy engages and
the M3 Trainer path is live in the SHIPPED code."""
import os
import sys

ex = sys.argv[1]  # extracted bundle dir
sys.path[:] = [p for p in sys.path if os.path.join("Pokemon TCG", "src") not in p]
os.chdir(ex)
src = open("main.py", encoding="utf-8").read()
env = {"__name__": "__main__"}
exec(compile(src, "main.py", "exec"), env)
agent = env["agent"]
print("deck call len:", len(agent({"select": None})))
print("_AGENT_READY:", env.get("_AGENT_READY"))

bench5 = [{"id": 722, "serial": 2, "hp": 70, "maxHp": 70, "appearThisTurn": False,
           "energies": [], "energyCards": [], "tools": [], "preEvolution": []}] * 5
obs = {"select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                  "remainDamageCounter": 0, "remainEnergyCost": 0,
                  "option": [{"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
                             {"type": 7, "index": 0}, {"type": 14}],
                  "deck": None, "contextCard": None, "effect": None},
       "logs": [],
       "current": {"turn": 3, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                   "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": True,
                   "retreated": False, "result": -1, "stadium": [], "looking": None,
                   "players": [
                       {"active": [{"id": 721, "serial": 1, "hp": 60, "maxHp": 60,
                                    "appearThisTurn": False, "energies": [3, 3], "energyCards": [],
                                    "tools": [], "preEvolution": []}],
                        "bench": bench5, "benchMax": 5, "deckCount": 30, "discard": [],
                        "prize": [None] * 6, "handCount": 1,
                        "hand": [{"id": 1152, "serial": 9, "playerIndex": 0}],
                        "poisoned": False, "burned": False, "asleep": False,
                        "paralyzed": False, "confused": False},
                       {"active": [{"id": 800, "serial": 3, "hp": 200, "maxHp": 200,
                                    "appearThisTurn": False, "energies": [], "energyCards": [],
                                    "tools": [], "preEvolution": []}],
                        "bench": [], "benchMax": 5, "deckCount": 30, "discard": [],
                        "prize": [None] * 6, "handCount": 4, "hand": None,
                        "poisoned": False, "burned": False, "asleep": False,
                        "paralyzed": False, "confused": False}]},
       "search_begin_input": "AAAA"}
action = agent(obs)
print("MAIN w/ Poke Pad(1152) in hand -> action:", action,
      "(expect [1] = play the search Item; proves the M3 Trainer path is live in the bundle)")
