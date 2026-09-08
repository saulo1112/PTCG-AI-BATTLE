"""Kaggle-faithful check of greedy-v5: exec main.py with no __file__ and a
clean sys.path, confirm greedy engages and the W2 Mega Signal path is live."""
import os
import sys

ex = sys.argv[1]
sys.path[:] = [p for p in sys.path if os.path.join("Pokemon TCG", "src") not in p]
os.chdir(ex)
env = {"__name__": "__main__"}
exec(compile(open("main.py", encoding="utf-8").read(), "main.py", "exec"), env)
agent = env["agent"]
print("deck call len:", len(agent({"select": None})))
print("_AGENT_READY:", env.get("_AGENT_READY"))

bench5 = [{"id": 722, "serial": 2, "hp": 70, "maxHp": 70, "appearThisTurn": False,
           "energies": [], "energyCards": [], "tools": [], "preEvolution": []}] * 5
# MAIN with Mega Signal (1145) in hand, bench full, energy attached -> W2 tier 2
# should fire (play the search Item), not END.
obs = {"select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                  "remainDamageCounter": 0, "remainEnergyCost": 0,
                  "option": [{"type": 7, "index": 0}, {"type": 14}],
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
                        "hand": [{"id": 1145, "serial": 9, "playerIndex": 0}],
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
print("MAIN w/ Mega Signal(1145) -> action:", agent(obs),
      "(expect [0] = play it; proves W2 whitelist is live in the bundle)")
