<!--
Title: Powerful Hand: An Alakazam Combo Agent
Subtitle: An imitation-learning agent trained on the ladder's replay data, choosing the strongest source by how learnable its decisions were, validated match by match against a 60 to 100 elo noise floor.

Paste everything below this comment into the "Project Description" field of the Kaggle Writeup.
Do not paste this comment block itself.
-->

This project builds an autonomous agent for the Pokemon Trading Card Game Simulation
competition: a single Python function that receives the current game state and returns a
legal move, playing full games under hidden information and shuffles against other
submitted agents. The final agent finished 859th of 6,725 entries.

Two things carried more weight in getting there than any single algorithmic idea. First,
training a policy on decision logs from strong agents already active on the ladder beat
every rule-based and search-based approach tried first, and choosing whose logs to train on
depended on a specific, measurable property of the data, not that agent's rating. Second, the
ladder and most offline evaluation methods are noisy enough to read a real improvement as a
loss, or a loss as a win, so almost none of this project's actual progress could be trusted
without a specific protocol built to catch that. Both are explained below, alongside the one
change that measurably worked.

## Training on the ladder's own replay data

Early attempts at this competition used hand-written rules and short lookahead search, and
both hit the same wall. One test made this concrete: an existing rule-based pilot played the
exact 60-card deck used by a strong agent already active on the ladder. That agent won 62
percent of the time with the list; the rule-based pilot won only 19 percent with the
identical cards. The deck was unchanged, so the 43-point gap was pure decision quality, and no
amount of rule tuning had closed it.

The response was to train a model directly on decisions recorded from a strong agent's own
games instead of writing more rules, an approach known as imitation learning or behavior
cloning. This is not an improvised shortcut: every submitted agent's full replay log is
downloadable through the competition's own API, so learning from what already works on the
ladder is a built-in, intended part of how the competition can be played. The first such
model beat the best rule-based pilot in a 300-game arena, winning 71.7 percent of matches
(95 percent confidence interval 66.3 to 76.5 percent), the first agent in the project to
clear that bar.

The next question was whose decisions to train on. The obvious answer, the highest-rated
agent available, turned out to be wrong. A model trained to predict a strong agent's
decisions with only 0.9 points more accuracy on average still lost a controlled rematch by
3.3 points, confidence interval entirely negative: predicting the average decision and
predicting the decisions that decide a game are not the same problem. The criterion that
mattered instead was how consistently an agent's decisions could be learned, not its raw
rating. This held with an unrelated deck earlier in the
project, where a higher-rated agent's decisions still produced a weaker model on an almost
identical decklist, and it held again later: a candidate whose decisions were the most
learnable ever measured in the project, predicted correctly 81.3 percent of the time, on a
much simpler deck, lost a 600-game match against the current agent 27.5 to 72.5, with a noise
control (explained below) confirming the loss was real.

The final training data was chosen on that basis rather than rating: a highly rated agent
active on the ladder at the time, piloting a three-stage evolution combo built around
Alakazam, whose decisions were both strong and, by the same measure that had already misled
once, genuinely learnable.

## The deck: building a hand to spend it

The final agent pilots a Stage 2 evolution combo deck built around Alakazam. Its signature
attack, Powerful Hand, deals damage equal to 20 times the number of cards in the attacker's
own hand, delivered as placed damage counters rather than a standard damage calculation, so
it ignores Weakness and Resistance entirely. That single mechanic decides almost every
strategic choice in the deck.

The tension is direct: every card played to develop the board, attaching energy, evolving, or
playing a Supporter, reduces the hand and therefore reduces the attack's own damage. A deck
built around this attack cannot simply play cards as they are drawn. The pattern that works
is building the hand across several turns, then committing it all at once on the turn
Powerful Hand fires.

Every card category in the list below serves that pattern, not generic consistency. Rare
Candy (3 copies) skips the Kadabra stage entirely, evolving Abra straight to Alakazam and
reaching the attack a turn earlier. Hilda, Dawn, and Poke Pad (4 copies each) are pure
card-search and draw: their job is to fill the hand fast in the turns before it gets spent,
not to develop the board directly. Fezandipiti ex, worth two Prize cards if knocked out, is a
secondary attacker for the turns before the hand is stacked, trading risk for tempo. Shaymin
is played for its Ability, protecting the bench, not its attack. Dunsparce and Dudunsparce are
the deck's draw engine, keeping the hand-building turns from stalling on a bad draw.

| Category | Cards |
|---|---|
| Pokemon (19) | 4 Abra, 4 Kadabra, 4 Alakazam, 3 Dunsparce, 2 Dudunsparce, 1 Fezandipiti ex, 1 Shaymin |
| Trainer, Item (17) | 4 Enhanced Hammer, 4 Buddy-Buddy Poffin, 4 Poke Pad, 3 Rare Candy, 1 Night Stretcher, 1 Sacred Ash |
| Trainer, Supporter (15) | 4 Hilda, 4 Dawn, 3 Boss's Orders, 3 Xerosic's Machinations, 1 Lana's Aid |
| Trainer, Stadium (2) | 2 Nighttime Mine |
| Energy (7) | 2 Basic Psychic Energy, 4 Telepath Psychic Energy, 1 Enriching Energy |

At 60 cards exactly (19 Pokemon, 34 Trainer, 7 Energy), every card in this list answers the
same question the attack itself asks: does this help build the hand, or does it spend it?

## Why every result had to survive a noise test

Every claim of improvement in this project had to survive a specific problem: the ladder
itself is noisy enough to look like signal. The same unchanged set of model weights,
uploaded on three separate occasions, scored 772.7, 856.8, and 899.4, a range of about 127
points, purely because each upload faced a different pool of opponents (two of the runs
shared only about 2 of roughly 110 opponents). The working rule became that a single
converged ladder score carries on the order of plus or minus 60 to 100 points of pure
opponent-pool noise, and that comparing two agents fairly means giving them the same
opponent pool, not comparing two separate scores.

Offline evaluation was not automatically safer. One instrument, an 8-opponent scored
gauntlet, was calibrated against a real, independently confirmed gap between two agents:
converged ladder ratings showed one agent was genuinely about 90 points better than the
other, a difference of 0.13 in win rate. The same instrument, run on the same two agents,
read that known gap as minus 0.003, a coin flip, with the sign reversed. Averaging a real
effect over a handful of matchups had cancelled it out; the effect only showed up across a
large, varied, adaptive field.

The clearest warning came from a self-mirror. Any agent played against an exact copy of
itself must, by construction, win close to half the time. At a sample of 200 games, that
same-agent mirror match read 43.5 percent. At 600 games, it converged to 50.3 percent. The
gap between those two numbers is not a real effect; it is the sampling noise of the
evaluation method itself, and it was large enough to be mistaken for a genuine 6-point swing.

From this, the project settled on one protocol for every decisive comparison: 600 paired
games, run alongside a self-mirror control that must itself land near 50 percent before the
result is trusted. If the control does not land near 50, the run is discarded, not
interpreted. Nothing described in the rest of this report was accepted without clearing that
bar.

## The one confirmed win

Auditing which decisions the agent actually modeled, rather than how well it modeled them,
found the single largest fix in the project. Across a corpus of 190,731 real decisions,
23,354 of them, 12.2 percent of everything the agent ever decided, were being made by a
hardcoded fallback that always picks the first listed option without looking at the board at
all. The fallback existed because training required any learned model to beat that same
fallback by a fixed margin, and for several decision types the fallback was already accurate
enough that the margin was mathematically out of reach.

Two examples make the blind spot concrete. In one recurring decision, whether to decline an
optional action, the source agent declined about 6.6 percent of the time across 760
instances; the shipped agent declined zero times, always taking the fallback. In another,
choosing which Pokemon to place on the bench during setup, the source agent never benched a
specific high-value attacker (worth two Prize cards if knocked out) across 261 occurrences, a
deliberate risk-avoidance choice; the shipped agent benched that same Pokemon in every one of
those situations, because it had no model there either.

Small learned models were built for four of these blind or under-modeled decision types.
Tested under the full protocol above, 600 paired games with a self-mirror control, the fixed
agent beat the previous version 0.596 to 0.404, 95 percent confidence interval 0.556 to
0.634, entirely above the no-improvement line: the first result in the project to clear 95
percent significance. A second, independent run confirmed it at 0.617, confidence interval
0.577 to 0.655.

| Test | Result | Reference | 95% CI |
|---|---|---|---|
| Fixed agent vs. previous version (run 1) | 0.596 | 0.500 | 0.556 to 0.634 |
| Fixed agent vs. previous version (run 2) | 0.617 | 0.500 | 0.577 to 0.655 |
| Self-mirror noise control | 0.490 to 0.516 | 0.500 | within range |

The result held up on the real ladder afterward, in the single hardest matchup the deck
faced: win rate against the archetype that had been the project's toughest opponent
throughout rose from 35 percent to 47 percent. An improvement measured in the worst matchup,
rather than a favorable one, is the opposite of what a cherry-picked result would look like.

| Matchup | Win rate before | Win rate after |
|---|---|---|
| Hardest opponent archetype in the field | 35% | 47% |

## What did not work, and why that still counts

Technical soundness includes being able to say why an idea did not transfer, with a real
mechanism, not just that it failed. Self-play reinforcement learning was attempted several
times in this project and closed for exactly that reason. A variance decomposition of the
training signal showed that, even after substantially improving the value model used to
judge each move, 87.5 percent of the signal driving any single training update was
statistically indistinguishable from pure game-outcome noise, identical across all roughly 45
decisions in a game regardless of which one actually mattered. A corrected training procedure
reduced that noise share to about 2.5 percent, and it measurably changed how the model
updated: a coherence statistic on the optimization path flipped from clearly negative to
clearly positive. The resulting agent still played worse in the 600-game decisive test.
Testing the exact opposite of that same update direction also played worse, which points to
the starting point already sitting at a local optimum along that axis, not to a broken
implementation.

## Result

The final Simulation standing was 859th of 6,725 entries, top 13 percent, on a ladder that
restarts every new agent at the same rating and settles slowly. That number reflects one
training source, one deck, and a real limit the project hit more than once: predicting an
agent's decisions more accurately did not reliably mean playing better. What generalizes past
this specific deck is the measurement discipline: distrust a converged score until its noise
floor is known, and require every claimed improvement to survive a self-mirror control before
it counts.
