"""Imitation learning (behavior cloning) — ladder rung 6 (M7).

Fits a per-context linear policy that reproduces a strong logged player's
decisions, targeting piloting skill directly instead of hand-deriving rules
(the wall M5/M6 hit: greedy pilots an engine deck to 0.02–0.19 that its owner
pilots to 0.47–0.62). Only :mod:`~ptcg_ai.imitation.features` and
:mod:`~ptcg_ai.imitation.policy` are pure-stdlib and shippable; the loader,
dataset builder, and trainer are dev-only tooling.
"""
