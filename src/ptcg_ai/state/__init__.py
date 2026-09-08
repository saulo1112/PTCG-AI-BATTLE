"""Perspective-normalized game state and derived tactical quantities.

Created in Phase 2 with the first heuristic policy (Growth Plan,
docs/architecture.md; design in docs/phase2_blueprint.md §4.1). ``GameState``
turns a wire-shaped :class:`~ptcg_ai.observation.models.ParsedObservation`
into a "me vs opponent" view with the derived quantities rules consult
(prize race, KO math, tempo flags), computed once per decision.
"""

from ptcg_ai.state.game_state import GameState

__all__ = ["GameState"]
