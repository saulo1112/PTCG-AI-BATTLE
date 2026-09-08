"""Determinized forward-model search plumbing (rung 5, ADR-0010).

``determinize`` samples concrete hidden-information worlds from a parsed
observation; ``session`` wraps the vendored ``cg`` search API in a
context-managed lifecycle. Both are consumed by
:class:`ptcg_ai.decision.search.SearchPolicy`.
"""
