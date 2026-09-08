"""ptcg_ai — research framework for the Pokémon TCG AI Battle Challenge.

The package is organized as thin, well-typed layers over the vendored
competition SDK (see ``docs/architecture.md``):

- ``environment``  — the only code allowed to import the vendored ``cg`` SDK
- ``observation``  — forward-compatible parsing of raw observation dicts
- ``decision``     — the Policy interface and baseline policies
- ``agent``        — the Kaggle-contract agent façade
- ``debug``        — observation pretty-printing and decision tracing
- ``bench``        — measurement harness that gates algorithm decisions
- ``submission``   — builds and validates the Kaggle submission tarball
"""

__version__ = "0.1.0"
