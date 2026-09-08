"""Submission artifact pipeline (ADR-0001): build and validate the tarball."""

from ptcg_ai.submission.builder import build_submission
from ptcg_ai.submission.validate import smoke_test_entrypoint, validate_submission

__all__ = ["build_submission", "smoke_test_entrypoint", "validate_submission"]
