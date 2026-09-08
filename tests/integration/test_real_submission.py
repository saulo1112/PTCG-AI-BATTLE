"""Build the real submission (with the actual vendored cg/) and validate it."""

import pytest

from ptcg_ai.config import AppConfig
from ptcg_ai.submission.builder import build_submission
from ptcg_ai.submission.validate import smoke_test_entrypoint, validate_submission

pytestmark = pytest.mark.sdk


def test_real_bundle_valid_and_smokes(app_config: AppConfig) -> None:
    tarball = build_submission(app_config, output_name="integration-test")
    assert validate_submission(tarball, app_config.submission.size_limit_mib) == []
    assert smoke_test_entrypoint(tarball) == []
    size_mib = tarball.stat().st_size / (1024 * 1024)
    assert size_mib < app_config.submission.size_limit_mib
