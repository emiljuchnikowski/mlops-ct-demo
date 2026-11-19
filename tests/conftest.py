"""Tests never depend on the run's settings: the workflow sets the holdout and Langfuse keys for the pipeline, not for pytest."""
import pytest


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for key in ("MLPIPE_HOLDOUT", "LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "GITHUB_OUTPUT"):
        monkeypatch.delenv(key, raising=False)
