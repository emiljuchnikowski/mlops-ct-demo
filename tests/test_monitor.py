"""The retraining loop: monitoring on fresh labelled data, then gates on the same, current holdout."""
import pytest

from mlpipe import stages, tracking

OK = stages.DATA / "incoming" / "batch_ok.csv"
NOV = stages.DATA / "incoming" / "batch_nov.csv"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A fresh v1 registry and artifacts in a temp dir, without Langfuse."""
    for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(stages, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(stages, "REGISTRY", tmp_path / "registry")
    stages.bootstrap()
    return tmp_path


def test_drift_drops_production_quality_below_the_threshold(registry):
    healthy, drifted = stages.monitor(OK), stages.monitor(NOV)
    assert healthy["auc"] >= healthy["threshold"]
    assert drifted["auc"] < drifted["threshold"]


def test_retrained_candidate_beats_production_on_the_current_holdout(registry, monkeypatch):
    monkeypatch.setenv("MLPIPE_HOLDOUT", "data/holdout_nov.csv")
    stages.train(NOV)
    result = stages.evaluate()
    assert result["champion_auc_source"] == "ten sam zbiór testowy"
    assert result["passed"] and result["auc"] > result["champion_auc"]
    assert stages.release()["version"] == "v2"


def test_without_langfuse_keys_logging_is_a_no_op(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert tracking.log("ml-pipeline", {"auc": 0.8}) is None
    assert tracking.read_score("auc", "any") is None
