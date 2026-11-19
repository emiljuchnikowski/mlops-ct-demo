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


def test_without_langfuse_keys_langfuse_calls_are_no_ops(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert tracking.log_predictions({"version": "v1", "labelled": "x.csv", "rows": []}) == 0
    assert tracking.feedback_for("v1") == []


def test_answers_are_traces_with_a_generation_in_a_session_per_batch(registry, monkeypatch):
    for key, value in {"LANGFUSE_HOST": "http://x", "LANGFUSE_PUBLIC_KEY": "pk-lf-a", "LANGFUSE_SECRET_KEY": "sk-lf-b"}.items():
        monkeypatch.setenv(key, value)
    sent = []
    monkeypatch.setattr(tracking, "_request", lambda method, path, payload=None: sent.append(payload) or {})
    sample = stages.predictions_sample(NOV)
    assert tracking.log_predictions(sample) == 10
    events = sent[0]["batch"]
    traces = [e["body"] for e in events if e["type"] == "trace-create"]
    answers = [e["body"] for e in events if e["type"] == "generation-create"]
    assert len(traces) == len(answers) == 10
    assert {t["sessionId"] for t in traces} == {"v1-batch_nov"}
    assert all(t["input"] and t["output"] and t["userId"].startswith("klient-") for t in traces)
    assert {a["traceId"] for a in answers} == {t["id"] for t in traces} and answers[0]["model"] == "churn-v1"


def test_low_ratings_trigger_retraining_even_with_a_good_auc():
    assert not stages.retrain_decision(0.80, [1, 1, 0])["retrain"]          # too few ratings to count
    low = stages.retrain_decision(0.80, [1, 0, 0, 0, 1])                     # 5 ratings, mean 0.4
    assert low["retrain"] and "średnia ocen" in low["reasons"][0]
    assert stages.retrain_decision(0.70, [])["retrain"]                     # AUC alone still decides


def test_prediction_sample_is_fixed_and_labelled(registry):
    sample = stages.predictions_sample(OK)
    assert sample["version"] == "v1" and len(sample["rows"]) == 10
    assert sample == stages.predictions_sample(OK)
    assert {"probability", "prediction", "actual", "features"} <= set(sample["rows"][0])


def test_one_retraining_per_piece_of_evidence(registry):
    low = stages.retrain_decision(0.80, [0, 0, 0, 0, 1])
    assert not stages.already_triggered("v1", OK, low)
    stages.remember_trigger("v1", OK, low)
    assert stages.already_triggered("v1", OK, low)
    assert not stages.already_triggered("v1", OK, stages.retrain_decision(0.80, [0, 0, 0, 0, 1, 0]))  # a new rating
    assert not stages.already_triggered("v2", OK, low)                                                  # a new model
    stages.bootstrap()
    assert not stages.already_triggered("v1", OK, low)
