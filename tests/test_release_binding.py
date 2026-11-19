import pytest

from mlpipe import stages

OK = stages.DATA / "incoming" / "batch_ok.csv"
SHUFFLED = stages.DATA / "incoming" / "batch_shuffled_labels.csv"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Artifacts and registry in a temp dir, so the demo registry is never touched."""
    monkeypatch.setattr(stages, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(stages, "REGISTRY", tmp_path / "registry")
    stages._write_json(tmp_path / "registry" / "production.json", {"version": "v1", "auc": 0.80, "trained_on": []})
    return tmp_path


def test_new_training_invalidates_the_previous_approval(sandbox):
    stages.train(OK)
    assert stages.evaluate()["passed"]
    stages.train(SHUFFLED)  # candidate B replaces the approved candidate A
    with pytest.raises(RuntimeError, match="brak oceny"):
        stages.release()


def test_approval_copied_from_another_candidate_is_rejected(sandbox):
    stages.train(OK)
    stages.evaluate()
    approval = (sandbox / "artifacts" / "evaluation.json").read_text(encoding="utf-8")
    stages.train(SHUFFLED)
    (sandbox / "artifacts" / "evaluation.json").write_text(approval, encoding="utf-8")
    with pytest.raises(RuntimeError, match="innego kandydata"):
        stages.release()


def test_one_approval_releases_once(sandbox):
    stages.train(OK)
    stages.evaluate()
    assert stages.release()["version"] == "v2"
    with pytest.raises(RuntimeError):
        stages.release()
