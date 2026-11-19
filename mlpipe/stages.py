"""Pipeline stages: validate -> train -> evaluate -> release. Each one can run alone (CI job) or in sequence."""
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .contract import load_contract, validate
from .features import TARGET, build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
# On the self-hosted runner the registry lives outside the checkout, so releases survive between runs.
REGISTRY = Path(os.path.expanduser(os.environ.get("MLPIPE_REGISTRY", str(ROOT / "registry"))))
CONTRACT = ROOT / "contracts" / "churn.v1.yaml"
GATES = ROOT / "gates.yaml"


def holdout_path() -> Path:
    """The evaluation set for this run: the fixed holdout by default, a fresher labelled window after drift."""
    path = Path(os.environ.get("MLPIPE_HOLDOUT", "data/holdout.csv"))
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def production() -> dict:
    return json.loads((REGISTRY / "production.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint() -> dict:
    """What an approval is about: this candidate file, this holdout, these gate thresholds, this champion."""
    return {"candidate_sha256": _sha256(ARTIFACTS / "candidate.pkl"), "holdout_sha256": _sha256(holdout_path()),
            "gates_sha256": _sha256(GATES), "champion": production()["version"]}


def check_contract(batch: str | Path) -> list:
    violations = validate(pd.read_csv(batch), load_contract(CONTRACT))
    _write_json(ARTIFACTS / "contract_report.json",
                {"batch": Path(batch).name, "passed": not violations, "violations": [str(v) for v in violations]})
    return violations


def train(batch: str | Path | None) -> dict:
    """Train a candidate on the historical data plus the new batch."""
    frames = [pd.read_csv(DATA / "reference.csv")]
    if batch:
        frames.append(pd.read_csv(batch))
    df = pd.concat(frames, ignore_index=True)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0)
    model.fit(build_features(df), df[TARGET])
    ARTIFACTS.mkdir(exist_ok=True)
    # A new candidate invalidates any earlier approval: training is not consent to release.
    (ARTIFACTS / "evaluation.json").unlink(missing_ok=True)
    with open(ARTIFACTS / "candidate.pkl", "wb") as f:
        pickle.dump(model, f)
    info = {"trained_on": ["reference.csv"] + ([Path(batch).name] if batch else []), "rows": len(df)}
    _write_json(ARTIFACTS / "candidate.json", info)
    return info


def evaluate() -> dict:
    """Score the candidate on the fixed holdout set and apply the quality gates."""
    with open(ARTIFACTS / "candidate.pkl", "rb") as f:
        model = pickle.load(f)
    holdout = pd.read_csv(holdout_path())
    auc = _auc(model, holdout)
    gates = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    champion = production()
    # Champion and challenger on the same holdout: re-score the production model when its file is available.
    champion_auc, source = champion["auc"], "rejestr"
    if _model_path(champion["version"]).exists():
        champion_auc, source = _auc(_load(_model_path(champion["version"])), holdout), "ten sam zbiór testowy"
    failed = []
    if auc < gates["min_auc"]:
        failed.append(f"AUC {auc} < próg minimalny {gates['min_auc']}")
    if auc < champion_auc - gates["max_auc_drop_vs_production"]:
        failed.append(f"AUC {auc} gorsze od produkcyjnego {champion['version']} ({champion_auc})")
    result = {"auc": auc, "champion": champion["version"], "champion_auc": champion_auc, "champion_auc_source": source,
              "holdout": holdout_path().name,
              "passed": not failed, "failed_gates": failed, "evaluated": _fingerprint()}
    _write_json(ARTIFACTS / "evaluation.json", result)
    return result


def _auc(model, df: pd.DataFrame) -> float:
    return round(float(roc_auc_score(df[TARGET], model.predict_proba(build_features(df))[:, 1])), 4)


def _model_path(version: str) -> Path:
    return REGISTRY / "models" / f"model-{version}.pkl"


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def monitor(labelled: str | Path) -> dict:
    """Quality of the production model on the newest labelled data, against the retraining threshold."""
    champion = production()
    auc = _auc(_load(_model_path(champion["version"])), pd.read_csv(labelled))
    threshold = yaml.safe_load(GATES.read_text(encoding="utf-8"))["monitor_min_auc"]
    return {"version": champion["version"], "auc": auc, "threshold": threshold, "labelled": Path(labelled).name}


def release() -> dict:
    """Promote the evaluated candidate to production in the (file-based) model registry."""
    approval = ARTIFACTS / "evaluation.json"
    if not approval.exists():
        raise RuntimeError("brak oceny bieżącego kandydata — wydanie zablokowane")
    evaluation = json.loads(approval.read_text(encoding="utf-8"))
    if not evaluation["passed"]:
        raise RuntimeError("kandydat nie przeszedł bramek jakości — wydanie zablokowane")
    if evaluation.get("evaluated") != _fingerprint():
        raise RuntimeError("ocena dotyczy innego kandydata, holdoutu, bramek lub championa — wydanie zablokowane")
    version = f"v{int(production()['version'][1:]) + 1}"
    (REGISTRY / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy(ARTIFACTS / "candidate.pkl", REGISTRY / "models" / f"model-{version}.pkl")
    candidate = json.loads((ARTIFACTS / "candidate.json").read_text(encoding="utf-8"))
    entry = {"version": version, "auc": evaluation["auc"], "trained_on": candidate["trained_on"]}
    _write_json(REGISTRY / "production.json", entry)
    approval.unlink()  # one approval, one release
    return entry


def bootstrap() -> dict:
    """Reset the registry to v1: a model trained on the reference data only."""
    train(None)
    _write_json(REGISTRY / "production.json", {"version": "v0", "auc": 0.0, "trained_on": []})
    shutil.rmtree(REGISTRY / "models", ignore_errors=True)
    evaluate()
    entry = release()
    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    return entry
