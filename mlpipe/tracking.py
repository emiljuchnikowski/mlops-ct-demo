"""Minimal Langfuse client (stdlib only) for the production model's answers and their ratings.

Langfuse holds what the model answered, not how it was trained: one trace per prediction (customer in, churn
probability out), grouped in a session per data batch, and the 'feedback' ratings people and the test reviewer
give those answers. Training, gates and releases stay in GitHub Actions (logs and artifacts).

Configured by LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY. Without them every call is a no-op,
so the pipeline and the tests run locally without Langfuse.
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK = "feedback"
TAG = "mlops-ct-demo"


def _key(env: str, prefix: str) -> str:
    """The key itself, even when a whole pasted line landed in the secret (e.g. LANGFUSE_SECRET_KEY = "sk-lf-...")."""
    raw = os.environ.get(env, "").strip()
    found = re.search(rf"{prefix}[A-Za-z0-9-]+", raw)
    return found.group(0) if found else raw


def _config():
    host = os.environ.get("LANGFUSE_HOST", "").strip().rstrip("/")
    public, secret = _key("LANGFUSE_PUBLIC_KEY", "pk-lf-"), _key("LANGFUSE_SECRET_KEY", "sk-lf-")
    return (host, public, secret) if host and public and secret else None


def enabled() -> bool:
    return _config() is not None


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    host, public, secret = _config()
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{host}{path}", data=data, method=method)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{public}:{secret}".encode()).decode())
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or b"{}")


def _ingest(batch: list, what: str) -> None:
    body = _request("POST", "/api/public/ingestion", {"batch": batch})
    if body.get("errors"):
        raise RuntimeError(f"Langfuse odrzucił {what}: {body['errors']}")


def run_url() -> str | None:
    env = os.environ
    if not env.get("GITHUB_RUN_ID"):
        return None
    return f"{env.get('GITHUB_SERVER_URL')}/{env.get('GITHUB_REPOSITORY')}/actions/runs/{env['GITHUB_RUN_ID']}"


# --- the model's answers ---------------------------------------------------------------------------------------

def prediction_trace_id(version: str, customer_id: int) -> str:
    """Deterministic, so logging the same answer again updates its trace and keeps its ratings."""
    return f"answer-{version}-{customer_id}"


def session_for(sample: dict) -> str:
    """One session per model version and data batch, e.g. 'v1-batch_nov': the answers given on that batch."""
    return f"{sample['version']}-{Path(sample['labelled']).stem}"


def log_predictions(sample: dict) -> int:
    """Each answer of the production model as a trace with a generation inside: customer in, churn probability out."""
    if not enabled():
        return 0
    now = datetime.now(timezone.utc).isoformat()
    model, session = f"churn-{sample['version']}", session_for(sample)
    batch = []
    for row in sample["rows"]:
        tid = prediction_trace_id(sample["version"], row["customer_id"])
        answer = {"churn_probability": row["probability"], "prediction": "odejdzie" if row["prediction"] else "zostanie"}
        meta = {"model_version": sample["version"], "data": sample["labelled"], "actual_churn": row["actual"],
                "github_run": run_url()}
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "trace-create", "body": {
            "id": tid, "name": "churn-prediction", "timestamp": now, "sessionId": session,
            "userId": f"klient-{row['customer_id']}", "input": row["features"], "output": answer, "metadata": meta,
            "release": sample["version"], "tags": [TAG, sample["version"]]}})
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "generation-create", "body": {
            "id": f"{tid}-answer", "traceId": tid, "name": "predict_proba", "model": model,
            "startTime": now, "endTime": now, "input": row["features"], "output": answer, "metadata": meta}})
    _ingest(batch, "odpowiedzi modelu")
    return len(sample["rows"])


# --- ratings of the answers ------------------------------------------------------------------------------------

def ensure_feedback_config() -> str | None:
    """The 'feedback' score (0 to 1) as a score config, so people can rate answers in the Langfuse UI."""
    if not enabled():
        return None
    configs = _request("GET", "/api/public/score-configs?limit=100").get("data") or []
    for config in configs:
        if config.get("name") == FEEDBACK and not config.get("isArchived"):
            return config["id"]
    created = _request("POST", "/api/public/score-configs", {
        "name": FEEDBACK, "dataType": "NUMERIC", "minValue": 0, "maxValue": 1,
        "description": "Ocena odpowiedzi modelu churn: 1 = trafna, 0 = chybiona. Średnia poniżej progu uruchamia ponowny trening."})
    return created["id"]


def rate(trace_id: str, value: float, comment: str, config_id: str | None) -> None:
    """A test reviewer's rating. Its id is fixed per trace, so a re-run overwrites it instead of adding another."""
    body = {"id": f"{trace_id}-feedback-test", "traceId": trace_id, "name": FEEDBACK, "value": float(value),
            "dataType": "NUMERIC", "comment": comment}
    if config_id:
        body["configId"] = config_id
    now = datetime.now(timezone.utc).isoformat()
    _ingest([{"id": uuid.uuid4().hex, "timestamp": now, "type": "score-create", "body": body}], "ocenę")


def _pages(path: str) -> list:
    """All items of a paginated list endpoint."""
    items, page = [], 1
    while True:
        body = _request("GET", f"{path}&limit=100&page={page}")
        items += body.get("data") or []
        if page >= (body.get("meta") or {}).get("totalPages", 1):
            return items
        page += 1


def _scores(name: str) -> list:
    """All scores with this name (v2 endpoint, falls back to v1)."""
    try:
        return [s for s in _pages(f"/api/public/v2/scores?name={name}") if s.get("name") == name]
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
        return [s for s in _pages(f"/api/public/scores?name={name}") if s.get("name") == name]


def feedback_for(version: str) -> list[float]:
    """All 'feedback' ratings on this model version's answers, from the test reviewer and from people."""
    if not enabled():
        return []
    prefix = f"answer-{version}-"
    return [float(s["value"]) for s in _scores(FEEDBACK) if str(s.get("traceId", "")).startswith(prefix)]


# --- maintenance and diagnostics -------------------------------------------------------------------------------

OLD_SCORES = ("production_auc", "feedback_mean", "feedback_count", "contract_ok", "contract_violations", "train_rows",
              "auc_candidate", "auc_production", "gates_passed", "released", "released_version")


def reset_demo() -> dict:
    """Delete this demo's traces (tag mlops-ct-demo) and their scores. Other data in the project stays."""
    if not enabled():
        return {"skipped": "brak kluczy"}
    trace_ids = [t["id"] for t in _pages(f"/api/public/traces?tags={TAG}")]
    ours = set(trace_ids)
    score_ids = [s["id"] for name in (FEEDBACK, *OLD_SCORES) for s in _scores(name)
                 if s.get("traceId") in ours or str(s.get("traceId", "")).startswith(("gh-", "pred-", "answer-", "local-"))]
    for sid in score_ids:
        _request("DELETE", f"/api/public/scores/{sid}")
    for i in range(0, len(trace_ids), 100):
        _request("DELETE", "/api/public/traces", {"traceIds": trace_ids[i:i + 100]})
    return {"traces_deleted": len(trace_ids), "scores_deleted": len(score_ids)}


def _latest_trace() -> dict | None:
    """The newest demo trace, to check it is well linked: session, user, input, output and its observations."""
    data = _request("GET", f"/api/public/traces?tags={TAG}&limit=1").get("data") or []
    if not data:
        return None
    t = data[0]
    return {"id": t.get("id"), "name": t.get("name"), "sessionId": t.get("sessionId"), "userId": t.get("userId"),
            "input": t.get("input"), "output": t.get("output"),
            "observations": len(t.get("observations") or []), "scores": len(t.get("scores") or [])}


def _counts() -> dict:
    """How many demo traces, sessions, observations and ratings the project holds (totals from the list endpoints)."""
    counts = {}
    for label, paths in {
        "traces_demo": [f"/api/public/traces?tags={TAG}&limit=1"],
        "sessions": ["/api/public/sessions?limit=1"],
        "observations": ["/api/public/observations?limit=1", "/api/public/v2/observations?limit=1"],
        "ratings_feedback": [f"/api/public/v2/scores?name={FEEDBACK}&limit=1", f"/api/public/scores?name={FEEDBACK}&limit=1"],
    }.items():
        for path in paths:
            try:
                meta = _request("GET", path).get("meta") or {}
                counts[label] = meta.get("totalItems", "brak meta")
                break
            except urllib.error.HTTPError as err:
                counts[label] = f"{err.code} {err.reason}"
    return counts


def check() -> dict:
    """Diagnose the Langfuse setup without revealing the keys: prefixes, stray whitespace, and the API's answer."""
    raw_public, raw_secret = os.environ.get("LANGFUSE_PUBLIC_KEY", ""), os.environ.get("LANGFUSE_SECRET_KEY", "")
    report = {
        "host": os.environ.get("LANGFUSE_HOST", "").strip() or None,
        "public_key_set": bool(raw_public.strip()), "secret_key_set": bool(raw_secret.strip()),
        "public_starts_pk_lf": raw_public.strip().startswith("pk-lf-"),
        "secret_starts_sk_lf": raw_secret.strip().startswith("sk-lf-"),
        "keys_swapped": raw_public.strip().startswith("sk-lf-") and raw_secret.strip().startswith("pk-lf-"),
        "secret_contains_sk_lf": "sk-lf-" in raw_secret, "public_contains_pk_lf": "pk-lf-" in raw_public,
        "extracted_secret_length": len(_key("LANGFUSE_SECRET_KEY", "sk-lf-")),
        "whitespace_around_keys": raw_public != raw_public.strip() or raw_secret != raw_secret.strip(),
        "public_key_length": len(raw_public.strip()), "secret_key_length": len(raw_secret.strip()),
    }
    if enabled():
        try:
            body = _request("GET", "/api/public/projects")
            report["api"] = "200 OK"
            report["projects"] = [p.get("name") for p in body.get("data", [])]
            report["counts"] = _counts()
            report["latest_trace"] = _latest_trace()
        except urllib.error.HTTPError as err:
            report["api"] = f"{err.code} {err.reason}"
    return report
