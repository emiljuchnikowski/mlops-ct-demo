"""Minimal Langfuse client (stdlib only): one trace per pipeline run, numeric scores on it, and reading scores back.

Configured by LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY. Without them every call is a no-op,
so the pipeline and the tests run locally without Langfuse.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone


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


_LOCAL_RUN = f"local-{uuid.uuid4().hex[:12]}"
STARTED = datetime.now(timezone.utc).isoformat()  # one CI step = one process, so this is the stage's start


def session_id() -> str:
    """Langfuse session: the monitor run and the ml-pipeline run it starts share one (LANGFUSE_SESSION)."""
    return os.environ.get("LANGFUSE_SESSION", "").strip() or f"ct-{run_id()}"


def run_id() -> str:
    """One trace per GitHub Actions run, shared by all its jobs; local runs get a random id."""
    rid = os.environ.get("GITHUB_RUN_ID")
    return f"gh-{rid}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}" if rid else _LOCAL_RUN


def run_url() -> str | None:
    env = os.environ
    if not env.get("GITHUB_RUN_ID"):
        return None
    return f"{env.get('GITHUB_SERVER_URL')}/{env.get('GITHUB_REPOSITORY')}/actions/runs/{env['GITHUB_RUN_ID']}"


def log(trace_name: str, scores: dict, metadata: dict | None = None, trace_id: str | None = None,
        stage: str | None = None, output=None) -> str | None:
    """Upsert the run's trace (in its session), add the stage as a span with its result, and write numeric scores.

    Span and score ids are deterministic, so a re-run of the same job overwrites instead of duplicating.
    """
    if not enabled():
        print("Langfuse: brak kluczy, pomijam zapis")
        return None
    tid = trace_id or run_id()
    now = datetime.now(timezone.utc).isoformat()
    trace = {"id": tid, "name": trace_name, "timestamp": now, "sessionId": session_id(), "tags": ["mlops-ct-demo"],
             "input": metadata or {}, "metadata": {"github_run": run_url()}}
    if output is not None:
        trace["output"] = output
    batch = [{"id": uuid.uuid4().hex, "timestamp": now, "type": "trace-create", "body": trace}]
    if stage:
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "span-create", "body": {
            "id": f"{tid}-{stage}", "traceId": tid, "name": stage, "startTime": STARTED, "endTime": now,
            "input": metadata or {}, "output": output, "metadata": {"github_run": run_url()}}})
    for name, value in scores.items():
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "score-create",
                      "body": {"id": f"{tid}-{name}", "traceId": tid, "name": name, "value": float(value),
                               "dataType": "NUMERIC"}})
    body = _request("POST", "/api/public/ingestion", {"batch": batch})
    if body.get("errors"):
        raise RuntimeError(f"Langfuse odrzucił zapis: {body['errors']}")
    print(f"Langfuse: ślad {tid} (sesja {session_id()}), etap {stage or '-'}, {len(scores)} ocen")
    return tid


def _scores(name: str) -> list:
    """Scores with this name, newest first (tries the v2 endpoint, falls back to v1)."""
    for path in (f"/api/public/v2/scores?name={name}&limit=100", f"/api/public/scores?name={name}&limit=100"):
        try:
            data = _request("GET", path).get("data") or []
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue
            raise
        return sorted((s for s in data if s.get("name") == name), key=lambda s: s.get("timestamp", ""), reverse=True)
    return []


def read_score(name: str, trace_id: str, wait_s: int = 90) -> float | None:
    """Read this trace's score back from Langfuse. Ingestion is asynchronous, so poll until it is visible."""
    if not enabled():
        return None
    deadline = time.time() + wait_s
    while time.time() < deadline:
        for score in _scores(name):
            if score.get("traceId") == trace_id:
                return float(score["value"])
        time.sleep(3)
    return None


def _counts() -> dict:
    """How many traces, sessions, observations and scores the project holds (totals from the list endpoints)."""
    counts = {}
    for label, paths in {
        "traces_mlops_ct_demo": ["/api/public/traces?tags=mlops-ct-demo&limit=1"],
        "traces_prediction": ["/api/public/traces?name=prediction&limit=1"],
        "sessions": ["/api/public/sessions?limit=1"],
        "observations": ["/api/public/observations?limit=1", "/api/public/v2/observations?limit=1"],
        "scores_feedback": [f"/api/public/v2/scores?name={FEEDBACK}&limit=1", f"/api/public/scores?name={FEEDBACK}&limit=1"],
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
        except urllib.error.HTTPError as err:
            report["api"] = f"{err.code} {err.reason}"
    return report


FEEDBACK = "feedback"


def prediction_trace_id(version: str, customer_id: int) -> str:
    """Deterministic, so logging the same sample again updates the traces and keeps their ratings."""
    return f"pred-{version}-{customer_id}"


def log_predictions(sample: dict) -> int:
    """One trace per prediction, with the customer as input and the churn probability as output, ready to rate."""
    if not enabled():
        return 0
    now = datetime.now(timezone.utc).isoformat()
    batch = []
    for row in sample["rows"]:
        tid = prediction_trace_id(sample["version"], row["customer_id"])
        output = {"churn_probability": row["probability"], "prediction": row["prediction"]}
        meta = {"model_version": sample["version"], "labelled": sample["labelled"],
                "customer_id": row["customer_id"], "actual_churn": row["actual"], "github_run": run_url()}
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "trace-create", "body": {
            "id": tid, "name": "prediction", "timestamp": now, "sessionId": f"predykcje-{sample['version']}",
            "input": row["features"], "output": output, "metadata": meta,
            "tags": ["mlops-ct-demo", "prediction", sample["version"]]}})
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "span-create", "body": {
            "id": f"{tid}-predict", "traceId": tid, "name": f"model {sample['version']}: predict_proba",
            "startTime": now, "endTime": now, "input": row["features"], "output": output, "metadata": meta}})
    body = _request("POST", "/api/public/ingestion", {"batch": batch})
    if body.get("errors"):
        raise RuntimeError(f"Langfuse odrzucił predykcje: {body['errors']}")
    return len(sample["rows"])


def ensure_feedback_config() -> str | None:
    """The 'feedback' score (0 to 1) as a score config, so people can rate predictions in the Langfuse UI."""
    if not enabled():
        return None
    configs = _request("GET", "/api/public/score-configs?limit=100").get("data") or []
    for config in configs:
        if config.get("name") == FEEDBACK and not config.get("isArchived"):
            return config["id"]
    created = _request("POST", "/api/public/score-configs", {
        "name": FEEDBACK, "dataType": "NUMERIC", "minValue": 0, "maxValue": 1,
        "description": "Ocena predykcji modelu churn: 1 = trafna, 0 = chybiona. Średnia poniżej progu uruchamia ponowny trening."})
    return created["id"]


def rate(trace_id: str, value: float, comment: str, config_id: str | None) -> None:
    """A test reviewer's rating. Its id is fixed per trace, so a re-run overwrites it instead of adding another."""
    body = {"id": f"{trace_id}-feedback-test", "traceId": trace_id, "name": FEEDBACK, "value": float(value),
            "dataType": "NUMERIC", "comment": comment}
    if config_id:
        body["configId"] = config_id
    now = datetime.now(timezone.utc).isoformat()
    result = _request("POST", "/api/public/ingestion",
                      {"batch": [{"id": uuid.uuid4().hex, "timestamp": now, "type": "score-create", "body": body}]})
    if result.get("errors"):
        raise RuntimeError(f"Langfuse odrzucił ocenę: {result['errors']}")


def feedback_for(version: str) -> list[float]:
    """All 'feedback' ratings on this model version's predictions, from the test reviewer and from people."""
    if not enabled():
        return []
    prefix = f"pred-{version}-"
    return [float(s["value"]) for s in _scores(FEEDBACK) if str(s.get("traceId", "")).startswith(prefix)]


def safe_log(trace_name: str, scores: dict, metadata: dict | None = None, stage: str | None = None, output=None) -> None:
    """Logging must never decide a gate: a Langfuse outage becomes a CI warning, not a failed stage."""
    try:
        log(trace_name, scores, metadata, stage=stage, output=output)
    except Exception as err:  # noqa: BLE001
        print(f"::warning title=Langfuse::zapis nieudany: {err}")
