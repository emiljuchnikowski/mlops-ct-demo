"""Minimal Langfuse client (stdlib only): one trace per pipeline run, numeric scores on it, and reading scores back.

Configured by LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY. Without them every call is a no-op,
so the pipeline and the tests run locally without Langfuse.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone


def _config():
    # Pasted keys often carry a trailing newline or space, which turns into 401 Unauthorized.
    host = os.environ.get("LANGFUSE_HOST", "").strip().rstrip("/")
    public, secret = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip(), os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
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


def run_id() -> str:
    """One trace per GitHub Actions run, shared by all its jobs; local runs get a random id."""
    rid = os.environ.get("GITHUB_RUN_ID")
    return f"gh-{rid}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}" if rid else f"local-{uuid.uuid4().hex[:12]}"


def run_url() -> str | None:
    env = os.environ
    if not env.get("GITHUB_RUN_ID"):
        return None
    return f"{env.get('GITHUB_SERVER_URL')}/{env.get('GITHUB_REPOSITORY')}/actions/runs/{env['GITHUB_RUN_ID']}"


def log(trace_name: str, scores: dict, metadata: dict | None = None, trace_id: str | None = None) -> str | None:
    """Upsert the run's trace and write numeric scores on it. Score ids are deterministic, so a re-run overwrites."""
    if not enabled():
        print("Langfuse: brak kluczy, pomijam zapis")
        return None
    tid = trace_id or run_id()
    now = datetime.now(timezone.utc).isoformat()
    batch = [{"id": uuid.uuid4().hex, "timestamp": now, "type": "trace-create",
              "body": {"id": tid, "name": trace_name, "timestamp": now, "tags": ["mlops-ct-demo"],
                       "metadata": {**(metadata or {}), "github_run": run_url()}}}]
    for name, value in scores.items():
        batch.append({"id": uuid.uuid4().hex, "timestamp": now, "type": "score-create",
                      "body": {"id": f"{tid}-{name}", "traceId": tid, "name": name, "value": float(value),
                               "dataType": "NUMERIC"}})
    body = _request("POST", "/api/public/ingestion", {"batch": batch})
    if body.get("errors"):
        raise RuntimeError(f"Langfuse odrzucił zapis: {body['errors']}")
    print(f"Langfuse: zapisano {len(scores)} ocen w śladzie {tid}")
    return tid


def _scores(name: str) -> list:
    """Scores with this name, newest first (tries the v2 endpoint, falls back to v1)."""
    for path in (f"/api/public/v2/scores?name={name}&limit=50", f"/api/public/scores?name={name}&limit=50"):
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


def check() -> dict:
    """Diagnose the Langfuse setup without revealing the keys: prefixes, stray whitespace, and the API's answer."""
    raw_public, raw_secret = os.environ.get("LANGFUSE_PUBLIC_KEY", ""), os.environ.get("LANGFUSE_SECRET_KEY", "")
    report = {
        "host": os.environ.get("LANGFUSE_HOST", "").strip() or None,
        "public_key_set": bool(raw_public.strip()), "secret_key_set": bool(raw_secret.strip()),
        "public_starts_pk_lf": raw_public.strip().startswith("pk-lf-"),
        "secret_starts_sk_lf": raw_secret.strip().startswith("sk-lf-"),
        "keys_swapped": raw_public.strip().startswith("sk-lf-") and raw_secret.strip().startswith("pk-lf-"),
        "whitespace_around_keys": raw_public != raw_public.strip() or raw_secret != raw_secret.strip(),
        "public_key_length": len(raw_public.strip()), "secret_key_length": len(raw_secret.strip()),
    }
    if enabled():
        try:
            body = _request("GET", "/api/public/projects")
            report["api"] = "200 OK"
            report["projects"] = [p.get("name") for p in body.get("data", [])]
        except urllib.error.HTTPError as err:
            report["api"] = f"{err.code} {err.reason}"
    return report


def safe_log(trace_name: str, scores: dict, metadata: dict | None = None) -> None:
    """Logging must never decide a gate: a Langfuse outage becomes a CI warning, not a failed stage."""
    try:
        log(trace_name, scores, metadata)
    except Exception as err:  # noqa: BLE001
        print(f"::warning title=Langfuse::zapis nieudany: {err}")
