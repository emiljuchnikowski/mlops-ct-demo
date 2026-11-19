"""One stage per call, for CI jobs: python -m mlpipe {validate,train,evaluate,release,monitor,bootstrap} [file.csv]

Every stage writes its scores to the run's Langfuse trace (a no-op without Langfuse keys).
`monitor` decides about retraining from the score it reads back from Langfuse.
"""
import json
import os
import sys

from . import stages, tracking

PIPELINE = "ml-pipeline"


def _context() -> dict:
    return {"batch": os.environ.get("BATCH"), "holdout": stages.holdout_path().name,
            "trigger": os.environ.get("TRIGGER"), "production": stages.production()["version"]}


def _output(**values) -> None:
    """Expose values to later workflow steps (GITHUB_OUTPUT), and print them for local runs."""
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in values.items()]
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(" ".join(lines))


def monitor(labelled: str) -> int:
    result = stages.monitor(labelled)
    print(json.dumps(result, ensure_ascii=False))
    trace = None
    try:
        trace = tracking.log("production-monitoring", {"production_auc": result["auc"]},
                             {"labelled": result["labelled"], "production": result["version"]})
    except Exception as err:  # noqa: BLE001
        print(f"::warning title=Langfuse::zapis nieudany: {err}")
    # The decision uses the score as Langfuse stores it; without Langfuse it falls back to the local value.
    stored = tracking.read_score("production_auc", trace) if trace else None
    source = "Langfuse"
    if stored is None:
        print("::warning title=Langfuse::brak oceny w Langfuse, decyzja na wartości lokalnej")
        stored, source = result["auc"], "lokalnie"
    retrain = stored < result["threshold"]
    verdict = "PONOWNY TRENING" if retrain else "bez zmian"
    print(f"::notice title=Monitoring::AUC produkcji {stored} ({source}) przy progu {result['threshold']}: {verdict}")
    _output(retrain=retrain, production_auc=stored)
    return 0


def main(argv: list[str]) -> int:
    cmd, path = argv[0], (argv[1] if len(argv) > 1 else None)
    if cmd == "validate":
        violations = stages.check_contract(path)
        for v in violations:
            print(f"::error title=Kontrakt danych::{v}")
        print("kontrakt OK" if not violations else f"kontrakt NARUSZONY ({len(violations)})")
        tracking.safe_log(PIPELINE, {"contract_ok": int(not violations), "contract_violations": len(violations)}, _context())
        return 1 if violations else 0
    if cmd == "train":
        info = stages.train(path)
        print(json.dumps(info, ensure_ascii=False))
        tracking.safe_log(PIPELINE, {"train_rows": info["rows"]}, _context())
        return 0
    if cmd == "evaluate":
        result = stages.evaluate()
        for reason in result["failed_gates"]:
            print(f"::error title=Bramka jakości::{reason}")
        print(json.dumps(result, ensure_ascii=False))
        tracking.safe_log(PIPELINE, {"auc_candidate": result["auc"], "auc_production": result["champion_auc"],
                                     "gates_passed": int(result["passed"])}, _context())
        return 0 if result["passed"] else 1
    if cmd == "release":
        entry = stages.release()
        print(json.dumps(entry, ensure_ascii=False))
        print(f"::notice title=Wydanie::produkcja = {entry['version']} (AUC {entry['auc']})")
        tracking.safe_log(PIPELINE, {"released": 1, "released_version": int(entry["version"][1:])}, _context())
        return 0
    if cmd == "monitor":
        return monitor(path)
    if cmd == "langfuse-check":
        report = tracking.check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("api") == "200 OK" else 1
    if cmd == "bootstrap":
        entry = stages.bootstrap()
        print(json.dumps(entry, ensure_ascii=False))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["help"]))
