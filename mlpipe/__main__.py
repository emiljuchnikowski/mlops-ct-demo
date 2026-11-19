"""One stage per call, for CI jobs: python -m mlpipe {validate,train,evaluate,release,monitor,review,langfuse-check,langfuse-reset,bootstrap} [file.csv]

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
    sample = stages.predictions_sample(labelled)
    feedback = []
    try:
        print(f"Langfuse: zapisano {tracking.log_predictions(sample)} predykcji modelu {sample['version']} do oceny")
        feedback = tracking.feedback_for(result["version"])
    except Exception as err:  # noqa: BLE001
        print(f"::warning title=Langfuse::predykcje lub oceny niedostępne: {err}")
    trace = None
    scores = {"production_auc": result["auc"]}
    if feedback:
        scores.update(feedback_mean=sum(feedback) / len(feedback), feedback_count=len(feedback))
    try:
        trace = tracking.log("production-monitoring", scores, {"labelled": result["labelled"], "production": result["version"]},
                             stage="pomiar jakości", output={**result, "feedback": feedback})
    except Exception as err:  # noqa: BLE001
        print(f"::warning title=Langfuse::zapis nieudany: {err}")
    # The AUC used in the decision is the one Langfuse stores; without Langfuse it falls back to the local value.
    stored = tracking.read_score("production_auc", trace) if trace else None
    source = "Langfuse"
    if stored is None:
        print("::warning title=Langfuse::brak oceny w Langfuse, decyzja na wartości lokalnej")
        stored, source = result["auc"], "lokalnie"
    decision = stages.retrain_decision(stored, feedback)
    ratings = (f"średnia ocen {decision['feedback_mean']} z {decision['feedback_count']}" if feedback
               else "brak ocen predykcji")
    verdict = "PONOWNY TRENING: " + "; ".join(decision["reasons"]) if decision["retrain"] else "bez zmian"
    # A scheduled run must not start the same retraining every hour: it waits for new ratings, data or a new model.
    if decision["retrain"]:
        if os.environ.get("MONITOR_TRIGGER") == "schedule" and stages.already_triggered(result["version"], labelled, decision):
            decision = {**decision, "retrain": False}
            verdict = "trening dla tych danych i ocen już uruchomiony, czekam na nowe oceny albo dane"
        else:
            stages.remember_trigger(result["version"], labelled, decision)
    print(f"::notice title=Monitoring::model {result['version']}: AUC {stored} ({source}), {ratings} -> {verdict}")
    if trace:
        tracking.safe_log("production-monitoring", {}, {"labelled": result["labelled"], "production": result["version"]},
                          stage="decyzja", output={**decision, "auc": stored, "auc_source": source})
    _output(retrain=decision["retrain"], production_auc=stored, reason="; ".join(decision["reasons"]) or "-")
    return 0


def review(labelled: str) -> int:
    """Test reviewer: rates the production model's sampled predictions, 1 when right and 0 when wrong."""
    if not tracking.enabled():
        print("Langfuse: brak kluczy, nie ma czego oceniać")
        return 1
    sample = stages.predictions_sample(labelled)
    tracking.log_predictions(sample)
    config = tracking.ensure_feedback_config()
    values = []
    for row in sample["rows"]:
        right = row["prediction"] == row["actual"]
        values.append(int(right))
        tracking.rate(tracking.prediction_trace_id(sample["version"], row["customer_id"]), int(right),
                      "ocena testowa: predykcja trafna" if right else "ocena testowa: predykcja chybiona", config)
    mean = round(sum(values) / len(values), 3)
    print(f"::notice title=Oceny testowe::model {sample['version']}: {len(values)} ocen w Langfuse, średnia {mean}")
    return 0


def main(argv: list[str]) -> int:
    cmd, path = argv[0], (argv[1] if len(argv) > 1 else None)
    if cmd == "validate":
        violations = stages.check_contract(path)
        for v in violations:
            print(f"::error title=Kontrakt danych::{v}")
        print("kontrakt OK" if not violations else f"kontrakt NARUSZONY ({len(violations)})")
        tracking.safe_log(PIPELINE, {"contract_ok": int(not violations), "contract_violations": len(violations)}, _context(),
                          stage="kontrakt danych", output={"violations": violations})
        return 1 if violations else 0
    if cmd == "train":
        info = stages.train(path)
        print(json.dumps(info, ensure_ascii=False))
        tracking.safe_log(PIPELINE, {"train_rows": info["rows"]}, _context(), stage="trening", output=info)
        return 0
    if cmd == "evaluate":
        result = stages.evaluate()
        for reason in result["failed_gates"]:
            print(f"::error title=Bramka jakości::{reason}")
        print(json.dumps(result, ensure_ascii=False))
        tracking.safe_log(PIPELINE, {"auc_candidate": result["auc"], "auc_production": result["champion_auc"],
                                     "gates_passed": int(result["passed"])}, _context(), stage="bramki jakości", output=result)
        return 0 if result["passed"] else 1
    if cmd == "release":
        entry = stages.release()
        print(json.dumps(entry, ensure_ascii=False))
        print(f"::notice title=Wydanie::produkcja = {entry['version']} (AUC {entry['auc']})")
        tracking.safe_log(PIPELINE, {"released": 1, "released_version": int(entry["version"][1:])}, _context(),
                          stage="wydanie", output=entry)
        return 0
    if cmd == "monitor":
        return monitor(path)
    if cmd == "review":
        return review(path)
    if cmd == "langfuse-check":
        report = tracking.check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("api") == "200 OK" else 1
    if cmd == "langfuse-reset":
        print(json.dumps(tracking.reset_demo(), ensure_ascii=False))
        return 0
    if cmd == "bootstrap":
        entry = stages.bootstrap()
        print(json.dumps(entry, ensure_ascii=False))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["help"]))
