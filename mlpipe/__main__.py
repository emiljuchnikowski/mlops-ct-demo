"""One stage per call, for CI jobs: python -m mlpipe {validate,train,evaluate,release,monitor,review,langfuse-check,langfuse-reset,bootstrap} [file.csv]

Training, gates and releases report in GitHub Actions (logs, notices, artifacts). Langfuse holds the production
model's answers and their ratings: `monitor` sends the answers and decides about retraining from the ratings it
reads back, `review` is a test reviewer that rates the answers.
"""
import json
import os
import sys

from . import stages, tracking


def _output(**values) -> None:
    """Expose values to later workflow steps (GITHUB_OUTPUT), and print them for local runs."""
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in values.items()]
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(" ".join(lines))


def monitor(labelled: str) -> int:
    """The production model answers on the newest labelled data; its answers go to Langfuse, its ratings come back."""
    result = stages.monitor(labelled)
    print(json.dumps(result, ensure_ascii=False))
    sample = stages.predictions_sample(labelled)
    feedback = []
    try:
        sent = tracking.log_predictions(sample)
        if sent:
            print(f"Langfuse: {sent} odpowiedzi modelu {sample['version']} w sesji {tracking.session_for(sample)}")
        feedback = tracking.feedback_for(result["version"])
    except Exception as err:  # noqa: BLE001
        print(f"::warning title=Langfuse::odpowiedzi lub oceny niedostępne: {err}")
    decision = stages.retrain_decision(result["auc"], feedback)
    ratings = (f"średnia ocen odpowiedzi {decision['feedback_mean']} z {decision['feedback_count']} (Langfuse)"
               if feedback else "brak ocen odpowiedzi w Langfuse")
    verdict = "PONOWNY TRENING: " + "; ".join(decision["reasons"]) if decision["retrain"] else "bez zmian"
    # A scheduled run must not start the same retraining every hour: it waits for new ratings, data or a new model.
    if decision["retrain"]:
        if os.environ.get("MONITOR_TRIGGER") == "schedule" and stages.already_triggered(result["version"], labelled, decision):
            decision = {**decision, "retrain": False}
            verdict = "trening dla tych danych i ocen już uruchomiony, czekam na nowe oceny albo dane"
        else:
            stages.remember_trigger(result["version"], labelled, decision)
    print(f"::notice title=Monitoring::model {result['version']}: AUC {result['auc']}, {ratings} -> {verdict}")
    _output(retrain=decision["retrain"], production_auc=result["auc"], reason="; ".join(decision["reasons"]) or "-")
    return 0


def review(labelled: str) -> int:
    """Test reviewer: rates the production model's answers in Langfuse, 1 when right and 0 when wrong."""
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
                      "ocena testowa: odpowiedź trafna" if right else "ocena testowa: odpowiedź chybiona", config)
    mean = round(sum(values) / len(values), 3)
    print(f"::notice title=Oceny testowe::model {sample['version']}, sesja {tracking.session_for(sample)}: "
          f"{len(values)} ocen w Langfuse, średnia {mean}")
    return 0


def main(argv: list[str]) -> int:
    cmd, path = argv[0], (argv[1] if len(argv) > 1 else None)
    if cmd == "validate":
        violations = stages.check_contract(path)
        for v in violations:
            print(f"::error title=Kontrakt danych::{v}")
        print("kontrakt OK" if not violations else f"kontrakt NARUSZONY ({len(violations)})")
        return 1 if violations else 0
    if cmd == "train":
        print(json.dumps(stages.train(path), ensure_ascii=False))
        return 0
    if cmd == "evaluate":
        result = stages.evaluate()
        for reason in result["failed_gates"]:
            print(f"::error title=Bramka jakości::{reason}")
        print(json.dumps(result, ensure_ascii=False))
        if result["passed"]:
            print(f"::notice title=Bramka jakości::kandydat {result['auc']} wobec produkcji {result['champion_auc']}: przechodzi")
        return 0 if result["passed"] else 1
    if cmd == "release":
        entry = stages.release()
        print(json.dumps(entry, ensure_ascii=False))
        print(f"::notice title=Wydanie::produkcja = {entry['version']} (AUC {entry['auc']})")
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
