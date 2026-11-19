"""Local run of the whole pipeline, stage by stage, stopping at the first failed gate.

    python pipeline.py data/incoming/batch_ok.csv
    python pipeline.py data/incoming/batch_broken.csv
    python pipeline.py data/incoming/batch_broken.csv --skip-contract   # what happens without a contract
    python pipeline.py --reset                                          # registry back to v1
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from mlpipe import stages

TTY = sys.stdout.isatty()


def paint(text: str, color: str) -> str:
    codes = {"green": "32", "red": "31", "grey": "90", "bold": "1", "yellow": "33"}
    return f"\033[{codes[color]}m{text}\033[0m" if TTY else text


def row(n: int, name: str, status: str, color: str, note: str = "") -> None:
    print(f"[{n}/5] {name:<30} {paint(f'{status:<5}', color)} {note}")


def run(batch: str, skip_contract: bool) -> int:
    prod = stages.production()
    print(paint(f"\nPipeline ML · partia: {Path(batch).name} · produkcja: {prod['version']} (AUC {prod['auc']})\n", "bold"))
    steps = ["Testy kodu (pytest)", "Kontrakt danych churn v1", "Trening kandydata",
             "Bramki jakości vs produkcja", "Wydanie do rejestru"]

    def stop(at: int, reason: str) -> int:
        for i in range(at + 1, 5):
            row(i + 1, steps[i], "--", "grey", "pominięty")
        print(paint(f"\nWYDANIE ZATRZYMANE na etapie „{steps[at]}”: {reason}", "red"))
        print(f"Produkcja bez zmian: {prod['version']} (AUC {prod['auc']}).\n")
        return 1

    # 1. CI for code: the same tests a pull request must pass.
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True,
                           cwd=stages.ROOT)
    passed = re.search(r"(\d+) passed", tests.stdout)
    if tests.returncode != 0:
        row(1, steps[0], "STOP", "red", tests.stdout.strip().splitlines()[-1])
        return stop(0, "testy kodu nie przechodzą")
    row(1, steps[0], "OK", "green", f"{passed.group(1) if passed else '?'} testów")

    # 2. CI for data: the new batch must honour the data contract.
    if skip_contract:
        row(2, steps[1], "SKIP", "yellow", "wyłączony flagą --skip-contract")
    else:
        violations = stages.check_contract(batch)
        if violations:
            row(2, steps[1], "STOP", "red", f"naruszenia: {len(violations)}")
            for v in violations:
                print(paint(f"        - {v}", "red"))
            return stop(1, "dane łamią kontrakt")
        row(2, steps[1], "OK", "green", "schemat, zakresy, kategorie, braki")

    # 3. Continuous training: historical data + the new batch.
    info = stages.train(batch)
    row(3, steps[2], "OK", "green", f"{info['rows']} wierszy")

    # 4. Quality gates: absolute floor and champion vs challenger on a fixed holdout.
    result = stages.evaluate()
    note = f"AUC {result['auc']} vs {result['champion']} {result['champion_auc']}"
    if not result["passed"]:
        row(4, steps[3], "STOP", "red", note)
        return stop(3, "; ".join(result["failed_gates"]))
    row(4, steps[3], "OK", "green", note)

    # 5. Release: the registry pointer moves to the new version.
    entry = stages.release()
    row(5, steps[4], "OK", "green", f"{prod['version']} -> {entry['version']}")
    print(paint(f"\nWYDANO {entry['version']} (AUC {entry['auc']}), trenowany na: {', '.join(entry['trained_on'])}\n", "green"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", nargs="?", help="CSV z nową partią danych")
    parser.add_argument("--skip-contract", action="store_true", help="pomiń kontrakt (antywzorzec do porównania)")
    parser.add_argument("--reset", action="store_true", help="przywróć rejestr do v1")
    args = parser.parse_args()
    if args.reset:
        entry = stages.bootstrap()
        print(f"Rejestr zresetowany: produkcja = {entry['version']} (AUC {entry['auc']})")
        return 0
    if not args.batch:
        parser.print_help()
        return 2
    return run(args.batch, args.skip_contract)


if __name__ == "__main__":
    sys.exit(main())
