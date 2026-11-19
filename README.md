# Ciągły trening w GitHub Actions z oceną w Langfuse

Demo do referatu 4 z MLOps („CI, CD i ciągły trening”), część 4. Model przewiduje rezygnację klientów (churn) na danych syntetycznych. Wszystko działa w GitHub Actions na runnerze na serwerze `ai-os-server`, a oceny trafiają do Langfuse.

Pętla ma dwa workflowy:

1. **`monitor`** liczy jakość modelu produkcyjnego (AUC) na najnowszych danych z etykietami i zapisuje ją w Langfuse jako ocenę `production_auc`. Decyzję podejmuje na podstawie oceny odczytanej z Langfuse. Gdy spadnie poniżej progu `monitor_min_auc` z `gates.yaml` (0,75), uruchamia trening.
2. **`ml-pipeline`** przechodzi przez testy kodu, kontrakt danych, trening kandydata, bramki jakości i wydanie. Każdy etap zapisuje swoje oceny w Langfuse pod jednym śladem uruchomienia. Bramka porównuje kandydata z modelem produkcyjnym na tym samym, aktualnym zbiorze testowym: model produkcyjny jest oceniany na nim od nowa.

Rejestr modeli leży na serwerze w `~/mlops-ct-demo/registry`, więc wydanie przetrwa między uruchomieniami. `reset-registry` przywraca w nim wersję v1.

## Scenariusz na zajęcia

| Krok | Workflow i dane | Wynik |
| --- | --- | --- |
| 0 | `reset-registry` | produkcja = v1 |
| 1 | `monitor`, `labelled` = `data/incoming/batch_ok.csv` | AUC 0,787 ≥ 0,75: bez zmian |
| 2 | `monitor`, `labelled` = `data/incoming/batch_nov.csv`, `holdout` = `data/holdout_nov.csv` | AUC 0,734 < 0,75: monitor sam uruchamia `ml-pipeline` |
| 3 | `ml-pipeline` uruchomiony przez monitor | kandydat 0,785 wobec produkcji 0,728 na tym samym zbiorze: wydanie v2 |
| 4 | `monitor`, `labelled` = `data/holdout_nov.csv` | AUC 0,785 ≥ 0,75: bez zmian |

Dane listopadowe mają dryf pojęć: po zmianie cennika klienci planu basic przestali odchodzić, a każde zgłoszenie do supportu waży dwa razy więcej. Na starym zbiorze testowym nowy kandydat wypadłby gorzej (0,804 wobec 0,823) i bramka by go zatrzymała. Dlatego po dryfie oceniamy oba modele na świeżym zbiorze z nowego okresu.

Partie z referatu (`batch_broken.csv`, `batch_shuffled_labels.csv`) dalej zatrzymują `ml-pipeline` na kontrakcie albo na bramce jakości.

## Oceny w Langfuse

| Ślad | Oceny |
| --- | --- |
| `production-monitoring` | `production_auc` |
| `ml-pipeline` | `contract_ok`, `contract_violations`, `train_rows`, `auc_candidate`, `auc_production`, `gates_passed`, `released`, `released_version` |

Identyfikator śladu to `gh-<numer uruchomienia>-<próba>`, a w metadanych jest link do uruchomienia w GitHub Actions.

## Konfiguracja

- Repozytorium jest publiczne, a runner wykonuje kod z repozytorium na serwerze. Dlatego workflowy startują tylko ręcznie albo po `push` na `main` (bez wyzwalaczy `pull_request` i bez harmonogramu), workflowy z pull requestów od osób spoza repozytorium czekają na zatwierdzenie właściciela, a dozwolone są tylko akcje GitHuba (`actions/*`).
- Zmienna repozytorium `LANGFUSE_HOST` i sekrety `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (klucze projektu w Langfuse). Bez kluczy pipeline działa, a zapis do Langfuse jest pomijany.
- Runner z etykietą `ai-os-server`. Środowisko Pythona tworzy `scripts/ci_env.sh` w `~/mlops-ct-demo/venv-<skrót requirements>` i używa go ponownie.

## Lokalnie

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python pipeline.py --reset && python pipeline.py data/incoming/batch_ok.csv   # pipeline z referatu, bez Langfuse
```

Dane generuje `scripts/make_data.py` (deterministycznie, ziarna w kodzie).
