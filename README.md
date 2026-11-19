# Ciągły trening w GitHub Actions z oceną w Langfuse

Demo do referatu 4 z MLOps („CI, CD i ciągły trening”), część 4. Model przewiduje rezygnację klientów (churn) na danych syntetycznych. Wszystko działa w GitHub Actions na runnerze na serwerze `ai-os-server`, a oceny trafiają do Langfuse.

Pętla ma trzy workflowy:

1. **`monitor`** działa co godzinę (i ręcznie). Liczy jakość modelu produkcyjnego (AUC) na najnowszych danych z etykietami i zapisuje ją w Langfuse jako ocenę `production_auc`. Do Langfuse wysyła też 10 predykcji modelu jako ślady `prediction`, które można ocenić. Decyzję podejmuje na podstawie oceny odczytanej z Langfuse i ocen predykcji: trening startuje, gdy AUC spadnie poniżej `monitor_min_auc` (0,75) albo gdy co najmniej `feedback_min_count` (5) ocen ma średnią poniżej `feedback_min_mean` (0,6). Progi są w `gates.yaml`. Przebieg z harmonogramu nie uruchamia drugi raz tego samego treningu: dla tej samej wersji modelu, tych samych danych i tej samej liczby ocen czeka na nowe oceny albo dane.
2. **`review`** to testowy recenzent: ocenia te same predykcje w Langfuse oceną `feedback` (1 = trafna, 0 = chybiona). Ludzie dodają tę samą ocenę w interfejsie Langfuse, a monitor liczy obie.
3. **`ml-pipeline`** przechodzi przez testy kodu, kontrakt danych, trening kandydata, bramki jakości i wydanie. Każdy etap zapisuje swoje oceny w Langfuse pod jednym śladem uruchomienia. Bramka porównuje kandydata z modelem produkcyjnym na tym samym, aktualnym zbiorze testowym: model produkcyjny jest oceniany na nim od nowa.

Rejestr modeli leży na serwerze w `~/mlops-ct-demo/registry`, więc wydanie przetrwa między uruchomieniami. `reset-registry` przywraca w nim wersję v1.

## Scenariusz na zajęcia

| Krok | Workflow i dane | Wynik |
| --- | --- | --- |
| 0 | `reset-registry` | produkcja = v1 |
| 1 | `monitor`, `labelled` = `data/incoming/batch_ok.csv` | AUC 0,787 ≥ 0,75: bez zmian |
| 1a | `review`, `labelled` = `data/incoming/batch_ok.csv` | 10 ocen testowych, średnia 0,8 |
| 2 | `monitor`, `labelled` = `data/incoming/batch_nov.csv`, `holdout` = `data/holdout_nov.csv` | AUC 0,734 < 0,75: monitor sam uruchamia `ml-pipeline` |
| 3 | `ml-pipeline` uruchomiony przez monitor | kandydat 0,785 wobec produkcji 0,728 na tym samym zbiorze: wydanie v2 |
| 4 | `monitor`, `labelled` = `data/holdout_nov.csv` | AUC 0,785 ≥ 0,75: bez zmian |

Dane listopadowe mają dryf pojęć: po zmianie cennika klienci planu basic przestali odchodzić, a każde zgłoszenie do supportu waży dwa razy więcej. Na starym zbiorze testowym nowy kandydat wypadłby gorzej (0,804 wobec 0,823) i bramka by go zatrzymała. Dlatego po dryfie oceniamy oba modele na świeżym zbiorze z nowego okresu.

Partie z referatu (`batch_broken.csv`, `batch_shuffled_labels.csv`) dalej zatrzymują `ml-pipeline` na kontrakcie albo na bramce jakości.

## Oceny w Langfuse

| Ślad | Oceny |
| --- | --- |
| `production-monitoring` | `production_auc`, a gdy są oceny predykcji także `feedback_mean`, `feedback_count` |
| `prediction` (`pred-<wersja>-<klient>`) | `feedback` od testowego recenzenta i od ludzi |
| `ml-pipeline` | `contract_ok`, `contract_violations`, `train_rows`, `auc_candidate`, `auc_production`, `gates_passed`, `released`, `released_version` |

Identyfikator śladu to `gh-<numer uruchomienia>-<próba>`, a w metadanych jest link do uruchomienia w GitHub Actions. Każdy etap to osobny span w śladzie, z wejściem (partia, zbiór testowy, wyzwalacz) i wynikiem etapu. Ślad monitoringu ma spany `pomiar jakości` i `decyzja`.

Sesje: monitor i `ml-pipeline`, który uruchomił, są w jednej sesji `ct-gh-<numer uruchomienia monitora>-<próba>`, więc w zakładce **Sessions** widać całą pętlę od pomiaru do wydania. Predykcje do oceny są w sesji `predykcje-<wersja modelu>`.

### Ocena predykcji w interfejsie Langfuse

Ocena `feedback` ma konfigurację w projekcie (liczba od 0 do 1), więc w Langfuse: **Tracing → Traces**, filtr nazwy `prediction`, otwórz ślad i kliknij **Annotate**, wybierz `feedback` i wpisz wartość. Wejście śladu to cechy klienta, wyjście to prawdopodobieństwo odejścia i predykcja, a w metadanych jest prawdziwa etykieta `actual_churn`. Następny `monitor` (najpóźniej po godzinie, albo od razu po ręcznym uruchomieniu) policzy tę ocenę razem z ocenami testowymi. Po 10 ocenach testowych ze średnią 0,8 wystarczą 4 ludzkie oceny 0, żeby średnia spadła do 0,57 i ruszył trening.

Przykład: na danych z listopada testowy recenzent da średnią 0,3 (model v1 myli się po dryfie), więc monitor uruchomi trening nawet wtedy, gdyby AUC było jeszcze powyżej progu.

## Konfiguracja

- Repozytorium jest publiczne, a runner wykonuje kod z repozytorium na serwerze. Dlatego workflowy startują ręcznie, po `push` na `main` albo (tylko `monitor`) z harmonogramu, nigdy z `pull_request`, workflowy z pull requestów od osób spoza repozytorium czekają na zatwierdzenie właściciela, a dozwolone są tylko akcje GitHuba (`actions/*`).
- Zmienna repozytorium `LANGFUSE_HOST` i sekrety `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (klucze projektu w Langfuse). Bez kluczy pipeline działa, a zapis do Langfuse jest pomijany.
- Harmonogram monitora (co godzinę, minuta 7) działa tylko przy zmiennej repozytorium `MONITOR_SCHEDULE=on`. Wyłączenie: `gh variable set MONITOR_SCHEDULE -b off`. Dane dla przebiegu z harmonogramu biorą się ze zmiennych `MONITOR_LABELLED` i `MONITOR_HOLDOUT` (domyślnie `data/incoming/batch_ok.csv` i `data/holdout.csv`). GitHub wyłącza harmonogramy w publicznym repozytorium po 60 dniach bez aktywności.
- Runner z etykietą `ai-os-server`. Środowisko Pythona tworzy `scripts/ci_env.sh` w `~/mlops-ct-demo/venv-<skrót requirements>` i używa go ponownie.

## Lokalnie

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python pipeline.py --reset && python pipeline.py data/incoming/batch_ok.csv   # pipeline z referatu, bez Langfuse
```

Dane generuje `scripts/make_data.py` (deterministycznie, ziarna w kodzie).
