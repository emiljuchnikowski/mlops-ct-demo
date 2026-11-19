# Ciągły trening w GitHub Actions z oceną w Langfuse

Demo do referatu 4 z MLOps („CI, CD i ciągły trening”), część 4. Model przewiduje rezygnację klientów (churn) na danych syntetycznych. Wszystko działa w GitHub Actions na runnerze na serwerze `ai-os-server`. Do Langfuse trafiają odpowiedzi modelu produkcyjnego i ich oceny, a wyniki treningu zostają w GitHub Actions.

Pętla ma trzy workflowy:

1. **`monitor`** działa co godzinę (i ręcznie). Model produkcyjny odpowiada na najnowszych danych z etykietami, a 10 jego odpowiedzi trafia do Langfuse. Monitor odczytuje z Langfuse oceny tych odpowiedzi i liczy u siebie AUC. Trening startuje, gdy co najmniej `feedback_min_count` (5) ocen ma średnią poniżej `feedback_min_mean` (0,6) albo AUC spadnie poniżej `monitor_min_auc` (0,75). Progi są w `gates.yaml`. Przebieg z harmonogramu nie uruchamia drugi raz tego samego treningu: dla tej samej wersji modelu, tych samych danych i tej samej liczby ocen czeka na nowe oceny albo dane.
2. **`review`** to testowy recenzent: ocenia odpowiedzi modelu w Langfuse oceną `feedback` (1 = trafna, 0 = chybiona). Ludzie dodają tę samą ocenę w interfejsie Langfuse, a monitor liczy obie.
3. **`ml-pipeline`** przechodzi przez testy kodu, kontrakt danych, trening kandydata, bramki jakości i wydanie. Wyniki etapów są w logach, komunikatach i artefaktach uruchomienia w GitHub Actions. Bramka porównuje kandydata z modelem produkcyjnym na tym samym, aktualnym zbiorze testowym: model produkcyjny jest oceniany na nim od nowa.

Rejestr modeli leży na serwerze w `~/mlops-ct-demo/registry`, więc wydanie przetrwa między uruchomieniami. `reset-registry` przywraca w nim wersję v1.

## Scenariusz na zajęcia

| Krok | Workflow i dane | Wynik |
| --- | --- | --- |
| 0 | `langfuse-reset`, potem `reset-registry` | czysty Langfuse (tylko dane dema), produkcja = v1 |
| 1 | `monitor`, `labelled` = `data/incoming/batch_ok.csv` | AUC 0,787 ≥ 0,75: bez zmian |
| 1a | `review`, `labelled` = `data/incoming/batch_ok.csv` | 10 ocen testowych predykcji v1, średnia 0,8 |
| 1b | `review`, `labelled` = `data/incoming/batch_nov.csv` | 10 ocen predykcji v1 na danych z listopada, średnia 0,3 |
| 2 | `monitor`, `labelled` = `data/incoming/batch_nov.csv`, `holdout` = `data/holdout_nov.csv` | AUC 0,734 < 0,75 i średnia ocen 0,55 < 0,6: monitor sam uruchamia `ml-pipeline` |
| 3 | `ml-pipeline` uruchomiony przez monitor | kandydat 0,785 wobec produkcji 0,728 na tym samym zbiorze: wydanie v2 |
| 4 | `monitor`, `labelled` = `data/holdout_nov.csv` | AUC 0,785 ≥ 0,75: bez zmian |

Dane listopadowe mają dryf pojęć: po zmianie cennika klienci planu basic przestali odchodzić, a każde zgłoszenie do supportu waży dwa razy więcej. Na starym zbiorze testowym nowy kandydat wypadłby gorzej (0,804 wobec 0,823) i bramka by go zatrzymała. Dlatego po dryfie oceniamy oba modele na świeżym zbiorze z nowego okresu.

Partie z referatu (`batch_broken.csv`, `batch_shuffled_labels.csv`) dalej zatrzymują `ml-pipeline` na kontrakcie albo na bramce jakości.

## Co jest w Langfuse

Tylko odpowiedzi modelu i ich oceny:

| Element | Zawartość |
| --- | --- |
| ślad `churn-prediction` (`pred-<wersja>-<klient>`) | wejście: cechy klienta; wyjście: prawdopodobieństwo odejścia i odpowiedź „odejdzie” albo „zostanie”; w metadanych prawdziwa etykieta `actual_churn` i link do uruchomienia w GitHub Actions |
| generacja `predict_proba` w śladzie | ta sama odpowiedź, z nazwą modelu `churn-<wersja>` |
| sesja `<wersja>-<partia danych>` | odpowiedzi jednej wersji modelu na jednej partii, np. `v1-batch_nov` |
| użytkownik `klient-<id>` | klient, którego dotyczy odpowiedź |
| ocena `feedback` | 1 = trafna, 0 = chybiona; od testowego recenzenta (`review`) i od ludzi |

`langfuse-reset` usuwa ślady z tagiem `mlops-ct-demo` i ich oceny, nic więcej. `langfuse-check` sprawdza klucze i pokazuje najnowszy ślad (sesja, wejście, wyjście, obserwacje).

### Ocena odpowiedzi w interfejsie Langfuse

Ocena `feedback` ma konfigurację w projekcie (liczba od 0 do 1), więc w Langfuse: **Sessions** albo **Tracing**, otwórz ślad `churn-prediction`, kliknij **Annotate**, wybierz `feedback` i wpisz wartość. Prawdziwa etykieta jest w metadanych `actual_churn`, więc łatwo sprawdzić, czy odpowiedź była trafna. Następny `monitor` (najpóźniej po godzinie, albo od razu po ręcznym uruchomieniu) policzy tę ocenę razem z ocenami testowymi. Po 10 ocenach testowych ze średnią 0,8 wystarczą 4 ludzkie oceny 0, żeby średnia spadła do 0,57 i ruszył trening.

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
