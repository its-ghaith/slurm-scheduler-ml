# RAPEC-G v1: task-generalisierter RAPEC-v3-PF

## Ziel und Abgrenzung

`RAPEC-G v1` überträgt die profilfreie Entscheidungslogik von `RAPEC-v3-PF`
auf iterative ML-Aufgaben. Der Controller erhält in jedem Checkpoint einen zu
maximierenden `quality_score` in `[0, 1]`, die Intervallenergie und die vorhandenen
Laufmetriken. Task-, Dataset-, Modell-, Scratch- und Pretrained-Namen beeinflussen
die Entscheidung nicht.

Die belastbare Aussage lautet:

> RAPEC-G v1 ist task-unabhängig für iterative Trainingsverfahren mit einer
> normierbaren Validierungsmetrik, messbaren Trainingsintervallen und einem
> speicherbaren Modellzustand.

Nicht abgedeckt sind geschlossene Trainingsdienste ohne Zwischenmetriken,
nicht-iterative Verfahren und Aufgaben ohne reproduzierbare Validierung.

## Identität mit RAPEC-v3-PF

RAPEC-G v1 erbt direkt von `ProfileFreeRapecV3Controller`. Prognose, dynamische
Gewinnschwelle, Effizienzquantil, Unsicherheitsprüfung, Warm-up, Patience und
Stop-Regel bleiben unverändert. Die Parameter in
`config/generalized-live-shadow-controllers.json` entsprechen dem normalen
`rapec-v3-pf` und nicht dem früheren Energy5-Profil.

Der Unterschied liegt ausschließlich in den Task-Adaptern außerhalb des
Controllers. Sie bilden fachliche Metriken auf `quality_score` ab und übersetzen
Trainingsschritte in ein gemeinsames Checkpoint-Protokoll.

Im Live-Shadow-Run beobachten nur zwei Verfahren dieselbe Full100-Lernkurve:

1. `standard-es-10` als klassische Early-Stopping-Baseline.
2. `rapec-g-v1` als generalisierter RAPEC-v3-PF-Controller.

## Einheitliche Qualitätsmetrik

| Aufgabentyp | Primärmetrik | Transformation zu `quality_score` |
|---|---|---|
| Klassifikation | Macro-F1 | `Q=M` |
| Anomalieerkennung | AUROC | `Q=M` |
| Regression und Forecasting | NRMSE | `Q=1/(1+M)` |
| Empfehlung | RMSE | `Q=1/(1+M)` |
| Sprachmodellierung | Cross-Entropy | `Q=exp(-M)` |
| Reinforcement Learning | normalisierter Evaluationsreward | `Q=M` |

Nur Validierungsdaten steuern die Stop-Entscheidung. Testdaten dienen nach dem
Training zur finalen Bewertung des gespeicherten besten Checkpoints.

## Versionierte Benchmark-Suiten

Die bisherige Suite `rapec-generalized-nonvision-v2` bleibt vollständig erhalten.
Sie kann weiterhin mit `-Suite Standard` erzeugt und ausgeführt werden. Die neue
Suite `rapec-generalized-nonvision-hard-v1` verwendet anspruchsvollere Daten und
ein schwierigeres Trainingsprotokoll. Alte Manifeste, Checkpoints und Ergebnisse
werden nicht überschrieben.

Die Hard-Suite enthält weiterhin acht Taskfamilien, drei Datasets beziehungsweise
Umgebungen pro Task und je eine Scratch- sowie Pretrained-Variante. Das ergibt
`8 × 3 × 2 = 48` reale Full100-Live-Shadow-Läufe.

| Task | Hard-v1-Datasets oder Umgebungen | Zusätzliche Schwierigkeit |
|---|---|---|
| Tabular-Klassifikation | Forest Covertype, Sensorless Drive, Letter Recognition | 7, 11 und 26 Klassen |
| Tabular-Regression | YearPredictionMSD, Online News Popularity, Superconductivity | größere, hochdimensionale oder stark verrauschte Ziele |
| Anomalieerkennung | KDD Cup 1999 Full, Covertype Minority, Sensorless Minority | seltene Klassen und heterogene Anomalien |
| Textklassifikation | IMDB, AG News, Yelp Review Full | bis 50.000 Trainingsbeispiele, 384 Tokens und fünf Klassen |
| Sprachmodellierung | WikiText-103, WikiText-2, Penn Treebank | längere Sequenzen, größeres Vokabular und mehr Tokens |
| Zeitreihenprognose | ETTh1-H24, ETTm1-H48, ETTm2-H96 | echte Mehrschrittprognose statt Ein-Schritt-Prognose |
| Empfehlung | MovieLens 1M, 10M, 20M | bis eine Million Ratings und größere Nutzer-/Item-Räume |
| Reinforcement Learning | CartPole Heavy, CartPole Noisy, MountainCar | schwere, stochastische oder sparse-reward Dynamik |

Bei Reinforcement Learning handelt es sich um Umgebungen und nicht um statische
Datasets.

## Scratch und Pretrained

`Scratch` bedeutet zufällige Modellinitialisierung. `Pretrained` ist ein
Source-Domain-Warmstart aus einem anderen Dataset derselben Taskfamilie. Es werden
nur formkompatible Gewichte geladen; die Ziel-Validierungs- und Testdaten werden
nicht zum Vortraining verwendet.

Die Hard-Suite speichert ihre 24 Source-Checkpoints getrennt unter:

```text
/workspace-cache/controller-pretrained-cross-domain-hard-v1
```

Die Vortrainingsenergie wird separat protokolliert und nicht der Energie des
Zieltrainings zugerechnet.

## Live-Shadow-Messung

Pro Fall läuft ein gemeinsames Full100-Training. Standard Early Stopping und
RAPEC-G beobachten dieselben Checkpoints und erzeugen virtuelle Stop-Zeitpunkte.
Die Controller-Sicht wird so berechnet:

```text
E_controller_view = E_training_to_stop + E_controller_overhead
```

Dataset-Download, Extraktion und Source-Pretraining liegen außerhalb des gemessenen
Zieltrainings. Dadurch verändert die größere Hard-Suite die Trainingsmessung nicht
durch einmalige Downloadkosten.

## Hard-v1 ausführen

Datasets einmalig im persistenten Rancher-Cache vorbereiten:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\prepare-cross-domain-assets-rancher.ps1 -Suite Hard
```

Source-Domain-Checkpoints einmalig asynchron erzeugen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\prepare-cross-domain-pretrained-models-rancher.ps1 -Suite Hard
```

Zuerst nur den 48-Fall-Plan prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-generalized-cross-domain-async.ps1 -Suite Hard -PlanOnly
```

Nach Abschluss des Source-Pretrainings den Benchmark starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-generalized-cross-domain-async.ps1 -Suite Hard
```

Nach Ausgabe der RunId kann der lokale Rechner ausgeschaltet werden. Status:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 -RunId <RUN_ID>
```

## Zentrale Dateien

- `generalized_manifest.py`: Standard- und Hard-v1-Matrix.
- `runners/cross_domain_assets.py`: persistente Dataset-Bereitstellung.
- `runners/cross_domain_tasks.py`: acht Task-Runner und Hard-Protokolle.
- `config/generalized-live-shadow-controllers.json`: Standard ES und RAPEC-G v1.
- `controllers/generalized_rapec_energy5.py`: kompatibler Dateiname des dünnen RAPEC-G-Adapters.

## Thesis-Dashboard: 48 reale plus 96 virtuelle Jobs

Die vollständige Thesis-Kampagne kombiniert die 18 Vision-Fälle mit den 30
Hardest-Cross-Domain-Fällen. Jeder der 48 Full100-Läufe wird genau einmal real
auf der GPU ausgeführt. Standard ES und RAPEC-G v4 beobachten diese Ausführung
online und werden anschließend als eigene virtuelle Jobs mit den Suffixen
`-es` und `-v4` dargestellt. Damit enthält das Dashboard 144 repräsentierte
Jobs, aber nur 48 physisch ausgeführte Trainingsjobs.

Plan und Voraussetzungen lokal prüfen, ohne etwas an Rancher zu senden:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-thesis-results-48-live-shadow-async.ps1 -PlanOnly
```

Die vollständige Kampagne an Rancher übergeben:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-thesis-results-48-live-shadow-async.ps1
```

Danach laufen Orchestrierung, Validierung, Wiederholungsversuche, Auswertung und
Prometheus-Publikation serverseitig weiter; der lokale Rechner kann ausgeschaltet
werden. Die öffentliche Statusseite ist
`https://speaks-var-syracuse-fantasy.trycloudflare.com/`. Jeder reale Job wird
nur akzeptiert, wenn 100 Epochendatensätze, CodeCarbon, SLURM-GPU-Messung, alle
drei Lebenszyklusphasen und beide Shadow-Controller vollständig vorliegen.

## Primärquellen

- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- ETT-Dataset: https://github.com/zhouhaoyi/ETDataset
- MovieLens-Datasets: https://grouplens.org/datasets/movielens/
- WikiText: https://huggingface.co/datasets/Salesforce/wikitext
- LC-PFN, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract.html
