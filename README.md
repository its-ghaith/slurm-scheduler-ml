# Rancher Branch: SLURM + MLflow + Prometheus + Grafana (Kubernetes)

Dieser Branch (`Rancher-sulrm`) enthaelt nur die Rancher/Kubernetes-Umgebung.
Ziel: reproduzierbarer End-to-End-Workflow fuer GPU-Training ueber SLURM mit Sichtbarkeit in MLflow und Grafana.

## Wissenschaftliche CARPK-Vergleichsstudie (Version 2)

Der finale Workflow trennt strikt zwischen **Validierung für Controllerentscheidungen** und **Testauswertung für die Ergebnisbewertung**. Jeder SLURM-Job trainiert genau einen Trainings-Seed; alle Strategien eines Szenarios verwenden denselben Split-Seed.

### Versuchsdesign

- Szenarien: `train128-scratch` und `train256-pretrained`
- Trainings-Seeds: `0, 1, 2, 3, 4`
- fester Split-Seed: `42`
- Strategien: `full100`, metrisches Standard-Early-Stopping, Delta-MAPE, konservativer Uncertainty-Controller, Fixed30 und Fixed50
- Umfang: `2 × 5 × 6 = 60` sequenzielle GPU-Jobs
- Zielmetrik aller Controller: Validierungs-`mAP50-95`
- Finale Qualitätsmetrik: einmalige Testauswertung von `best.pt`
- Reihenfolge: pro Szenario und Seed deterministisch randomisiert
- Cache-Policy: für alle Runs identisch, standardmäßig `ram`
- Cooldown: standardmäßig 30 Sekunden zwischen Runs
- Laufzeitinstallation: deaktiviert; fehlende Abhängigkeiten führen zum Abbruch

Die vorab festgelegte Matrix steht in `results/stop-policy-study/run_matrix.csv`.

### Strategien

1. `full100`: vollständiges Training bis Epoche 100.
2. `standard_early_stopping`: überwacht wie die eigenen Controller `mAP50-95` und nutzt ein explizites `min_delta`.
3. `delta_mape_controller`: stoppt bei dauerhaft geringem geglättetem Qualitätsgewinn und geringem marginalem Gewinn pro Netto-Wh.
4. `uncertainty_aware_controller`: darf erst nach Warm-up, Mindestqualität und gleichzeitig niedrigem MAPE stoppen; die Prognose nutzt 128 Bootstrap-Samples.
5. `fixed_epoch_30`: feste 30-Epochen-Baseline.
6. `fixed_epoch_50`: feste 50-Epochen-Baseline.

### Faire Messung

Die GPU-Leistung wird mit der Trapezregel integriert. Zusätzlich wird vor jedem Job die Idle-Leistung gemessen:

```text
gross GPU energy = integral(P_gpu(t) dt)
idle energy      = P_idle * duration
net GPU energy   = max(0, gross GPU energy - idle energy)
MAPE             = delta(mAP50-95) / net epoch energy in Wh
```

Getrennt berichtet werden:

- Epochenenergie
- Trainings-/Phasenenergie
- vollständige Jobenergie einschließlich Overhead
- Brutto- und Netto-GPU-Energie
- CodeCarbon-GPU- und CodeCarbon-Gesamtenergie

Pro Job werden außerdem GPU-Modell, Treiber, Power-Limit, Takte, Temperatur, andere GPU-Prozesse, Kubernetes-/SLURM-Knoten, Paketversionen und die tatsächlich laufende Container-Image-ID gespeichert. Die Auswertung warnt, wenn Split-Seed, Cache, GPU oder Image-Digest zwischen Runs abweichen.

### Studie planen und ausführen

Nur Matrix erzeugen und prüfen, ohne Jobs zu starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-stop-policy-comparison.ps1 -PlanOnly
```

Vollständige Studie starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-stop-policy-comparison.ps1
```

Vor der vollständigen Ausführung muss das Runtime-Image alle Abhängigkeiten enthalten. Für die finale Studie sollte `GHCR_IMAGE` auf einen unveränderlichen `@sha256:...`-Digest zeigen; jeder Run zeichnet zusätzlich die reale `imageID` des Pods auf.

Das Forschungsimage wird aus `Dockerfile.runtime` und `containers/runtime-requirements.txt` gebaut. Der GitHub-Workflow `.github/workflows/build-runtime-image.yml` veröffentlicht `research-v2`, einen Commit-Tag und zeigt anschließend den unveränderlichen Digest in der Workflow-Zusammenfassung. Dieser neue Digest muss vor der Studie in allen Runtime-Image-Feldern von `rancherConfigs/slurm-stack.yaml` eingetragen werden.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\set-rancher-runtime-image.ps1 `
  -Image "ghcr.io/its-ghaith/slurm-scheduler-ml/mlops-slurm-runtime@sha256:<64-hex-digest>" `
  -Apply
```

### Delta-MAPE-Sensitivitätsanalyse

Die Profile `aggressive`, `balanced` und `conservative` variieren Mindestepochen, Patience, Glättungsfenster, Delta- und MAPE-Schwelle:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-delta-mape-sensitivity.ps1 -PlanOnly
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-delta-mape-sensitivity.ps1
```

### Ergebnisse herunterladen und analysieren

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\analyze-rancher-stop-policy-study.ps1
```

Erzeugte Dateien unter `results/stop-policy-study/analysis`:

- `runs.csv`: ein Datensatz pro Job
- `aggregates.csv`: Mittelwert, Standardabweichung und 95%-Student-t-Konfidenzintervall pro Szenario/Strategie
- `pue_sensitivity.csv`: Kosten-/CO₂-Sensitivität für PUE `1.0`, `1.2`, `1.4`
- `study_summary.json`: maschinenlesbare Gesamtauswertung und Provenance-Prüfung
- `pareto_front.png`: mAP50-95 gegen GPU-Energie

Zusätzlich werden berechnet:

- Accuracy Regret gegenüber `Full100` desselben Szenarios und Seeds
- Energy-to-Target für Validierungs-mAP50-95 `50 %`, `60 %` und `68 %`
- Accuracy pro Wh
- Pareto-Dominanz

### Grafana-Forschungsbereich

Das Dashboard bietet Filter für Szenario, Strategie, Trainings-Seed und Job-ID. Neue Panels zeigen:

- Test-mAP50-95 des besten Checkpoints
- Accuracy Regret gegenüber Full100
- Pareto-Dominanz
- Brutto- und Netto-Jobenergie
- Energy-to-Target bei 50 %, 60 % und 68 % mAP50-95

`slurm/export_study_metrics_prom.py` berechnet diese Werte nach jedem Job über alle bis dahin vorhandenen Runs neu. Dadurch erscheinen auch Regret-Werte korrekt, wenn die randomisierte `Full100`-Baseline erst später abgeschlossen wird.

### Getrennte historische Vergleichsdashboards

Grafana stellt die gesicherten Vergleichsstudien als eigenständige Dashboards im selben Ordner bereit:

```text
Dashboards
└── SLURM Energy
    ├── 7 Run
    ├── 60 Run
    └── 9 Run - Complete Re-Run
```

- `7 Run` übernimmt unverändert die Panelstruktur der ursprünglichen 7-Job-Studie mit 37 Panels.
- `60 Run` übernimmt unverändert die Panelstruktur der Multi-Seed-Studie mit 45 Panels.
- `9 Run - Complete Re-Run` zeigt den vollständig neu ausgeführten Neun-Job-Vergleich mit SLURM-IDs 1 bis 9. Die X-Achsen der jobbasierten Panels und die Job-Auswahl werden numerisch aufsteigend sortiert.
- Die Prometheus-Zeitreihen erhalten intern `comparison_set="7_run"`, `comparison_set="60_run"` oder `comparison_set="8_run"`. Dadurch bleiben die Sicherungen strikt getrennt.
- Die Sicherungsordner unter `D:\Masterarbeit_Vergleichssicherung` werden ausschließlich gelesen und nicht verändert.

Beide Dashboards und beide gesicherten Metriksätze werden mit einem Befehl wiederhergestellt:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-rancher-comparison-dashboards.ps1
```

Bei abweichenden Sicherungspfaden können `-SevenRunSnapshot` und `-SixtyRunSnapshot` übergeben werden. Das Skript validiert die erwarteten 7 beziehungsweise 60 Jobs, erzeugt getrennte Dashboard-Kopien, lädt beide Metriksätze in den Node Exporter und provisioniert Grafana neu. Wegen der vollständigen historischen Epochenreihen verwendet der `slurmd-node-exporter` in Prometheus ein Intervall von 60 Sekunden und ein Timeout von 55 Sekunden.

Grafana kann anschließend per Port-Forward geöffnet werden:

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/grafana 3000:3000
```

URLs:

- Ordner: `http://localhost:3000/dashboards/f/dfs2ifgo1acqod/slurm-energy`
- 7 Runs: `http://localhost:3000/d/slurm-energy-7-run/7-run`
- 60 Runs: `http://localhost:3000/d/slurm-energy-60-run/60-run`
- 9 Runs, vollständiger Neuvergleich: `http://localhost:3000/d/slurm-energy-8-run/9-run-complete-re-run`

### Direkter Vergleich: alter gegen verbesserten Uncertainty-Controller

Dieser reproduzierbare Ablauf stellt zuerst die historische 7-Job-Studie wieder her und ergänzt nur einen neuen Lauf:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-improved-uncertainty-comparison.ps1
```

Die ersten sieben Ergebnisse werden aus `CARPK_StopPolicy_7Jobs_2026-07-13` gelesen und nicht verändert. Der zusätzliche Lauf verwendet dieselben zentralen Bedingungen wie die ursprüngliche Studie:

- CARPK `train128-scratch`
- YOLOv8n aus `yolov8n.yaml`, ohne Pretraining
- 128 Trainingsbilder, Batchgröße 8 und Bildgröße 640
- Training-Seed und Split-Seed 0
- maximal 100 Epochen
- Überwachungsmetrik mAP50-95

Der verbesserte Controller ergänzt eine Seed-0-Holdout-Kalibrierung, Mindestepoche 40, Mindestqualität 0,60, maximal 1,5 Prozentpunkte prognostizierten Regret und drei bestätigte Entscheidungen. Die Prognose läuft nur alle drei Epochen und mit acht Bootstrap-Samples, damit der Controller-Overhead die Energieeinsparung nicht aufhebt.

Der Lauf vom 18.07.2026 ist separat gespeichert unter:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_8Jobs_ImprovedUncertainty_2026-07-18_190704
```

Ergebnis des direkten Vergleichs:

| Strategie | Epochen | mAP50-95 | GPU-Energie | Dauer |
|---|---:|---:|---:|---:|
| Full100 | 100 | 70,81 % | 13,57 Wh | 522,03 s |
| Alter Uncertainty-Controller | 20 | 42,96 % | 3,41 Wh | 137,16 s |
| Verbesserter Uncertainty-Controller | 100 | 71,12 % | 13,44 Wh | 521,09 s |

Der verbesserte Controller beseitigt den schädlichen Frühstopp und verursacht durch die periodische Auswertung keinen sichtbaren Laufzeit- oder Energieoverhead. In dieser konservativen Konfiguration spart er jedoch keine Epochen. Das Ergebnis zeigt daher einen sicheren Controller, aber noch keinen besseren Accuracy-Energy-Kompromiss als Full100.

### Hybrid-Pareto-Controller als Job 10

Der nächste Controller wird ohne Reset zu demselben Vergleich hinzugefügt:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-hybrid-controller-job10.ps1
```

Der Modus `hybrid_pareto_energy_aware` kombiniert die miteinander vereinbaren Verbesserungen in einer zweistufigen Entscheidung:

1. Vor Mindestepoche 55 und vor dem Qualitätsziel `mAP50-95 >= 0,70` ist ein Stopp ausgeschlossen.
2. Danach werden best-so-far-mAP50-95, robuste Theil-Sen-Plateauerkennung, Netto-Energie, adaptive Accuracy-per-Wh-Grenze, Bootstrap-Prognose und Seed-0-Holdout-Conformal-Kalibrierung gemeinsam bewertet.
3. Ein Pareto-Nutzen vergleicht den prognostizierten Qualitätsgewinn mit der dafür noch erforderlichen Energie.
4. Stop-Evidenz wird über die Zustände `continue`, `observe`, `candidate_stop` und `stop` gesammelt. Eine einzelne schwankende Epoche reduziert die Evidenz nur, statt sie vollständig zu löschen.
5. Die Patience sinkt erst in den letzten 10 % des Epochenbudgets von drei auf zwei Bestätigungen.
6. Ein Epochenbudget von 95 ist nur nach Erreichen des Qualitätsziels als Sicherheitsgrenze aktiv.

Die Parameter wurden vor dem GPU-Lauf offline auf den vollständigen historischen Lernkurven geprüft. Die Conformal-Kalibrierung enthält keine Kurve mit Training-Seed 0; dadurch wird Job 10 nicht mit seiner eigenen Evaluationskurve kalibriert.

Ergebnis des 7+2-Vergleichs vom 19.07.2026:

| Strategie | Epochen | mAP50-95 | GPU-Energie | Dauer |
|---|---:|---:|---:|---:|
| Full100, Job 1 | 100 | 70,81 % | 13,57 Wh | 522,03 s |
| Alter Uncertainty-Controller, Job 4 | 20 | 42,96 % | 3,41 Wh | 137,16 s |
| Konservativer Conformal-Controller, Job 9 | 100 | 71,12 % | 13,44 Wh | 521,09 s |
| Hybrid-Pareto-Controller, Job 10 | **91** | **70,98 %** | **12,85 Wh** | **506,60 s** |

Job 10 spart gegenüber Full100 neun Epochen, rund 5,31 % Brutto-GPU-Energie und 2,96 % Laufzeit. Seine finale mAP50-95 ist gleichzeitig 0,18 Prozentpunkte höher; dieser kleine Unterschied liegt bei einem einzelnen Seed im Bereich möglicher Trainingsvariation und ist noch kein statistischer Qualitätsgewinn. Vergleicht man stattdessen die besten Validierungscheckpoints, liegt Job 10 mit 70,98 % um 0,35 Prozentpunkte unter Full100 mit 71,33 %. Gegenüber Job 9 spart der Hybridcontroller rund 4,41 % GPU-Energie bei lediglich 0,13 Prozentpunkten niedrigerer finaler mAP50-95.

Der neue Zustand ist separat gesichert unter:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_9Jobs_HybridController_2026-07-19_135356
```

Der historische 7+2-Zustand kann weiterhin aus seinem separaten Snapshot wiederhergestellt werden. Bei dieser Wiederherstellung heißt das Dashboard mit UID `slurm-energy-8-run` `9 Run - Uncertainty Controllers`:

```text
http://localhost:3000/d/slurm-energy-8-run/9-run-uncertainty-controllers
```

### Vollständiger Neun-Job-Vergleich

Der komplette Vergleich kann mit Reset und neuen SLURM-IDs 1 bis 9 reproduziert werden:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-complete-9run-comparison.ps1
```

Das Skript leert die aktiven Vergleichsdaten, startet alle neun Strategien sequenziell, wartet auf jeden Abschluss, sichert die Ergebnisse separat und provisioniert `9 Run - Complete Re-Run`. Die historischen 7- und 60-Run-Snapshots werden nicht verändert.

Frisches Ergebnis vom 19.07.2026:

| Job | Strategie | Epochen | mAP50-95 | GPU-Energie | Dauer |
|---:|---|---:|---:|---:|---:|
| 1 | Full100 | 100 | 71,12 % | 14,62 Wh | 569,65 s |
| 2 | Standard Early Stopping | 100 | 71,12 % | 14,49 Wh | 568,21 s |
| 3 | Delta-MAPE | 100 | 71,12 % | 14,54 Wh | 565,08 s |
| 4 | Alter Uncertainty-Controller | 24 | 35,81 % | 4,61 Wh | 185,62 s |
| 5 | Fixed20 | 20 | 36,69 % | 3,99 Wh | 160,86 s |
| 6 | Fixed30 | 30 | 51,29 % | 5,29 Wh | 211,61 s |
| 7 | Fixed50 | 50 | 62,13 % | 7,97 Wh | 317,91 s |
| 8 | Verbesserter Conformal-Controller | 100 | 71,12 % | 14,91 Wh | 581,70 s |
| 9 | Hybrid-Pareto-Controller | 91 | 70,98 % | 13,57 Wh | 527,24 s |

Die aktive Sicherung liegt unter:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_9Jobs_CompleteRerun_2026-07-19_155800
```

Aktives Dashboard:

```text
http://localhost:3000/d/slurm-energy-8-run/9-run-complete-re-run
```

### Stufe A: Seed-Generalisation des Hybrid-Pareto-Controllers

Die Stufe-A-Studie vergleicht `Full100` und den Hybrid-Pareto-Controller paarweise für die Trainings-Seeds `0, 1, 2, 3, 4`. Split, CARPK-Daten, YOLOv8n-Scratch-Modell, Batchgröße, Bildgröße, Runtime-Image und Hardware bleiben konstant. Die Strategiereihenfolge wird zwischen den Seed-Paaren alterniert, um Reihenfolgeeffekte zu reduzieren.

Der vollständige Reset-, Trainings-, Sicherungs- und Dashboard-Ablauf wird mit einem Befehl ausgeführt:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-stage-a-generalisation.ps1
```

Ergebnis vom 19.07.2026:

| Seed | Full-Job | Hybrid-Job | Hybrid-Epochen | Best-mAP50-95-Regret | GPU-Energieeinsparung |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 2 | 91 | 0,52 pp | 7,22 % |
| 1 | 4 | 3 | 88 | 0,71 pp | 8,06 % |
| 2 | 5 | 6 | 91 | 0,25 pp | 7,06 % |
| 3 | 8 | 7 | 91 | 0,79 pp | 7,99 % |
| 4 | 9 | 10 | 97 | 0,22 pp | -0,53 % |

Über alle fünf Seeds stoppte der Controller adaptiv und hielt das Regret-Budget von 1,5 Prozentpunkten ein. Im Mittel sparte er `8,4 ± 2,94` Epochen, `5,96 ± 3,27 %` GPU-Energie und `5,43 ± 3,42 %` Laufzeit bei `0,50 ± 0,23` Prozentpunkten Best-mAP50-95-Regret. Vier von fünf Läufen stoppten aufgrund der eigentlichen Pareto-Evidenz und erzielten eine positive Energieeinsparung. Seed 4 zeigt eine wichtige Grenze: Der Budget-Fallback stoppte erst nach Epoche 97 und sparte zwar drei Epochen, benötigte wegen Laufzeit- und Leistungsschwankungen aber 0,53 % mehr GPU-Energie als Full100.

Snapshot:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StageA_HybridGeneralisation_10Jobs_2026-07-19_185006
```

Dashboard:

```text
http://localhost:3000/d/slurm-energy-stage-a/stage-a-seed-generalisation-full100-vs-hybrid-pareto
```

Die Paaranalyse steht in `reports/stage-a-paired-analysis.csv` und `reports/stage-a-paired-analysis.json` des Snapshots. Das Dashboard bietet Filter für Training-Seed, Strategie und Job sowie separate Diagramme für Qualität, Energie, Epochen, Laufzeit, Quality Regret und Einsparungen.

### Stufe B: Szenario-Generalisation des Hybrid-Pareto-Controllers

Stufe B prüft, ob der in Stufe A untersuchte Hybrid-Pareto-Controller ohne neue Schwellenkalibrierung auf veränderte CARPK-Trainingsszenarien übertragbar ist. Jedes Szenario besteht aus einem paarweisen Vergleich zwischen `Full100` und `hybrid_pareto_uncertainty_controller`. Training-Seed und Split-Seed bleiben jeweils `0`; verändert werden Trainingsdatenmenge, Initialisierung und Modellgröße.

| Szenario | Trainingsbilder | Modell | Initialisierung |
|---|---:|---|---|
| `train128-scratch-yolov8n` | 128 | YOLOv8n | Scratch |
| `train256-scratch-yolov8n` | 256 | YOLOv8n | Scratch |
| `trainfull-scratch-yolov8n` | vollständiger Trainingssplit | YOLOv8n | Scratch |
| `train128-pretrained-yolov8n` | 128 | YOLOv8n | Pretrained |
| `train256-pretrained-yolov8n` | 256 | YOLOv8n | Pretrained |
| `trainfull-pretrained-yolov8n` | vollständiger Trainingssplit | YOLOv8n | Pretrained |
| `train256-pretrained-yolov8s` | 256 | YOLOv8s | Pretrained |

Der vollständige Reset-, Trainings-, Sicherungs-, Analyse- und Dashboard-Ablauf wird mit einem Befehl ausgeführt:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-stage-b-generalisation.ps1
```

Ergebnis vom 19.07.2026:

| Szenario | Full-Job | Hybrid-Job | Hybrid-Epochen | Best-mAP50-95-Regret | GPU-Energieeinsparung |
|---|---:|---:|---:|---:|---:|
| `train128-scratch-yolov8n` | 1 | 2 | 97 | 0,000 pp | 0,14 % |
| `train256-scratch-yolov8n` | 4 | 3 | 97 | 0,131 pp | -0,62 % |
| `trainfull-scratch-yolov8n` | 5 | 6 | 97 | 0,000 pp | 2,52 % |
| `train128-pretrained-yolov8n` | 8 | 7 | 97 | 0,000 pp | 2,57 % |
| `train256-pretrained-yolov8n` | 9 | 10 | 97 | 0,105 pp | 0,95 % |
| `trainfull-pretrained-yolov8n` | 12 | 11 | 97 | 0,054 pp | 2,04 % |
| `train256-pretrained-yolov8s` | 13 | 14 | 97 | 0,007 pp | 2,78 % |

Alle sieben Controllerläufe hielten das Regret-Budget von 1,5 Prozentpunkten deutlich ein. Im Mittel wurden jedoch nur `3,0` Epochen, `1,48 ± 1,24 %` GPU-Energie und `1,51 ± 1,12 %` Laufzeit eingespart. Sechs von sieben Paaren erzielten eine positive Energieeinsparung. Im Szenario `train256-scratch-yolov8n` benötigte der Hybridlauf trotz drei weniger Epochen `0,62 %` mehr Energie; dies liegt innerhalb plausibler Laufzeit- und Leistungsschwankungen eines einzelnen Laufs.

Die wichtigste wissenschaftliche Erkenntnis ist, dass sämtliche Stopps durch den festen Budget-Fallback bei Epoche 97 ausgelöst wurden. Die Rate evidenzbasierter Stopps beträgt `0 %`, die Budget-Fallback-Rate `100 %`. Der Controller ist in diesen Szenarien daher sehr qualitätssicher, aber zu konservativ für eine starke Energieeinsparung. Stufe B belegt somit noch keine allgemeine Controllerüberlegenheit, sondern identifiziert den nächsten Forschungsbedarf: szenariounabhängige Normalisierung der Energieeffizienz, bessere Kalibrierung der Unsicherheit und adaptive statt absoluter Schwellen.

Snapshot:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StageB_ScenarioGeneralisation_14Jobs_2026-07-19_233518
```

Dashboard:

```text
http://localhost:3000/d/slurm-energy-stage-b/410827c
```

Die Paaranalyse steht in `reports/stage-b-paired-analysis.csv` und `reports/stage-b-paired-analysis.json` des Snapshots. Das Dashboard bietet Mehrfachfilter für Szenario, Strategie und Job sowie separate Darstellungen für Best-mAP50-95, GPU-Energie, Epochen, Laufzeit, Regret, Einsparungen und Stop-Arten.

### Wiederholung mit Mindestepoche 30 und Controllerprüfung nach jeder Epoche

Am 20. und 21.07.2026 wurden der isolierte Hybrid-Lauf, Stufe A und Stufe B mit einer gezielten
Controlleränderung wiederholt. Frühere Snapshots und Ergebnisse wurden dabei nicht überschrieben.

Geänderte Parameter:

| Parameter | Vorher | Neue Wiederholung |
|---|---:|---:|
| Früheste Controllerentscheidung | Epoche 55 | Epoche 30 |
| Controller-Auswertungsintervall | alle 3 Epochen | jede Epoche |
| Evidenz-Patience | 3 Bewertungen | 3 Bewertungen |
| Maximales Hybrid-Budget | Epoche 95 | Epoche 95 |

Die Patience bleibt bei drei Bewertungen. Da jetzt jede Epoche bewertet wird, entsprechen drei
aufeinanderfolgende Evidenztreffer drei Epochen statt bis zu neun Epochen. Die Mindestepoche 30
erlaubt eine frühere Entscheidung, erzwingt aber keinen Stop bei Epoche 30.

#### Isolierter 9-Run-Hybrid

Nur der Hybrid-Pareto-Lauf wurde auf Basis der unveränderten acht historischen Jobs neu ausgeführt.
Der neue Job 10 stoppte evidenzbasiert nach Epoche 89 und erreichte eine finale mAP50-95 von
`70,14 %`. Die gemessene GPU-Energie betrug `14,72 Wh`. Sie lag in diesem Einzelvergleich über der
historischen Full100-Referenz (`13,57 Wh`), weshalb daraus allein keine Energieüberlegenheit
abgeleitet wird.

Snapshot:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_9Jobs_HybridController_Min30Eval1_2026-07-20_215603
```

Dashboard:

```text
http://localhost:3000/d/slurm-energy-8-run/9-run-hybrid-min30-eval1
```

#### Neue Stufe A: fünf Trainings-Seeds

Stufe A wiederholt `Full100` gegen Hybrid-Pareto für die Trainings-Seeds `0` bis `4`. Der
CARPK-Split-Seed bleibt `0`, und Modell, Datenmenge, Hardware sowie alle Trainingsparameter außer
der Stop-Policy bleiben innerhalb jedes Paares gleich.

| Seed | Full-Job | Hybrid-Job | Hybrid-Epochen | Gesparte Epochen | Best-mAP50-95-Regret | GPU-Energieeinsparung |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 2 | 89 | 11 | 0,944 pp | 3,38 % |
| 1 | 4 | 3 | 82 | 18 | 1,082 pp | 13,26 % |
| 2 | 5 | 6 | 88 | 12 | 0,650 pp | 1,37 % |
| 3 | 8 | 7 | 88 | 12 | 0,787 pp | 4,03 % |
| 4 | 9 | 10 | 90 | 10 | 0,692 pp | 3,07 % |

Alle fünf Controllerläufe stoppten aufgrund der eigentlichen Pareto-Evidenz und nicht durch den
Budget-Fallback. Alle hielten das Regret-Budget von `1,5` Prozentpunkten ein und sparten positive
GPU-Energie. Im Mittel wurden `12,6 ± 2,8` Epochen, `5,02 ± 4,21 %` GPU-Energie und
`4,60 ± 4,07 %` Laufzeit eingespart. Der mittlere Best-mAP50-95-Regret betrug
`0,831 ± 0,161` Prozentpunkte.

Interpretation: Für das feste `train128-scratch-yolov8n`-Szenario verbessert die Prüfung nach jeder
Epoche die Reaktionsfähigkeit. Gegenüber der älteren Stage-A-Wiederholung stieg die Rate
evidenzbasierter Stopps von `80 %` auf `100 %`. Die Einsparung variiert jedoch deutlich zwischen
Seeds; mehrere Wiederholungen bleiben deshalb notwendig.

Snapshot:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StageA_HybridGeneralisation_Min30Eval1_10Jobs_2026-07-20_234033
```

Dashboard:

```text
http://localhost:3000/d/slurm-energy-stage-a/stage-a-seed-generalisation-full100-vs-hybrid-pareto
```

#### Neue Stufe B: sieben CARPK-Szenarien

Stufe B verwendet dieselben sieben Szenarien wie die frühere Szenario-Generalisation. Jeder
Hybrid-Lauf wurde mit seiner eigenen Full100-Referenz verglichen.

| Szenario | Full-Job | Hybrid-Job | Hybrid-Epochen | Best-mAP50-95-Regret | GPU-Energieeinsparung |
|---|---:|---:|---:|---:|---:|
| `train128-scratch-yolov8n` | 1 | 2 | 95 | 0,289 pp | -0,76 % |
| `train256-scratch-yolov8n` | 4 | 3 | 95 | 0,254 pp | 0,05 % |
| `trainfull-scratch-yolov8n` | 5 | 6 | 95 | 0,000 pp | 2,32 % |
| `train128-pretrained-yolov8n` | 8 | 7 | 95 | 0,000 pp | -0,63 % |
| `train256-pretrained-yolov8n` | 9 | 10 | 95 | 0,187 pp | 1,15 % |
| `trainfull-pretrained-yolov8n` | 12 | 11 | 95 | 0,317 pp | 2,32 % |
| `train256-pretrained-yolov8s` | 13 | 14 | 95 | -0,080 pp | 0,59 % |

Alle sieben Läufe hielten das Regret-Budget ein, aber alle stoppten ausschließlich am festen
Hybrid-Budget bei Epoche 95. Die evidenzbasierte Stopprate beträgt somit `0 %`, die
Budget-Fallback-Rate `100 %`. Im Mittel wurden genau `5` Epochen, aber nur `0,72 ± 1,18 %`
GPU-Energie und `0,12 ± 1,42 %` Laufzeit eingespart. Nur fünf von sieben Paaren hatten eine
positive Energieeinsparung.

Interpretation: Die Absenkung der Mindestepoche und die häufigere Berechnung reichen allein nicht
für eine Szenario-Generalisation. Die eigentlichen Evidenzschwellen bleiben für wechselnde
Datenmengen, Pretraining und Modellgröße zu konservativ oder nicht ausreichend normalisiert. Das
Qualitätsrisiko ist gering, der Energiegewinn aber ebenfalls gering. Der nächste wissenschaftliche
Schritt ist daher eine szenarioadaptive Kalibrierung, beispielsweise durch normierte
Accuracy-per-Energy-Grenzen, online kalibrierte Unsicherheitsintervalle oder ein relatives
Qualitätsziel statt fixer absoluter Schwellen.

Snapshot:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_StageB_ScenarioGeneralisation_Min30Eval1_14Jobs_2026-07-21_204649
```

Dashboard:

```text
http://localhost:3000/d/slurm-energy-stage-b/stage-b-scenario-generalisation-full100-vs-hybrid-pareto
```

Prometheus trennt die drei neuen Ergebnismengen über `comparison_set`: `8_run` enthält 9 Jobs,
`stage_a` enthält 10 Jobs und `stage_b` enthält 14 Jobs. Dadurch können die Dashboards gleichzeitig
angezeigt werden, ohne Metriken oder Jobnummern miteinander zu vermischen.

### Paarweise Conformal-Controller-Studie mit acht Läufen

Die neue Studie verändert weder die 7-Run- noch die 60-Run-Sicherung. Sie prüft den neuen
`conformal_energy_aware`-Controller mit zwei zuvor nicht verwendeten Trainings-Seeds (`10`, `11`):

- Szenarien: `train128-scratch` und `train256-pretrained`
- Strategien je Szenario und Seed: `full100` und `conformal_energy_aware_controller`
- Umfang: `2 Szenarien × 2 Seeds × 2 Strategien = 8 Jobs`
- Split-Seed: konstant `42`
- Controller-Metrik: Best-so-far Validierungs-mAP50-95
- Zulässiger Accuracy-Regret: `0,015`, also 1,5 Prozentpunkte
- Mindestqualität: `0,60` für Scratch und `0,78` für Pretrained
- Entscheidungspatience: drei aufeinanderfolgende Epochen

Der Oracle-Stopp für eine vollständige Lernkurve ist definiert als:

```text
t* = erste Epoche t mit best_mAP(T) - best_mAP(t) <= 0,015
```

Der Controller kennt diesen Oracle-Wert während des Trainings nicht. Er kombiniert stattdessen:

1. ein Ensemble aus asymptotischem Exponentialmodell, inverser Potenzkurve und lokalen Trends,
2. Moving-Block-Residual-Bootstrap zur Abbildung zeitlich abhängiger Lernkurvenschwankungen,
3. Split-Conformal-Kalibrierung aus historischen Full100-Kurven,
4. eine robuste Prognose der bis Epoche 100 verbleibenden Netto-GPU-Energie,
5. die drei Zustände `continue`, `observe` und `stop`.

Ein Stopp ist nur möglich, wenn gleichzeitig Mindestepoche, Mindestqualität, kalibrierte obere
Regret-Grenze, maximale Intervallbreite und die obere zukünftige Accuracy-per-Energy-Grenze erfüllt
sind. Fehlt die Kalibrierung oder ist die Prognose zu unsicher, läuft das Training sicher weiter.

Kalibrierung reproduzierbar neu erzeugen:

```powershell
python .\slurm\build_conformal_calibration.py `
  --input-dir "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125\data\energy_metrics" `
  --output .\slurm\conformal_calibration.json
```

Nur die 8-Run-Matrix prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-conformal-8run-study.ps1 -PlanOnly
```

Komplette Studie ausführen, separat sichern und alle drei Dashboards provisionieren:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-conformal-8run-study.ps1
```

Das Skript setzt die aktive Rancher-Messumgebung zurück, führt genau acht sequenzielle Jobs aus,
erstellt anschließend eine neue Sicherung unter `D:\Masterarbeit_Vergleichssicherung` und ruft das
Dashboard-Deployment mit diesem Snapshot auf. Die bestehenden Sicherungen werden nur gelesen.

Danach enthält der Grafana-Ordner:

```text
SLURM Energy
|-- 7 Run
|-- 8 Run - Conformal Controller
`-- 60 Run
```

Die zusätzlichen 8-Run-Panels zeigen pro Epoche:

- Controllerentscheidung (`0=continue`, `1=observe`, `2=stop`)
- rohe und konformal korrigierte obere Grenze des verbleibenden Accuracy-Gewinns
- prognostizierte verbleibende Trainingsenergie
- obere zukünftige Accuracy pro Wh
- verwendete Konformal-Korrektur

#### Ergebnis des validierten 8-Run-E2E-Tests vom 16.07.2026

| Szenario | Seed | Controller-Epochen | Full100 Test-mAP50-95 | Controller Test-mAP50-95 | Test-Regret | Epochenenergie-Ersparnis | Gesamtjobenergie-Ersparnis |
|---|---:|---:|---:|---:|---:|---:|---:|
| train128-scratch | 10 | 100 | 38,548 % | 38,548 % | 0,000 pp | -2,11 % | -8,33 % |
| train128-scratch | 11 | 88 | 38,923 % | 39,412 % | -0,488 pp | 11,53 % | -38,77 % |
| train256-pretrained | 10 | 65 | 58,517 % | 56,546 % | 1,971 pp | 37,74 % | 30,89 % |
| train256-pretrained | 11 | 72 | 58,735 % | 58,356 % | 0,379 pp | 27,21 % | 23,20 % |

Die negativen Scratch-Werte bei der Gesamtjobenergie sind ein wichtiges Ergebnis und werden nicht
ausgeblendet: Die Controllerberechnung liegt außerhalb der gemessenen YOLO-Epochenfenster. Bei Seed 11
spart der Frühstopp zwar 11,53 % reine Epochenenergie, aber der Prognose- und Bootstrap-Overhead erhöht
die gesamte Jobdauer und damit die Brutto-GPU-Energie. Für Pretrained amortisiert sich derselbe Overhead,
weil 28 beziehungsweise 35 Epochen entfallen. Deshalb müssen in der Arbeit immer Epochenenergie und
vollständige Jobenergie gemeinsam berichtet werden.

Der separate, unveränderliche Snapshot liegt unter:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_ConformalController_8Jobs_2026-07-16_181750
```

Reproduzierbare Auswertungen stehen unter `results/conformal-controller-study`, darunter Run-Matrix,
Kalibrierungs-Replay, gepaarte Ergebnisse, Aggregationen, PUE-Sensitivität und Provenance-Prüfung.

8-Run-Dashboard: `http://localhost:3000/d/slurm-energy-8-run/8-run-conformal-controller`

## Zielbild

Pro SLURM-Job soll es geben:
- genau **einen** MLflow-Run (`job-energy-<JOBID>`) mit Trainings- und Energie-Metriken
- Prometheus-Metriken fuer Dashboard/KPIs in Grafana

## Architektur

Komponenten im Namespace `mlops-energy`:
- `slurmctld` (Controller)
- `slurmd` (Worker mit GPU)
- `slurmd-node-exporter` (Textfile Collector Sidecar)
- `mlflow`
- `prometheus`
- `grafana`

Manifest:
- `rancherConfigs/slurm-stack.yaml`

Visualisierung:

![Workflow](/D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/docs_assets/workflow.png)

![Datenfluss](/D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/docs_assets/dataflow.png)

## Datenfluss

1. `sbatch` wird in `slurmctld` gestartet.
2. `slurmd` fuehrt Training auf GPU aus.
3. GPU-CSV + Summary-JSON werden in `/workspace/energy_metrics` erzeugt.
4. `export_job_metrics_prom.py` schreibt `job_<id>.prom` und `aggregate.prom`.
5. `slurmd-node-exporter` liest Textfiles, Prometheus scraped, Grafana visualisiert.
6. MLflow erhaelt Trainings- und Energie-Metriken im selben Run `job-energy-<JOBID>`.

## Relevante Dateien

- `scripts/setup_rancher.ps1`
- `.env.example`
- `rancherConfigs/slurm-stack.yaml`
- `slurm/train_mlflow_local.slurm`
- `slurm/log_job_energy_mlflow.py`
- `train_with_energy_tracking_mlflow.py`
- `slurm/gpu_energy_utils.py`
- `slurm/epoch_energy_controller.py`
- `slurm/summarize_gpu_metrics_epochs.py`
- `slurm/export_job_epoch_metrics_prom.py`
- `slurm/export_job_metrics_prom.py`
- `grafana/dashboards/slurm-energy-overview.json`
- `prometheus/prometheus.yml`

## Konfiguration

### 1) `.env` erstellen

```powershell
Copy-Item .env.example .env
```

`.env` wird nicht committed (`.gitignore`).

### 2) Kubeconfig lokal bereitstellen

Pfad:

`rancherConfigs/main.yaml`

Wenn die Datei fehlt, schlagen `bootstrap`, `submit` und `test` sofort fehl.

### 3) Pflicht-/Empfehlungswerte

```env
CR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GHCR_USER=<your-github-username>
GHCR_SERVER=ghcr.io
GHCR_IMAGE=ghcr.io/<your-github-username>/slurm-scheduler-ml/mlops-slurm-runtime:latest

ACTION=bootstrap
KUBECONFIG_PATH=rancherConfigs/main.yaml
K8S_NAMESPACE=<your-k8s-namespace>
K8S_MANIFEST=rancherConfigs/slurm-stack.yaml

TRAIN_CMD=python train_with_energy_tracking_mlflow.py
BATCH_SIZE=64
MLFLOW_TRACKING_URI=http://mlflow:5000
GPU_SAMPLE_INTERVAL_SEC=1
ELECTRICITY_PRICE_EUR_PER_KWH=0.30
GRID_CO2_KG_PER_KWH=0.40
PUE_FACTOR=1.0
MLFLOW_LOG_JOB_ENERGY=1
MLFLOW_JOB_ENERGY_EXPERIMENT=ml-energy-poc
```

## Betrieb

### A) Bootstrap

```powershell
./scripts/setup_rancher.ps1 -Action bootstrap
```

Macht:
- Namespace check
- GHCR Secret (`ghcr-pull-secret`) mit `CR_PAT`
- `kubectl apply` auf Manifest
- Rollout restart/wait fuer `mlflow`, `slurmctld`, `slurmd`
- Basis-Checks

### B) Training-Job submit

```powershell
./scripts/setup_rancher.ps1 -Action submit
```

Optional:

```powershell
./scripts/setup_rancher.ps1 -Action submit -TrainCmd "python train.py"
```

### C) Schnelltest

```powershell
./scripts/setup_rancher.ps1 -Action test
```

## Hilfsskripte

Die folgenden PowerShell-Skripte unter `scripts/` vereinfachen Reset, E2E-Test und manuelle Job-Einreichung auf Rancher.

### `scripts/reset-rancher-slurm-observability.ps1`

Zweck:
- bricht laufende oder wartende SLURM-Jobs ab
- leert Energy-Metriken, Prometheus-Textfiles und Logdateien im `slurmd`-Pod
- leert optional MLflow-Daten und Artefakte
- startet optional `slurmctld`, `slurmd`, `prometheus`, `grafana` und `mlflow` neu

Wichtige Parameter:
- `-Kubeconfig`: Pfad zur Rancher-Kubeconfig
- `-Namespace`: Ziel-Namespace, Standard `mlops-energy`
- `-SkipMlflowReset`: laesst MLflow-Daten unangetastet
- `-SkipRestart`: fuehrt nur Cleanup aus, ohne Rollout-Neustarts
- `-Force`: fuehrt das Reset ohne Rueckfrage aus

Beispiele:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reset-rancher-slurm-observability.ps1 -Force
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reset-rancher-slurm-observability.ps1 -SkipMlflowReset -SkipRestart -Force
```

### `scripts/run-rancher-e2e-job1.ps1`

Zweck:
- fuehrt einen kompletten Rancher-E2E-Test mit frischem Reset aus
- kopiert den Repro-Workspace in `slurmd` und `slurmctld`
- reicht einen SLURM-Job ein
- erwartet nach dem Reset bewusst `Job-ID 1`
- prueft danach Prometheus- und MLflow-Ergebnisse

Wichtige Parameter:
- `-Kubeconfig`
- `-Namespace`
- `-ExperimentName`
- `-Dataset`
- `-Model`
- `-Epochs`
- `-BatchSize`
- `-ImageSize`
- `-Patience`
- `-TimeoutSeconds`
- `-SkipDependencyInstall`

Beispiel:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-e2e-job1.ps1
```

Sinn des Skripts:
- reproduzierbarer Smoke-Test
- sauberer Startzustand fuer Grafana, Prometheus und MLflow
- schneller Nachweis, dass genau ein neuer Job alle Metriken erzeugt

### `scripts/submit-rancher-job.ps1`

Zweck:
- reicht einen neuen Rancher-SLURM-Job ein
- fuehrt im Gegensatz zum E2E-Skript **kein Reset** durch
- kann optional auf Job-Abschluss warten

Wichtige Parameter:
- `-Kubeconfig`
- `-Namespace`
- `-ExperimentName`
- `-Dataset`
- `-Model`
- `-Epochs`
- `-BatchSize`
- `-ImageSize`
- `-Patience`
- `-TimeoutSeconds`
- `-SkipDependencyInstall`
- `-WaitForCompletion`

Beispiele:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\submit-rancher-job.ps1
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\submit-rancher-job.ps1 -WaitForCompletion
```

Einsatzfall:
- weitere Jobs nach einem erfolgreichen Bootstrap starten
- Metriken fuer neue Jobs sammeln, ohne Prometheus, Grafana oder MLflow vorher zurueckzusetzen

## Webzugriff

Port-Forward manuell (separate Terminals):

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/grafana 3000:3000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/mlflow 5000:5000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/prometheus 9090:9090
```

UIs:
- Grafana: [http://localhost:3000](http://localhost:3000)
- MLflow: [http://localhost:5000](http://localhost:5000)
- Prometheus: [http://localhost:9090](http://localhost:9090)

## End-to-End Verifikation

1. Pods gesund:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy get pods -o wide
```

2. SLURM gesund:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "scontrol ping && sinfo -N -l"
```

3. Job submit + warten:
```powershell
./scripts/setup_rancher.ps1 -Action submit
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- squeue
```

4. Energieartefakte vorhanden:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmd -c slurmd -- bash -lc "ls -lah /workspace/energy_metrics && ls -lah /workspace/energy_metrics/node_exporter"
```

5. Prometheus Query liefert Werte:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_jobs_total'"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
```

6. MLflow: Experiment `ml-energy-poc` zeigt Run `job-energy-<JOBID>` mit:
- Training: `loss`, `duration_seconds`
- Energie: `training_energy_kwh`, `estimated_electricity_cost_eur`, `estimated_co2_kg`, etc.

## Troubleshooting

### MLflow "Die Website ist nicht erreichbar"
- Ursache meist lokaler Port-Forward.
- Neu starten:
```powershell
Get-Process kubectl -ErrorAction SilentlyContinue | Stop-Process -Force
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/mlflow 5000:5000
```
- Check:
```powershell
Invoke-WebRequest http://localhost:5000 -UseBasicParsing
```

### Grafana "No data"
- Prometheus Query direkt pruefen (`slurm_jobs_total`, `slurm_job_training_energy_kwh`).
- Zeitfenster in Grafana vergroessern (`Last 30 days`).
- `job_*.prom` und `aggregate.prom` im `slurmd` Pod pruefen.

### CR_PAT Fehler bei Bootstrap
- `.env` muss `CR_PAT` enthalten oder `CR_PAT` in Shell setzen.
- Secret bei Bedarf neu erzeugen (`bootstrap` macht das automatisch).

### Zwei Experimente / zwei Runs pro Job
- Zielzustand ist **ein Experiment** `ml-energy-poc` und **ein Run** `job-energy-<JOBID>` pro Job.
- Altes Experiment `slurm-job-energy` wurde auf `deleted` gesetzt.

## Wichtiger Hinweis zum Runtime-Image

Wenn Pods ein aelteres Runtime-Image nutzen, koennen Skript-Aenderungen lokal nicht sofort im Cluster aktiv sein.
Fuer dauerhafte Wirkung:
1. Image neu bauen/pushen.
2. `rancherConfigs/slurm-stack.yaml` mit neuem Tag aktualisieren.
3. `./scripts/setup_rancher.ps1 -Action rollout` oder `bootstrap`.

## Branch-Policy

- Dieser Branch ist strikt Rancher/Kubernetes.
- Local-Docker-Setup liegt in `local-sulrm`.
- `.env` nie committen, nur `.env.example`.
- `rancherConfigs/main.yaml` bleibt lokal/ignoriert.

## Reproduzierbares Retraining von `rotationally-invariant-cnns` mit Phasen-Energie-Tracking

Ohne Aenderungen im Originalordner `rotationally-invariant-cnns`:
- Runner: `slurm/train_repro_phase_tracked.py`
- SLURM Job: `slurm/train_repro_rotacnn_job.slurm`
- Phase-Summary: `slurm/summarize_gpu_metrics_phases.py`
- Prometheus Export (Phasen): `slurm/export_job_phase_metrics_prom.py`

## Energieadaptives Training pro Epoche

Der Rancher-Workflow misst jetzt waehrend YOLO-Training nicht nur Job- und Phasen-Energie, sondern auch pro Epoche:

- `mAP50`
- `mAP50-95`
- `Precision`
- `Recall`
- SLURM-GPU-Energie
- SLURM-Gesamtenergie (GPU * PUE)
- `MAPE = delta(mAP50) / delta(Wh)`

Wichtig:
- Die adaptive Stop-Entscheidung verwendet dieselbe SLURM-GPU-Energie-Integrationslogik wie die spaetere Offline-Auswertung.
- Dadurch bleibt der Vergleich zwischen Epochenmetriken und finalen Summary-Dateien konsistent.
- Der adaptive Modus ist standardmaessig **aus**, das Epochen-Tracking jedoch **an**, sobald `slurm/train_repro_rotacnn_job.slurm` genutzt wird.

### Neue Artefakte pro Job

Im `slurmd`-Pod unter `/workspace/energy_metrics`:

- `epoch_timeline_job_<JOBID>.jsonl`
- `epoch_summary_job_<JOBID>.json`

Im Node-Exporter-Textfile-Verzeichnis:

- `job_<JOBID>_epochs.prom`

### Neue Prometheus-Metriken

Beispiele:

- `slurm_job_epoch_energy_kwh`
- `slurm_job_epoch_gpu_energy_kwh`
- `slurm_job_epoch_map50`
- `slurm_job_epoch_precision`
- `slurm_job_epoch_recall`
- `slurm_job_epoch_delta_map50`
- `slurm_job_epoch_mape_map50_per_wh`
- `slurm_job_epoch_should_stop`

### Baseline vs. adaptiver Lauf

Baseline:
- `ENERGY_ADAPTIVE_ENABLED=false`
- Training laeuft mit der regulären Epochenzahl
- trotzdem werden Epochenmetriken fuer die spaetere Auswertung gespeichert

Adaptiv:
- `ENERGY_ADAPTIVE_ENABLED=true`
- nach jeder Epoche werden `delta(mAP50)` und `MAPE` berechnet
- wenn geglaettetes `delta(mAP50)` und geglaettetes `MAPE` unter die konfigurierten Schwellwerte fallen, stoppt das Training frueh

### Konfigurierbare adaptive Parameter

Im SLURM-Job bzw. ueber Rancher-Submit-Skripte:

- `ENERGY_ADAPTIVE_ENABLED`
- `ENERGY_ADAPTIVE_MIN_EPOCHS`
- `ENERGY_ADAPTIVE_PATIENCE`
- `ENERGY_ADAPTIVE_SMOOTHING_WINDOW`
- `ENERGY_ADAPTIVE_MIN_DELTA_MAP50`
- `ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH`

### Job ueber Rancher-Hilfsskript einreichen

Baseline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\submit-rancher-job.ps1 `
  -Dataset carpk `
  -Model yolov8 `
  -Epochs 100 `
  -BatchSize 8 `
  -Patience 20
```

Adaptiv:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\submit-rancher-job.ps1 `
  -Dataset carpk `
  -Model yolov8 `
  -Epochs 100 `
  -BatchSize 8 `
  -Patience 20 `
  -AdaptiveEnabled `
  -AdaptiveMinEpochs 20 `
  -AdaptivePatience 3 `
  -AdaptiveSmoothingWindow 3 `
  -AdaptiveMinDeltaMap50 0.001 `
  -AdaptiveMinMapeMap50PerWh 0.0001 `
  -WaitForCompletion
```

### Forschungslogik

Damit lassen sich jetzt direkt zwei Trainingsmodi vergleichen:

1. fester Trainingsplan, z. B. `100` Epochen
2. energieadaptiver Trainingsplan mit dynamischem Stopp

Die Vergleichswerte liegen danach gemeinsam in:

- MLflow
- Prometheus
- Grafana
- `epoch_summary_job_<JOBID>.json`

Damit koennt ihr fuer die Masterarbeit systematisch auswerten:

- finale Genauigkeit
- Trainingsdauer
- verbrauchte Energie
- marginale Genauigkeitsgewinne pro Wattstunde

### Submit Beispiel

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "REPRO_DATASET=carpk REPRO_MODEL=yolov8 REPRO_OVERRIDES='experiment=repro_carpk_y8;dataset.train_size=500;model.epochs=100;model.patience=8;model.batch_size=8;dataset.img_size=1280,720' sbatch /workspace/slurm/train_repro_rotacnn_job.slurm"
```

`REPRO_OVERRIDES` Format:
- Semikolon-getrennt: `key=value;key2=value2`
- Beispiel: `dataset.path=/workspace/rotationally-invariant-cnns/data/carpk`

### Neue Metriken

Im erweiterten Summary (`gpu_summary_job_<id>_phases.json`):
- `phase_metrics.<phase>.gpu_energy_kwh`
- `phase_metrics.<phase>.total_energy_kwh`
- `phase_metrics.<phase>.codecarbon_energy_kwh`
- `phase_metrics.<phase>.codecarbon_gpu_energy_kwh`
- `codecarbon_vs_slurm.training_abs_diff_kwh`
- `codecarbon_vs_slurm.training_rel_diff_pct`
- `codecarbon_vs_slurm.compare_basis`
- `codecarbon_vs_slurm.training_gpu_abs_diff_kwh`
- `codecarbon_vs_slurm.training_gpu_rel_diff_pct`
- `codecarbon_vs_slurm.training_total_abs_diff_kwh`
- `codecarbon_vs_slurm.training_total_rel_diff_pct`

In Prometheus (zusaeztlich):
- `slurm_job_phase_energy_kwh{job_id,phase}`
- `slurm_job_phase_gpu_energy_kwh{job_id,phase}`
- `slurm_job_phase_codecarbon_energy_kwh{job_id,phase}`
- `slurm_job_phase_codecarbon_gpu_energy_kwh{job_id,phase}`
- `slurm_job_training_codecarbon_energy_kwh{job_id}`
- `slurm_job_training_codecarbon_gpu_energy_kwh{job_id}`
- `slurm_job_training_energy_compare_abs_diff_kwh{job_id}`
- `slurm_job_training_energy_compare_rel_diff_pct{job_id}`
- `slurm_job_training_gpu_energy_compare_abs_diff_kwh{job_id}`
- `slurm_job_training_gpu_energy_compare_rel_diff_pct{job_id}`
- `slurm_job_training_total_energy_compare_abs_diff_kwh{job_id}`
- `slurm_job_training_total_energy_compare_rel_diff_pct{job_id}`

## Faire Vergleichslogik: CodeCarbon vs. SLURM

Die Phasenabgrenzung fuer beide Systeme ist identisch:
- Start und Ende jeder Phase werden in `tracked_phase(...)` in `slurm/train_repro_phase_tracked.py` geschrieben
- dieselben Zeitstempel werden spaeter fuer die SLURM/GPU-Auswertung in `slurm/summarize_gpu_metrics_phases.py` verwendet

Dadurch messen beide Systeme denselben Codeabschnitt fuer:
- `preprocessing_labels`
- `preprocessing_split`
- `preprocessing_crop`
- `training`

Wichtige Klarstellung zur Fairness:
- SLURM/GPU basiert auf integrierter GPU-Leistung aus `gpu_monitor.sh`
- CodeCarbon liefert sowohl Gesamtenergie als auch GPU-Energie

Der eigentliche Vergleich `training_abs_diff_kwh` und `training_rel_diff_pct` verwendet jetzt bewusst:
- SLURM: `gpu_energy_kwh`
- CodeCarbon: `codecarbon_gpu_energy_kwh`

Also:
- gleicher Codeabschnitt
- gleiche Zeitgrenzen
- gleiche Vergleichsbasis: GPU gegen GPU

Zusaetzlich werden weiterhin Vergleichswerte fuer Gesamtenergie exportiert, damit man GPU-vs-GPU und Total-vs-Total getrennt analysieren kann.

Wichtige Dashboard-Klarstellung:
- `CodeCarbon Phase Total Energy by Job (Wh)` zeigt die gesamte CodeCarbon-Energie der Phase
- diese Gesamtenergie kann deutlich groesser sein als `codecarbon_gpu_energy_kwh`, weil CodeCarbon auch CPU- und RAM-Anteile beruecksichtigt
- Beispiel Training:
  - CodeCarbon Total: `codecarbon_energy_kwh`
  - CodeCarbon GPU-only: `codecarbon_gpu_energy_kwh`
  - der faire Differenz-Chart `Training GPU Energy Diff % (CodeCarbon GPU vs SLURM GPU)` nutzt die GPU-only-Werte

## Wichtige Code-Aenderungen

### `slurm/export_job_metrics_prom.py`

Hier wurde die Aggregation so korrigiert, dass Dateien vom Typ `gpu_summary_job_<id>_phases.json` nicht als eigenstaendige Jobs mitgezaehlt werden.

Nutzen:
- `slurm_jobs_total` zaehlt echte Jobs korrekt
- Grafana-KPIs wie `Jobs Total` und aggregierte Summen werden nicht durch Phase-Summaries verfaelscht

### `slurm/train_repro_phase_tracked.py`

Hier wurden drei praktische Verbesserungen ergaenzt:
- direkte Nutzung von `MLFLOW_TRACKING_URI`, wenn eine HTTP/HTTPS-URL gesetzt ist
- Normalisierung von `seeds`, falls nur ein einzelner Wert uebergeben wird
- eindeutige Run-Namen pro Seed bei Multi-Seed-Laeufen

Nutzen:
- stabileres MLflow-Logging in Rancher
- weniger Fehler bei Konfigurationsvarianten
- nachvollziehbare Runs wie `job-energy-1-seed-0`, falls mehrere Seeds trainiert werden

##

Dieser Abschnitt dokumentiert, was konkret umgesetzt wurde, welche Probleme auftraten und wie der finale stabile Zustand aussieht.

### 1) Umgesetzter Plan

1. Reproduzierbaren Retrain-Workflow fuer `rotationally-invariant-cnns` aufgebaut, ohne den Originalordner zu aendern.
2. Tracking entlang des gesamten Flows integriert: SLURM Job -> GPU/Energie-Summaries -> Prometheus -> Grafana sowie MLflow-Runs.
3. Energieverbrauch in Phasen getrennt erfasst (Training vs. Preprocessing) und Vergleich CodeCarbon vs. SLURM ermoeglicht.
4. Wiederholbare "Reset auf Null"-Schritte fuer MLflow, Grafana und Energy-Metrics eingefuehrt.

### 2) Was technisch gemacht wurde

- Repro-Runner und SLURM-Job fuer den getrennten Stage-Ansatz verwendet:
  - `_stage_rotacnn/`
  - `slurm/train_repro_rotacnn_job.slurm`
- Phasenbasierte Auswertung/Export umgesetzt:
  - `slurm/summarize_gpu_metrics_phases.py`
  - `slurm/export_job_phase_metrics_prom.py`
- Dashboarding fuer neue Phasenmetriken erweitert:
  - `grafana/dashboards/slurm-energy-overview.json`
- Zusaetzliches Dashboard erstellt, das bei Inaktivitaet auf 0 faellt:
  - `grafana/dashboards/slurm-energy-overview-zero-on-idle.json`

### 3) Wichtige aufgetretene Probleme (Root Causes)

1. Falscher Kubernetes-Context:
   - Lokal war teils `docker-desktop` aktiv statt Rancher-Kubeconfig (`rancherConfigs/main.yaml`).
   - Effekt: "No data", obwohl im richtigen Cluster Daten vorhanden waren.
2. Pod-Restarts und fluechtige Workspaces:
   - Nach Neustarts fehlten teils Skripte/Abhaengigkeiten im Runtime-Pod.
3. Abhaengigkeiten/Import-Themen:
   - `opencv-python` vs. `opencv-python-headless`, fehlende Python-Pakete, fehlende Module im Pod.
4. Dashboard-Provisioning-Inkonsistenzen:
   - Unterschiedliche ConfigMap-Staende/Versionen fuehrten zu alten Queries trotz vermeintlicher Updates.
5. Fragile Query-Variante in Grafana:
   - Komplexe `label_replace + group_left`-Konstruktionen erzeugten in bestimmten Situationen `No data`.

### 4) Finale Dashboard-Strategie

Es gibt jetzt zwei getrennte Dashboards mit klar unterschiedlichem Zweck:

1. `SLURM Energy Overview` (`uid=slurm-energy-overview`)
   - Beibehalten fuer "klassische" Anzeige und bestehende Ergebnisse.
2. `SLURM Energy Overview (Zero on Idle)` (`uid=slurm-energy-overview-zero-on-idle`)
   - Speziell fuer das Verhalten "nach Job-Ende gegen 0".
   - Nutzt robuste Zero-on-Idle-Queries mit Aktivitaetsmaske + `vector(0)`-Fallback.

Hinweis zur Bedienung:
- URL-Parameter wie `var-stale_window_sec=120` koennen Darstellung ueberschreiben.
- Fuer Historie immer groesseres Zeitfenster verwenden (z. B. `Last 30 days`).

### 5) Team-Runbook (Kurzfassung)

1. Immer mit korrekter Kubeconfig arbeiten:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy get pods
```

2. Verfuegbarkeit der Metriken zuerst in Prometheus pruefen:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_phase_energy_kwh'"
```

3. Erst dann Grafana beurteilen:
- Falls `No data`: Zeitfenster, URL-Variablen, und aktives Dashboard (normal vs. zero-on-idle) pruefen.

4. Fuer sauberen Neustart:
- MLflow/Grafana/Prometheus-Daten und Energy-Textfiles resetten.
- Deployments neu starten.
- Testlauf mit `model.epochs=3` als Smoke-Test.

### 6) Ergebnis fuer Wissenstransfer

- Reproduzierbarer Trainingsflow mit phasengetrenntem Energie-Tracking ist etabliert.
- CodeCarbon-vs-SLURM-Vergleich fuer Training ist messbar und in Grafana sichtbar.
- Ein separates Zero-on-Idle-Dashboard ist vorhanden, ohne bestehende Dashboard-Ergebnisse zu verfaelschen.

## Generalized Energy Guard: mindestens 20 % GPU-Energieeinsparung

Der neue Controller-Modus `generalized_energy_guard` verfolgt ein explizites Energie-Budget. Er maximiert die Zahl der ausgeführten Epochen unter der Nebenbedingung, gegenüber einem Training bis zur Zielepoche mindestens 20 % Gesamtjob-GPU-Energie einzusparen.

Pro Epoche berechnet der Controller:

1. die seit Start des GPU-Monitors beobachtete Gesamtjob-Energie,
2. den Median der letzten Energieinkremente,
3. die bis Epoche 100 projizierte Full-Training-Energie,
4. die Energie bei sofortigem Stopp einschließlich Post-Training-Reserve,
5. die projizierte Einsparung der aktuellen und der nächsten Epoche.

Die Stop-Grenze lautet:

```text
projected_saving_now >= 0.20
and projected_saving_next < 0.21
```

Zusätzlich stoppt der strikte Modus spätestens bei `floor(100 * 0.77) = 77`. Dieser konservative Fallback wurde aus den unabhängigen Full100-Messkurven von Stufe A und Stufe B abgeleitet. Ein möglicher Qualitätskonflikt wird als `energy_guard_quality_conflict=1` protokolliert und nicht verschwiegen.

Wichtig: Die 20-%-Anforderung ist ein Energie-Ziel, keine gleichzeitige Accuracy-Garantie. Auf unbekannter Hardware oder neuen Aufgaben muss sie erneut mit vollständigen Referenzläufen validiert werden.

### Validierung per historischem Replay

```powershell
python .\scripts\evaluate-generalized-energy-guard.py
```

Der Replay verwendet fünf Seeds aus Stufe A und sieben Szenarien aus Stufe B. Der aktuelle geprüfte Stand:

- 12 von 12 Referenzen erreichen mindestens 20 % Einsparung,
- minimale Einsparung: 21,24 %,
- mediane Einsparung: 21,72 %,
- maximales mAP50-95-Regret: 2,33 Prozentpunkte,
- mittleres mAP50-95-Regret: 1,33 Prozentpunkte.

Maschinenlesbare Ergebnisse liegen unter `results/generalized-energy-guard-replay/`.

### Frische Rancher-Läufe über beide Stufen

Matrix zuerst ohne Ausführung prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-generalized-energy-guard-study.ps1 -Stage Both -PlanOnly
```

Danach die zwölf Controller-Läufe sequenziell ausführen:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-generalized-energy-guard-study.ps1 -Stage Both
```

Stufe A prüft `train128-scratch` mit den Seeds `0` bis `4`. Stufe B prüft sieben CARPK-Szenarien mit unterschiedlichen Datenmengen, Scratch/Pretrained und YOLOv8n/YOLOv8s. Vor der finalen wissenschaftlichen Aussage müssen die frischen Läufe mit den zugehörigen Full100-Läufen verglichen werden.

### Gepaarte 24-Run-Validierungsstudie und eigenes Dashboard

Für die finale Validierung wird jede der zwölf Bedingungen frisch und paarweise ausgeführt:

- `Full100` ohne adaptiven Abbruch,
- `generalized_energy_guard_20pct` mit identischer Daten-, Modell-, Seed- und Ressourcenbelegung.

Die Reihenfolge innerhalb der Paare wechselt, damit Cache-, Temperatur- und zeitabhängige Clustereffekte nicht systematisch nur eine Strategie begünstigen. Insgesamt entstehen `12 Fälle × 2 Strategien = 24 Jobs`.

Versuchsplan prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-generalized-energy-guard-paired-study.ps1 -PlanOnly
```

Komplette Studie ausführen, sichern, analysieren und bereitstellen:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-rancher-generalized-energy-guard-paired-study.ps1
```

Das Skript erzeugt eine neue, von bisherigen Sicherungen unabhängige Sicherung unter:

```text
D:\Masterarbeit_Vergleichssicherung\CARPK_GeneralizedEnergyGuard_20pct_24Jobs_<timestamp>
```

Das Dashboard `Generalized Energy Guard - Cross-Stage 20% Validation` zeigt:

- Zielerfolgsrate und minimale/mittlere Energieeinsparung,
- GPU-Energie `Full100` gegen Controller pro Fall,
- beste mAP50-95 und Accuracy-Regret pro Fall,
- Epochenzahl und Dauer pro Fall,
- Qualitätskonflikte,
- projizierte Einsparung und 20-%-Grenze pro Epoche,
- beobachtete sowie prognostizierte Energie,
- Controllerentscheidung, Fallback und Konfliktsignal.

## Austauschbare Controller-Benchmark-Plattform

Alle Bestandteile der neuen, wiederverwendbaren Plattform liegen strukturiert unter `controller_benchmark/`. Nur die unvermeidbaren Anschlusspunkte an Training, SLURM und Prometheus bleiben in den bestehenden Laufzeitdateien.

Die Plattform trennt fünf Verantwortlichkeiten:

1. `api.py` definiert die stabile Controller-Schnittstelle.
2. `controllers/` enthält austauschbare Controller-Implementierungen.
3. `config/benchmark-v2-shared-seed.json` friert Versuchsbedingungen und Akzeptanzkriterien ein.
4. `planner.py` und `scripts/run-controller-benchmark.ps1` erzeugen Referenz- und Controllerläufe.
5. `analysis.py` und `dashboard/` berechnen Ergebnisse und stellen sie in einem eigenen Grafana-Dashboard dar.

Die Controller-Schnittstelle ist aufgabenunabhängig. Für jede Epoche `t` gilt:

```text
Q_t = quality_score_t,  0 <= Q_t <= 1
delta_Q_t = Q_t - Q_(t-1)
```

Der Controller verarbeitet ausschließlich `Q_t`, `delta_Q_t`, Epochenenergie, kumulierte Energie, Laufzeit und Historie. Aufgabenspezifische Begriffe wie mAP, F1 oder mIoU erscheinen nicht in seiner Entscheidungslogik.

### Eigenen Controller implementieren

Eine neue Python-Datei wird unter `controller_benchmark/controllers/` angelegt. Die Klasse erbt von `Controller` und implementiert nur `evaluate()`:

```python
from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


class MyController(Controller):
    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        stop = observation.epoch >= 30 and (observation.delta_quality or 0.0) < 0.0005
        return ControllerDecision(
            stop=stop,
            reason="my_controller_stop",
            diagnostics={"current_delta": observation.delta_quality},
        )
```

`EpochObservation` stellt unter anderem Epoche, maximale Epochenzahl, aktuelle und beste Qualität, Qualitätsänderung, Epochen- und kumulierte Energie, Dauer, GPU-Auslastung und die vollständige Historie bereit. Frei benannte numerische Werte in `diagnostics` werden automatisch als JSON, MLflow-Metriken und Prometheus-Metriken gespeichert. Damit erfordert ein neuer Controller keine Änderung an der Monitoring-Pipeline.

Als Vorlagen stehen bereit:

- `controller_benchmark.controllers.example_controller:ExampleController`
- `controller_benchmark.controllers.fixed_epoch:FixedEpochController`
- `controller_benchmark.controllers.marginal_efficiency:MarginalEfficiencyController`

### RAPEC-v6: Online Multi-Horizon Pareto Controller

RAPEC-v6 liegt unter `controller_benchmark/controllers/online_multi_horizon_pareto.py`. Der Controller arbeitet vollständig online und verwendet keine historischen Full100-Trainingskurven, keinen datensatzspezifischen Meta-Predictor und keine hart codierten Profile für CARPK, VEDAI, Detection, Classification oder Segmentation.

Nach jeder Epoche prognostiziert RAPEC-v6 für die Horizonte `1`, `3`, `5`, `10` und `20`:

- die erwartete zukünftige normalisierte Qualität,
- den erwarteten Qualitätsgewinn samt Unsicherheitsintervall,
- die Wahrscheinlichkeit eines relevanten Qualitätsgewinns,
- die zusätzliche Trainings-/Epochenenergie in Wh,
- die zusätzliche Laufzeit,
- den konservativen Qualitätsgewinn pro Wh.

Als Eingaben werden die bisherige Qualitäts- und Loss-Kurve, Gradient Norm, Learning Rate, Epochenzeit, GPU-Auslastung, GPU-Speicher, GPU-Leistung, Epochenenergie, Parameteranzahl und FLOPs verwendet. Da Parameteranzahl und FLOPs innerhalb eines einzelnen Jobs konstant sind, dienen sie ohne historischen Meta-Predictor primär zur vollständigen Charakterisierung und zur späteren Vergleichbarkeit. Die kurzfristige Entscheidung wird durch die tatsächlich beobachteten Laufzeit-, Energie-, Gradienten- und Qualitätsverläufe bestimmt.

Die Qualitätstoleranz `epsilon`, die Risikogrenze, Mindestepoche, Patience und Utility-Grenze werden aus dem aktuellen Lauf dynamisch bestimmt. Frühere Vorhersagefehler desselben Jobs kalibrieren die Prognoseintervalle online. Eine Recovery-Erkennung verhindert das Stoppen, wenn Loss, Gradient Norm, Learning Rate oder frühere Plateau-Wiederanläufe noch auf weiteres Lernen hindeuten.

Gestoppt wird nur, wenn:

- kein Prognosehorizont einen belastbaren Pareto-Nutzen zeigt,
- die Wahrscheinlichkeit eines relevanten zukünftigen Gewinns unter der dynamischen Risikogrenze liegt,
- die Recovery-Wahrscheinlichkeit ausreichend niedrig ist,
- die Lernkurve `slow_learning`, `plateau` oder `overfit_risk` zeigt,
- die Unsicherheit aus den verfügbaren Signalen entscheidbar ist,
- die Bedingung über die dynamische Patience hinweg stabil bleibt.

Die Controllerentscheidung verwendet ausschließlich zukünftige Trainings-/Epochenenergie. Gesamtjobenergie wird weiterhin separat für die betriebliche Benchmark-Auswertung erfasst und darf nicht mit der marginalen Entscheidungsenergie verwechselt werden. LC-PFN bleibt eine getrennte Quality-only-Baseline; RAPEC-v6 bindet LC-PFN nicht intern ein.

Versuchsplan prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v6-controller-benchmark-async.ps1 `
  -ControllerId "rapec-v6" `
  -RequireAllStages `
  -PlanOnly
```

Vollständigen asynchronen Rancher-Benchmark starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v6-controller-benchmark-async.ps1 `
  -ControllerId "rapec-v6" `
  -RequireAllStages
```

### RAPEC-v7: Online Bayesian Multi-Horizon Controller

RAPEC-v7 liegt unter `controller_benchmark/controllers/online_bayesian_rapec.py`. Der Controller verbindet die task-unabhängige Energieentscheidung von RAPEC mit Online-Bayesian-Inferenz. Er verwendet ausschließlich Messungen des aktuell laufenden Trainings. Historische Lernkurven, Full100-Verläufe, LC-PFN-Gewichte, Dataset-Profile und Task-Profile werden nicht geladen.

Nach jeder Epoche werden drei bayesianische Modellensembles aktualisiert:

- zukünftige normalisierte Qualität `Q_t`,
- zusätzliche Trainings-/GPU-Energie,
- zusätzliche Trainingsdauer.

Für die Qualitätskurve konkurrieren lineare, logarithmische, exponentielle, Sättigungs-, Change-Point- und lokale Trendmodelle. Ihre Gewichte werden durch bayesianisches Model Averaging bestimmt. Die Gewichtung verwendet keine In-Sample-Anpassung, sondern Mehrhorizont-Hindcasts innerhalb des aktuellen Jobs: Ein früheres Präfix prognostiziert 1, 5 und 10 Epochen voraus und wird anschließend mit den wirklich beobachteten Werten verglichen.

Loss, Gradient Norm und Learning Rate bilden eine konservative Recovery-Evidenz. GPU-Auslastung, GPU-Speicher, GPU-Leistung, Epochenzeit und Epochenenergie gehen in die bayesianischen Energie- und Laufzeitmodelle ein. Hardwaretelemetrie wird bewusst nicht direkt zur Qualitätsregression verwendet, weil ihr kausaler Qualitätseinfluss innerhalb eines einzelnen Jobs nicht identifizierbar ist. Parameteranzahl und FLOPs werden dokumentiert; weil sie innerhalb eines Laufs konstant sind, kann ihr isolierter Einfluss ohne historische Vergleichsläufe ebenfalls nicht geschätzt werden.

RAPEC-v7 prognostiziert für `1`, `3`, `5`, `10` und `20` weitere Epochen:

- die posteriore Qualitätsverteilung,
- den erwarteten Gewinn und sein Credible Interval,
- die Wahrscheinlichkeit eines statistisch relevanten Gewinns,
- die posteriore Energie- und Laufzeitverteilung,
- die Verteilung des Qualitätsgewinns pro Wh.

Die Stop-Schwellen sind nicht task- oder datasetspezifisch. Der kleinste relevante Qualitätsgewinn wird aus Rauschen, Messauflösung und effektiver Stichprobengröße des aktuellen Laufs bestimmt. Die Wahrscheinlichkeitsgrenze folgt aus effektiver Stichprobengröße und Modelluneinigkeit. Die Utility-Grenze entsteht aus der bislang beobachteten Verteilung von Qualitätsgewinn pro Wh. Patience folgt aus der Autokorrelation des aktuellen Lernregimes.

Die Mindestdatenmenge folgt aus Modellkomplexität und Prognosehorizont. Ohne Fremdhistorie müssen für den längsten Horizont mindestens zwei vollständig beobachtbare Fenster vorhanden sein. Bei einem 20-Epochen-Horizont beginnt die Stop-Freigabe daher frühestens nach 40 Beobachtungen, ohne dass eine CARPK-, Dataset- oder Task-Epoche hart codiert wird. Die schwachen Priors werden per Empirical Bayes auf den Median des sichtbaren Präfixes zentriert; alle posterioren Updates verwenden nur den aktuellen Lauf.

Ein Jeffreys-Beta-Posterior schätzt aus Plateau- und Recovery-Ereignissen desselben Jobs die Wahrscheinlichkeit eines späteren Wiederanlaufs. Gestoppt wird erst, wenn kein Pareto-relevanter Horizont verbleibt, die posteriore Recovery-Wahrscheinlichkeit unter der dynamischen Risikogrenze liegt und die Entscheidung über die dynamische Bestätigungsdauer stabil bleibt.

Versuchsplan prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v7-controller-benchmark-async.ps1 `
  -ControllerId "rapec-v7" `
  -RequireAllStages `
  -PlanOnly
```

Vollständigen asynchronen Rancher-Benchmark starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v7-controller-benchmark-async.ps1 `
  -ControllerId "rapec-v7" `
  -RequireAllStages
```

### RAPEC-v8: RAPEC-v3 mit Bayesian Safety Guard

RAPEC-v8 liegt unter `controller_benchmark/controllers/bayesian_guarded_rapec_v3.py`.
Die bewährte RAPEC-v3-Logik bleibt die primäre Stop-Regel: dynamischer relevanter
Qualitätsgewinn, Energie-Utility, Lernzustand, Unsicherheit und Patience werden
unverändert von v3 berechnet. RAPEC-v8 ergänzt nur eine zweite notwendige
Freigabe, damit ein aggressiver v3-Stopp bei noch plausiblem zukünftigem Lernen
verhindert wird.

Der Bayes-Guard verwendet ein konjugiertes Normal-Inverse-Gamma-Modell für die
jüngsten Verbesserungen der Bestqualität. Aus Messungen des aktuell laufenden
Jobs entsteht nach jeder Epoche eine posteriore prädiktive Verteilung für den
Qualitätsgewinn in den nächsten fünf Epochen. Daraus werden Erwartungswert,
10-%-/90-%-Credible-Bounds und
`P(future_gain > dynamic_meaningful_gain)` berechnet. Der relevante Gewinn ist
keine neue feste Qualitätsschwelle, sondern exakt die bereits dynamisch aus
Rauschen, Lernfortschritt und verbleibendem Qualitätsraum bestimmte
RAPEC-v3-Schwelle.

Der Controller stoppt nur, wenn:

- RAPEC-v3 nach seiner eigenen Patience einen Stopp verlangt und
- der Bayes-Posterior dem relevanten Fünf-Epochen-Gewinn nur eine geringe
  Wahrscheinlichkeit zuweist.

Die Bayesian-Schicht verwendet keine historischen Lernkurven, Full100-Verläufe,
Dataset-Profile oder Task-Priors. Sie ist rechnerisch klein: Statt des
mehrteiligen Model Averaging aus RAPEC-v7 werden nur eindimensionale
Posteriorparameter aktualisiert und standardmäßig 512 sehr günstige Samples
gezogen. RAPEC-v8 soll damit den Energiecharakter von v3 behalten, aber dessen
zu frühe Stopps bei Scratch- und VEDAI-Läufen abfangen.

Versuchsplan prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v8-controller-benchmark-async.ps1 `
  -ControllerId "rapec-v8" `
  -RequireAllStages `
  -PlanOnly
```

Vollständigen asynchronen Rancher-Benchmark starten:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v8-controller-benchmark-async.ps1 `
  -ControllerId "rapec-v8" `
  -RequireAllStages
```

### Versuchsplan vorab prüfen

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-controller-benchmark.ps1 `
  -Controller "controller_benchmark.controllers.example_controller:ExampleController" `
  -ControllerId "example-v1" `
  -ControllerParametersJson '{"minimum_epochs":30,"minimum_gain":0.0005}' `
  -PlanOnly
```

Der Plan enthält 15 Vergleichsfälle und deckt vier Ebenen ab:

| Stufe | Fälle | Aussage |
|---|---:|---|
| CARPK-Stabilität | 5 Seeds | Robustheit gegenüber Trainingszufall bei unverändertem Split und Szenario |
| CARPK-Szenario-Generalisation | 7 | Transfer über Datenmenge, Initialisierung und YOLO-Modellkapazität |
| Dataset-Generalisation | 1 | Transfer der Fahrzeugerkennung von CARPK auf VEDAI 512 Fold 1 |
| Task-Generalisation | 2 zusätzliche Aufgaben | Transfer von Objekterkennung auf Klassifikation und Segmentierung über einen gemeinsamen Qualitätsvertrag |

In der Stabilitätsstufe wird Full100 nur einmal mit Seed `0` ausgeführt und als gemeinsame Referenz für die Controller-Seeds `0–4` verwendet. Jedes technisch unterschiedliche Szenario sowie Dataset und Task besitzen jeweils genau eine eigene Full100-Referenz. Insgesamt werden daher höchstens elf Full100-Referenzen gespeichert. Eine Full100-Messung wird aus dem versionierten Baseline-Katalog geladen, wenn Konfigurations-Fingerprint, Dateihashes, Runtime-Image, GPU-Modell und Power-Limit übereinstimmen. Nach Aufbau des vollständigen Katalogs entstehen standardmäßig nur 15 neue Controllerjobs.

### Vollständigen Benchmark ausführen

Nach der Planprüfung wird nur `-PlanOnly` entfernt:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-controller-benchmark.ps1 `
  -Controller "controller_benchmark.controllers.example_controller:ExampleController" `
  -ControllerId "example-v1" `
  -ControllerParametersJson '{"minimum_epochs":30,"minimum_gain":0.0005}'
```

Der Ablauf validiert Controller und Manifest, setzt nur aktive Versuchsmetriken zurück, startet die tatsächlich benötigten Rancher-Jobs, erstellt eine unabhängige Sicherung, berechnet die Vergleiche und stellt das Dashboard `Controller Benchmark - Stability and Generalisation` bereit. Frühere Sicherungen und historische, versionierte Prometheus-Dateien werden dabei nicht überschrieben.

Mit dem derzeitigen achtteiligen Baseline-Katalog startet der erste Vier-Stufen-Lauf 18 Jobs: 15 Controllerfälle sowie die drei neuen Full100-Referenzen für VEDAI, Klassifikation und Segmentierung. Danach enthält der Katalog elf Referenzen und jeder weitere Controllervergleich startet nur noch 15 Jobs. Ohne Cache umfasst eine vollständig frische Messung 26 Jobs. Nur fehlende Full100-Referenzen werden automatisch ergänzt. Für eine bewusst vollständig frische Abschlussmessung kann die Wiederverwendung deaktiviert werden:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-controller-benchmark.ps1 `
  -Controller "controller_benchmark.controllers.example_controller:ExampleController" `
  -ControllerId "example-v1-fresh" `
  -ControllerParametersJson '{"minimum_epochs":30,"minimum_gain":0.0005}' `
  -RefreshBaselines
```

Die Cache-Wiederverwendung ist für schnelle Controllerentwicklung vorgesehen. Für die finale statistische Auswertung der Masterarbeit sollte mindestens eine vollständig frisch gepaarte Messung mit `-RefreshBaselines` durchgeführt werden, damit zeitabhängige Cluster-, Temperatur- und Lastunterschiede nicht unbemerkt in den Vergleich eingehen.

Eine vorhandene Controller-Benchmark-Sicherung kann ohne erneutes Training wieder in Prometheus und Grafana geladen werden:

```powershell
$snapshot = "D:\Masterarbeit_Vergleichssicherung\ControllerBenchmarks\<benchmark_run_id>"
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\deploy-dashboard.ps1 `
  -Snapshot $snapshot `
  -Dashboard "$snapshot\controller_benchmark\controller-benchmark.json"
```

Die Akzeptanzkriterien der eingefrorenen Benchmark-Version 2 sind:

- mindestens 20 % GPU-Energieeinsparung gegenüber der zugeordneten Full100-Referenz,
- höchstens 1,5 Prozentpunkte Verlust bei der besten normalisierten Qualitätsmetrik (`mAP50-95` beziehungsweise `quality_score`),
- gemeinsamer Erfolg in mindestens 80 % der Fälle jeder aktivierten Stufe.

### Dataset-Generalisation mit VEDAI

VEDAI 512 wird beim ersten vollständigen Benchmark direkt von der offiziellen Quelle in den persistenten Cluster-Cache `/workspace-cache/vedai512-source` geladen und reproduzierbar in YOLO-Struktur umgewandelt. Verwendet werden das offizielle Fold 1, ein fester Validierungsanteil und `split_seed=0`. Der Cache verhindert erneute Downloads. Dateigrößen und SHA-256-Prüfsummen der beiden Bildarchive, Annotationen und Nutzungsbedingungen sind im Adapter gepinnt; der Dataset-Fingerprint fließt in den Baseline-Fingerprint ein. VEDAI darf gemäß den mitgespeicherten Nutzungsbedingungen ausschließlich für Forschung und nichtkommerzielle Zwecke verwendet werden; die dort genannte Publikation muss zitiert werden.

Manuelle Vorbereitung:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\prepare-vedai.ps1
```

### Task-Generalisation über `quality_score`

Der Controller erhält für jede Aufgabe ausschließlich eine abstrakte Validierungsqualität

```text
Q_t = quality_score in [0, 1]
```

Er kennt weder YOLO noch die fachliche Bedeutung der zugrunde liegenden Metrik. Der gleiche Plugin-Code entscheidet deshalb anhand von `Q_t`, `delta(Q_t)`, Energie und Historie für alle Aufgaben.

| Aufgabe | Modell | Primäre fachliche Metrik | Abbildung auf `quality_score` |
|---|---|---|---|
| Objekterkennung | YOLOv8 | mAP50-95 | `quality_score = mAP50-95` |
| Bildklassifikation | ResNet18 | Macro-F1 | `quality_score = macro_f1` |
| Segmentierung | Tiny U-Net | Foreground-mIoU | `quality_score = miou` |

Die Klassifikationsaufgabe ordnet CARPK-Bilder anhand ausschließlich aus dem Trainingssplit abgeleiteter Fahrzeugzahl-Tertile den Klassen `low`, `medium` und `high vehicle density` zu. Zusätzlich zu Macro-F1 wird Accuracy gespeichert.

Da CARPK keine Pixelmasken bereitstellt, werden die binären Segmentierungsmasken deterministisch aus den vorhandenen Fahrzeug-Bounding-Boxes erzeugt. Zusätzlich zu mIoU wird Dice gespeichert. Das ist eine schwach annotierte Fahrzeugregions-Segmentierung und muss in der wissenschaftlichen Interpretation ausdrücklich von einer Segmentierung mit manuell annotierten Objektkonturen unterschieden werden.

Fachliche Metriken bleiben in MLflow und Prometheus erhalten. Sie werden jedoch nicht als aufgabenspezifische Eingabe in den Controller eingebaut.

Das Grafana-Dashboard kombiniert beide Sichten:

- Der Bereich `Unified Task-Independent Comparison` verwendet ausschließlich den normalisierten `quality_score` sowie Energie, Regret und Epochen.
- Der Filter `Task Type` grenzt anschließend kaskadierend die auswählbaren Stufen und Fälle ein.
- Der Bereich `Task-Specific Diagnostic Metrics` zeigt zusätzlich mAP/Precision/Recall für Objekterkennung, Accuracy/Macro-F1 für Klassifikation und mIoU/Dice für Segmentierung.

Damit bleibt der Controllervergleich aufgabenunabhängig, ohne die fachlich wichtigen Originalmetriken der einzelnen Aufgaben zu verlieren.

Der Bereich `Energy-Quality Trade-off and Measurement Comparison` ergänzt:

- einen task-unabhängigen Scatterplot mit SLURM-GPU-Energie in Wh auf der X-Achse und normalisiertem `quality_score` auf der Y-Achse,
- task-spezifische Scatterplots für mAP50-95, Macro-F1 und mIoU,
- eine direkte Punktbeschriftung als `Job <SLURM_JOB_ID>`,
- Farben nach Task Type und unterschiedliche Symbole für Full100 und Controller,
- einen gespiegelten SLURM-vs.-CodeCarbon-Plot auf identischer Wh-Skala.

Im gespiegelten Plot wird SLURM oberhalb und CodeCarbon unterhalb der Nulllinie dargestellt. Das negative Vorzeichen ist ausschließlich eine Visualisierung; Tooltip, Beschriftung und wissenschaftliche Interpretation verwenden positive Energieverbräuche. Verglichen wird jeweils GPU-Energie innerhalb derselben Training-Phase. Für Klassifikation und Segmentierung startet und stoppt CodeCarbon deshalb im gleichen Telemetrie-Lebenszyklus wie der SLURM-GPU-Sampler.

Mit `-RequireAllStages` wird geprüft, dass keine der vier Stufen deaktiviert oder blockiert ist.

### Controller-Benchmark vollständig auf Rancher ausführen

Das synchrone Skript `run-controller-benchmark.ps1` benötigt während der gesamten Laufzeit den lokalen Rechner. Für mehrstündige Versuchsreihen steht deshalb zusätzlich eine serverseitige Queue zur Verfügung. Nach dem Upload übernimmt Rancher selbstständig:

1. die sequenzielle Einreichung aller geplanten Full100- und Controllerjobs über SLURM,
2. das Warten auf freie GPU-Ressourcen und auf jeden Jobabschluss,
3. die Wiederaufnahme nach einem Neustart des Orchestrator-Pods,
4. die Analyse der gepaarten Ergebnisse,
5. die Veröffentlichung der Vergleichsmetriken über Pushgateway und Prometheus,
6. die Bereitstellung des Grafana-Dashboards,
7. die persistente Aktualisierung des Full100-Baseline-Katalogs.

Der Orchestrator wird einmalig installiert oder nach Änderungen aktualisiert:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\install-server-orchestrator.ps1
```

Ein neuer Controller wird anschließend asynchron eingereiht:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-controller-benchmark-async.ps1 `
  -Controller "controller_benchmark.controllers.example_controller:ExampleController" `
  -ControllerId "example-v1" `
  -ControllerParametersJson '{"minimum_epochs":30,"minimum_gain":0.0005}' `
  -RequireAllStages
```

Mit dem zusätzlichen Schalter `-PlanOnly` werden Jobanzahl, Reihenfolge und Baseline-Wiederverwendung geprüft, ohne Daten hochzuladen oder einen Auftrag einzureihen.

Sobald die Meldung `Benchmark queued on Rancher` erscheint, sind Code, Manifest, Plan, Controllerparameter und vorhandene Baselines auf dem persistenten Cluster-Volume gespeichert. Ab diesem Zeitpunkt darf der lokale Rechner ausgeschaltet werden. Ein belegter GPU-Knoten ist kein Fehler: Der SLURM-Job bleibt in der Queue und startet, sobald die A100 verfügbar ist. Nur ein Ausfall des Rancher-Clusters selbst unterbricht die Ausführung; nach Wiederherstellung setzt der Orchestrator den persistent gespeicherten Auftrag fort.

Status eines konkreten Laufs anzeigen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 `
  -RunId "<benchmark_run_id>" -Follow
```

Ohne `-RunId` wird der zuletzt eingereichte Lauf angezeigt. `-Follow` ist nur eine Anzeige und wird für die eigentliche Ausführung nicht benötigt.

Nach Abschluss werden alle Rohmetriken, SLURM-Zuordnungen, Analysen, das Dashboard und der aktualisierte Baseline-Katalog heruntergeladen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\download-controller-benchmark-results.ps1 `
  -RunId "<benchmark_run_id>"
```

Die serverseitigen Verzeichnisse sind:

```text
/workspace-cache/controller-benchmark-queue/   # pending, running, done, failed
/workspace-cache/controller-benchmarks/        # vollständige, versionierte Läufe
/workspace-cache/controller-baselines/         # wiederverwendbare Full100-Referenzen
```

`-RefreshBaselines` erzwingt weiterhin frische Full100-Läufe. Ohne diesen Schalter lädt das Submit-Skript zuerst den kompatiblen Baseline-Katalog von Rancher. Damit muss eine Full100-Referenz pro technisch eindeutiger Versuchsbedingung nur einmal trainiert werden, auch wenn der lokale Rechner zwischen zwei Controllerentwicklungen gewechselt oder ausgeschaltet wurde.

### PEP-Controller: probabilistischer Energy-Pareto-Stopp

Der neue Controller `ProbabilisticEnergyParetoController` ist als task-unabhängiger Forschungscontroller umgesetzt. Er arbeitet nicht direkt mit `mAP50-95`, `Macro-F1` oder `mIoU`, sondern mit dem gemeinsamen Vertrag:

```text
Q_t = quality_score in [0, 1]
```

Nach jeder Epoche prognostiziert der Controller für die nächsten fünf Epochen:

- erwartete Qualität nach dem Horizont,
- erwarteter Qualitätsgewinn,
- erwarteter zusätzlicher Energieverbrauch in Wh,
- Wahrscheinlichkeit, dass der Qualitätsgewinn größer als 0,2 Prozentpunkte ist,
- erwartete Qualität pro Wh,
- projizierte Energieersparnis gegenüber Full100.

Die Entscheidung ist dadurch probabilistisch statt deterministisch. Ein Stopp erfolgt nur, wenn die erwartete Verbesserung klein ist, die Wahrscheinlichkeit eines nützlichen Gewinns niedrig ist, die Qualität-pro-Wh unter der Schwelle liegt, die Ziel-Energieersparnis noch erreichbar ist und diese Evidenz über mehrere Epochen stabil bleibt.

Zusätzlich zum Qualitätsverlauf nutzt der Controller optionale System- und Trainingssignale:

```text
gradient_norm
learning_rate
duration_seconds
gpu_util_avg_pct
gpu_memory_used_mb
gpu_power_avg_w
model_parameter_count
model_flops
train_loss
```

Fehlende Signale blockieren die Entscheidung nicht. Der Controller protokolliert `pep_feature_coverage_fraction`, damit sichtbar bleibt, wie vollständig die jeweilige Datenbasis war.

Im Controller-Benchmark-Dashboard zeigt die Zeile `PEP Controller Probabilistic Diagnostics` die zentrale Entscheidungsevidenz: erwarteter Fünf-Epochen-Gewinn, Wahrscheinlichkeit eines Gewinns über 0,2 Prozentpunkte, erwartete Zusatzenergie, Qualität pro Wh, Feature-Abdeckung und Stop-Streak.

Asynchroner Rancher-Start mit den empfohlenen Default-Parametern:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-pep-controller-benchmark-async.ps1 `
  -ControllerId "pep-v1" `
  -RequireAllStages
```

Nur Plan prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-pep-controller-benchmark-async.ps1 `
  -ControllerId "pep-v1" `
  -RequireAllStages `
  -PlanOnly
```

Für den wissenschaftlichen Vergleich bleibt LC-PFN die starke quality-only Baseline: LC-PFN extrapoliert probabilistisch den Lernkurvenverlauf aus frühen Epochen. Der PEP-Controller erweitert diese Idee im MLOps-Kontext um Energie, GPU-Leistung, Speicher, Laufzeit, Modellgröße und Pareto-Entscheidung. Der optionale Adapter liegt unter:

```text
controller_benchmark.controllers.lcpfn_quality_baseline:LcpfnQualityBaselineController
```

Wichtig: LC-PFN ist bewusst nicht als harte Abhängigkeit im Rancher-Runtime-Image enthalten, weil das Paket eigene Python-/Torch-Versionsgrenzen hat. Für eine faire wissenschaftliche LC-PFN-Baseline sollte dieser Adapter in einem separaten kompatiblen Image oder als Offline-Replay auf gespeicherten Full100-Lernkurven ausgeführt werden.
## Pareto-Optimierung der RAPEC-Parameter

Die RAPEC-Parameter können mit einem getrennten Offline-Workflow optimiert werden, ohne bestehende Full100-Läufe oder historische Dashboard-Daten zu verändern. Der Workflow kopiert nur die versionierten Full100-Epochenzusammenfassungen aus einem abgeschlossenen Rancher-Run, prüft alle SHA-256-Hashes und sperrt die Quelle durch `source-lock.json` gegen unbemerkte Änderungen.

Die Optimierung verwendet Optuna NSGA-II mit drei Zielen: mittleren Qualitätsverlust minimieren, schlechtesten normalisierten Qualitätsverlust minimieren und mediane Energieeinsparung maximieren. Die erlaubte Qualitätsabweichung wird pro Lernkurve aus Validierungsrauschen und Spätphasenunsicherheit berechnet. Der Optimizer betrachtet keinen Abstand zur idealen Stop-Epoche. Es gibt keine feste Mindestenergieeinsparung und keine für alle Aufgaben identische Qualitätsgrenze.

Der Standardlauf führt drei Leave-One-Task-Out-Studien und eine finale All-Task-Studie aus. `500` Trials pro Studie ergeben insgesamt `2.000` Offline-Auswertungen:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-pareto-optimization.ps1 `
  -SourceRunId "20260729T170915Z-rapec-v3-v8-nine-dataset" `
  -TrialsPerStudy 500
```

Jeder Optimierungslauf erhält einen neuen Ordner unter `results/controller-optimization/runs/`. Wichtige Ausgaben sind `optimization-manifest.json`, `all-trials.csv`, `pareto-front.csv`, `leave-one-task-out-report.json`, `selected-case-results.csv` und `selected-controllers.json`. Die drei Profile bedeuten:

- `conservative`: kleinster schlechtester Qualitätsverlust,
- `balanced`: kleinster normalisierter Abstand zum mehrdimensionalen Pareto-Ideal,
- `aggressive`: größte mediane Energieeinsparung unter den dynamischen Qualitätsbedingungen.

Offline-Replay ist eine kontrafaktische Simulation: Bis zur Stop-Epoche wird dieselbe Lernkurve wie bei Full100 angenommen. Deshalb müssen die ausgewählten Profile anschließend als echte Rancher-Jobs validiert werden. Die vorhandenen Full100-Baselines werden dabei per Checksumme wiederverwendet:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-pareto-controller-validation-async.ps1 `
  -OptimizationDir ".\results\controller-optimization\runs\<optimization-id>" `
  -Profiles conservative,balanced,aggressive
```

Mit `-PlanOnly` wird nur kontrolliert, welche Kandidatenjobs geplant wären. Jeder reale Kandidat wird als eigener Rancher-Run gespeichert; bestehende 7-, 9-, 27- und 60-Run-Ergebnisse bleiben unverändert.
