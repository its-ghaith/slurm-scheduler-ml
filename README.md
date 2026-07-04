# Rancher Branch: SLURM + MLflow + Prometheus + Grafana (Kubernetes)

Dieser Branch (`Rancher-sulrm`) enthaelt nur die Rancher/Kubernetes-Umgebung.
Ziel: reproduzierbarer End-to-End-Workflow fuer GPU-Training ueber SLURM mit Sichtbarkeit in MLflow und Grafana.

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
- Runner: `rotationally-invariant-cnns-changes/train_repro_phase_tracked.py`
- SLURM Job: `slurm/train_repro_rotacnn_job.slurm`
- Phase-Summary: `slurm/summarize_gpu_metrics_phases.py`
- Prometheus Export (Phasen): `slurm/export_job_phase_metrics_prom.py`

### Submit Beispiel

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "REPRO_DATASET=foci REPRO_MODEL=yolov8 REPRO_OVERRIDES='experiment=repro_foci_y8;dataset.train_size=158;model.epochs=100;model.patience=8;model.batch_size=8;dataset.img_size=660' sbatch /workspace/slurm/train_repro_rotacnn_job.slurm"
```

`REPRO_OVERRIDES` Format:
- Semikolon-getrennt: `key=value;key2=value2`
- Beispiel: `dataset.path=/workspace/rotationally-invariant-cnns/data/uc_cells`

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
- Start und Ende jeder Phase werden in `tracked_phase(...)` in `rotationally-invariant-cnns-changes/train_repro_phase_tracked.py` geschrieben
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

### `rotationally-invariant-cnns-changes/train_repro_phase_tracked.py`

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

