# Local Branch: SLURM + MLflow + Prometheus + Grafana (Docker Compose)

Dieser Branch (`local-sulrm`) enthaelt nur die lokale Umgebung. Ziel ist ein reproduzierbarer End-to-End-Workflow:

1. Training wird ueber SLURM gestartet.
2. GPU-/Energie-Metriken werden pro Job gesammelt.
3. Job-Metriken landen in Prometheus/Grafana.
4. Trainingsmetriken und Artefakte landen in MLflow.

## Architektur und Zielbild

Komponenten im lokalen Stack:
- `slurmctld` (Controller)
- `slurmd` (Worker mit GPU)
- `mlflow` (Tracking)
- `node-exporter` (textfile collector)
- `prometheus` (Scrape + Query)
- `grafana` (Dashboard)
- `cadvisor` und `nvidia-dcgm-exporter` (Host-/GPU-Telemetrie)

Visualisierung:

![Workflow](/D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/docs_assets/workflow.png)

![Datenfluss](/D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/docs_assets/dataflow.png)

## Entwicklungsstand und wichtige Entscheidungen

- Lokale Ausfuehrung erfolgt zentral ueber `docker-compose.yml`.
- Training wird ueber `slurm/train_mlflow_local.slurm` gestartet.
- Energie-Metriken werden aus Job-Outputs erzeugt und als `.prom` exportiert.
- Dashboard-Provisioning erfolgt automatisch ueber `grafana/provisioning/*`.
- Alle uebergebbaren Laufzeitparameter liegen in `.env` / `.env.example`.

## Relevante Dateien

- `docker-compose.yml`
- `setup_poc.ps1`
- `.env.example`
- `slurm/train_mlflow_local.slurm`
- `slurm/train_mlflow_job.slurm`
- `slurm/gpu_monitor.sh`
- `slurm/summarize_gpu_metrics.py`
- `slurm/export_job_metrics_prom.py`
- `slurm/log_job_energy_mlflow.py`
- `grafana/dashboards/slurm-energy-overview.json`
- `prometheus/prometheus.yml`

## Konfiguration

### 1) `.env` erstellen

```powershell
Copy-Item .env.example .env
```

`.env` wird nicht committed (`.gitignore`).

Beispielparameter:

```env
TRAIN_CMD=python train_with_energy_tracking_mlflow.py
BATCH_SIZE=32
MLFLOW_TRACKING_URI=http://mlflow:5000
GPU_SAMPLE_INTERVAL_SEC=0.2
ELECTRICITY_PRICE_EUR_PER_KWH=0.30
GRID_CO2_KG_PER_KWH=0.4
PUE_FACTOR=1.0
MLFLOW_LOG_JOB_ENERGY=1
MLFLOW_JOB_ENERGY_EXPERIMENT=slurm-job-energy
```

## Betrieb (End-to-End)

### 2) Stack starten

```powershell
docker compose up -d --build
```

### 3) Training ueber SLURM starten

```powershell
docker exec -i slurmctld bash -lc "TRAIN_CMD='python train.py' sbatch /workspace/slurm/train_mlflow_local.slurm"
```

Optional mit `setup_poc.ps1`:

```powershell
./setup_poc.ps1 -Action all
```

### 4) Status pruefen

```powershell
docker exec -i slurmctld squeue
docker exec -i slurmctld scontrol show job <JOBID>
```

### 5) UIs aufrufen

- Grafana: [http://localhost:3000](http://localhost:3000)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- MLflow: [http://localhost:5000](http://localhost:5000)

Grafana Dashboard:
- [http://localhost:3000/d/slurm-energy-overview/slurm-energy-overview?orgId=1&from=now-30d&to=now](http://localhost:3000/d/slurm-energy-overview/slurm-energy-overview?orgId=1&from=now-30d&to=now)

## Was pro Job erzeugt wird

- `energy_metrics/gpu_metrics_job_<JOBID>.csv`
- `energy_metrics/gpu_summary_job_<JOBID>.json`
- `energy_metrics/node_exporter/job_<JOBID>.prom`
- `energy_metrics/node_exporter/aggregate.prom`
- `logs/ml-energy-local-<JOBID>.out`
- `logs/ml-energy-local-<JOBID>.err`

## E2E-Testcheckliste

1. Container laufen:
```powershell
docker ps
```

2. SLURM Health:
```powershell
docker exec -i slurmctld sinfo
docker exec -i slurmctld squeue
```

3. Prometheus Query:
```powershell
curl http://localhost:9090/api/v1/query?query=slurm_jobs_total
curl http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh
```

4. node-exporter:
```powershell
curl http://localhost:9100/metrics
```
Erwartet: `node_textfile_scrape_error 0` und `slurm_job_*` Metriken.

5. Grafana:
- Dashboard `SLURM Energy Overview` zeigt KPI-Karten und Job-Zeitreihen.

## Troubleshooting

### Grafana zeigt "No data"
- Prometheus-Query direkt testen (`slurm_jobs_total`).
- Zeitbereich in Grafana vergroessern (`Last 30 days`).
- Dashboard-URL mit `orgId=1` verwenden.

### `sbatch` Fehler mit DOS line breaks
- SLURM-Skripte mit Unix-LF speichern.

### `node_textfile_scrape_error = 1`
- Sicherstellen, dass `.prom` Dateien gueltig sind und lesbar (`0644`).

### Job startet, aber keine GPU-Werte
- In `slurmd` pruefen, ob GPU sichtbar ist (`nvidia-smi`).
- `nvidia-dcgm-exporter` und GPU Runtime in Docker pruefen.

## Hinweise fuer spaetere HPC/Rancher-Migration

- Rancher/Kubernetes ist absichtlich getrennt im Branch `Rancher-sulrm`.
- Dieser Local-Branch enthaelt keine Rancher-spezifische Anleitung.
- Fuer Clusterbetrieb denselben Trainingsfluss mit cluster-spezifischem `sbatch` verwenden.
