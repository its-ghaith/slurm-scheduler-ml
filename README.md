# Rancher Branch: SLURM + MLflow + Prometheus + Grafana (Kubernetes)

Dieser Branch (`Rancher-sulrm`) enthaelt nur die Rancher/Kubernetes-Umgebung. Ziel ist ein reproduzierbarer End-to-End-Workflow:

1. Training wird ueber SLURM im Kubernetes-Cluster gestartet.
2. GPU-/Energie-Metriken werden pro Job gesammelt.
3. Job-Metriken landen in Prometheus/Grafana.
4. Trainingsmetriken und Artefakte landen in MLflow.

## Architektur und Zielbild

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

1. `sbatch` wird in `slurmctld` ausgefuehrt.
2. `slurmd` fuehrt Training auf GPU aus.
3. CSV/JSON Energie-Metriken werden unter `/workspace/energy_metrics` geschrieben.
4. `export_job_metrics_prom.py` erzeugt `job_<id>.prom` und `aggregate.prom`.
5. `slurmd-node-exporter` liest Textfiles.
6. Prometheus scraped node-exporter.
7. Grafana visualisiert `SLURM Energy Overview`.
8. MLflow speichert Trainings- und Energie-Metriken.

## Relevante Dateien

- `setup_rancher.ps1`
- `.env.example`
- `rancherConfigs/slurm-stack.yaml`
- `slurm/train_mlflow_local.slurm`
- `slurm/export_job_metrics_prom.py`
- `slurm/summarize_gpu_metrics.py`
- `slurm/log_job_energy_mlflow.py`
- `grafana/dashboards/slurm-energy-overview.json`
- `prometheus/prometheus.yml`

## Konfiguration

### 1) `.env` erstellen

```powershell
Copy-Item .env.example .env
```

`.env` wird nicht committed (`.gitignore`).

### 1a) Kubeconfig lokal bereitstellen

Diese Branch erwartet eine lokale Datei:

`rancherConfigs/main.yaml`

Wenn die Datei fehlt, schlagen `bootstrap`, `submit` und `test` sofort fehl. Die Datei bleibt absichtlich unversioniert.

Pflichtwert:

```env
CR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Weitere optionale Parameter stehen vollstaendig in `.env.example`:
- `ACTION`, `KUBECONFIG_PATH`, `K8S_NAMESPACE`, `K8S_MANIFEST`
- `TRAIN_CMD`, `BATCH_SIZE`
- `MLFLOW_TRACKING_URI`, `GPU_SAMPLE_INTERVAL_SEC`
- `ELECTRICITY_PRICE_EUR_PER_KWH`, `GRID_CO2_KG_PER_KWH`, `PUE_FACTOR`
- `MLFLOW_LOG_JOB_ENERGY`, `MLFLOW_JOB_ENERGY_EXPERIMENT`

## Betrieb (End-to-End)

### 2) Stack bootstrap

```powershell
./setup_rancher.ps1 -Action bootstrap
```

Enthaelt automatisch:
- Namespace-Pruefung
- GHCR Pull-Secret aus `CR_PAT`
- Manifest apply
- Rollout restart/wait
- Basischecks

### 3) Job ueber SLURM starten

```powershell
./setup_rancher.ps1 -Action submit
```

Optional:

```powershell
./setup_rancher.ps1 -Action submit -TrainCmd "python train.py"
```

### 4) Funktionstest

```powershell
./setup_rancher.ps1 -Action test
```

## Direkter E2E-Test per kubectl

1. Health:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy get pods -o wide
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "scontrol ping && sinfo -N -l"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmd -c slurmd -- bash -lc "python -c 'import torch; print(torch.cuda.is_available())'"
```

2. Submit:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "TRAIN_CMD='python train.py' sbatch /workspace/slurm/train_mlflow_local.slurm"
```

3. Metrics:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmd -c slurmd -- bash -lc "ls -lah /workspace/energy_metrics/node_exporter"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
```

## Zugriff auf Web-UIs

Port-Forward:

```powershell
./setup_rancher.ps1 -Action portforward
```

Oder manuell:

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/grafana 3000:3000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/mlflow 5000:5000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/prometheus 9090:9090
```

- Grafana: [http://localhost:3000](http://localhost:3000)
- MLflow: [http://localhost:5000](http://localhost:5000)
- Prometheus: [http://localhost:9090](http://localhost:9090)

## Troubleshooting

### GHCR pull Fehler (403/401)
- Secret neu erstellen (`ghcr-pull-secret`).
- PAT-Scope pruefen (`read:packages`; bei privatem Paket zusaetzlich passende Repo-Rechte).

### Grafana zeigt "No data"
- Prometheus Query direkt testen (`slurm_jobs_total`, `slurm_job_training_energy_kwh`).
- Zeitfenster vergroessern (`Last 30 days`).
- Pruefen, ob `.prom` Dateien im `slurmd` existieren.

### SLURM node ist DOWN/unknown
- `slurmd` Logs pruefen.
- GPU/GRES Mapping pruefen.
- Sicherstellen, dass Worker auf GPU-Node geplant ist.

### MLflow nicht erreichbar
- `kubectl get svc -n mlops-energy` pruefen.
- Port-Forward pruefen.
- Bei externem Zugriff Ingress/NodePort separat konfigurieren.

## Git / Registry Hinweise

- Branch ist absichtlich getrennt von Local (`local-sulrm`).
- `.env` nie committen, nur `.env.example`.
- `rancherConfigs/main.yaml` bleibt lokal (ignoriert).
- Runtime-Image: `ghcr.io/its-ghaith/slurm-scheduler-ml/mlops-slurm-runtime:latest`.
