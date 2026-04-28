# Kubernetes Runbook (Rancher + SLURM + MLflow + Grafana)

This document explains how to reproduce the current working setup in namespace `mlops-energy` on Rancher/Kubernetes.

## Scope and goal

Goal:
- Developer keeps training logic in `train.py` / `train_with_energy_tracking_mlflow.py`
- Training is started via SLURM (`sbatch`)
- Metrics are visible in MLflow and Grafana
- Energy metrics are exported via node-exporter textfile collector and scraped by Prometheus

Stack components:
- `slurmctld` (controller)
- `slurmd` (worker + GPU + node-exporter sidecar)
- `mlflow`
- `prometheus`
- `grafana`

Kubernetes manifest:
- `rancherConfigs/slurm-stack.yaml`

## Architecture diagram

```mermaid
flowchart LR
  A[Developer
train.py / train_with_energy_tracking_mlflow.py] --> B[sbatch on slurmctld]
  B --> C[slurmd executes training on GPU]
  C --> D[Training logs + metrics JSON/CSV]
  D --> E[node_exporter textfile .prom]
  E --> F[Prometheus scrape]
  F --> G[Grafana dashboard]
  C --> H[MLflow tracking API]
  H --> I[MLflow UI]

  subgraph K8s Namespace mlops-energy
    B
    C
    E
    F
    G
    H
  end
```

## Data flow details

1. `sbatch /workspace/slurm/train_mlflow_local.slurm` is submitted inside `slurmctld`.
2. `slurmd` runs training and GPU monitor.
3. GPU metrics are written under `/workspace/energy_metrics`.
4. `export_job_metrics_prom.py` writes:
   - `job_<id>.prom`
   - `aggregate.prom`
5. `node-exporter` sidecar reads textfiles.
6. Prometheus scrapes `slurmd-node-exporter:9100`.
7. Grafana uses Prometheus datasource and dashboard `SLURM Energy Overview`.
8. Training and job energy are logged into MLflow.

## Prerequisites

- Access to Rancher cluster (`rancherConfigs/main.yaml`)
- Namespace `mlops-energy` exists
- User has permissions in namespace (`create/get/list/watch/update/patch` for deployments/services/configmaps/secrets/pods)
- GPU node available in cluster
- Docker login/credentials for GHCR

## Repository images in use

Image used by `mlflow`, `slurmctld`, `slurmd`:
- `ghcr.io/its-ghaith/slurm-scheduler-ml/mlops-slurm-runtime:latest`

Reference digests are visible with:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy get pod -l app=slurmd -o jsonpath='{.items[0].status.containerStatuses[0].imageID}'
```

## Secrets and environment

Create `.env` locally (already ignored by `.gitignore`):

```env
# GitHub Container Registry Personal Access Token
# Required scope: read:packages (pull), write:packages (if you push), repo (if package is private and linked)
CR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional helper values
GHCR_USER=its-ghaith
GHCR_SERVER=ghcr.io
GHCR_IMAGE=ghcr.io/its-ghaith/slurm-scheduler-ml/mlops-slurm-runtime:latest
```

Create/refresh pull secret:

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy delete secret ghcr-pull-secret --ignore-not-found
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy create secret docker-registry ghcr-pull-secret `
  --docker-server=ghcr.io `
  --docker-username=its-ghaith `
  --docker-password=$env:CR_PAT
```

## Deploy / redeploy

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml apply -f rancherConfigs/slurm-stack.yaml
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy rollout restart deploy/mlflow deploy/slurmctld deploy/slurmd
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy rollout status deploy/mlflow --timeout=300s
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy rollout status deploy/slurmctld --timeout=300s
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy rollout status deploy/slurmd --timeout=300s
```

## End-to-end test

### 1) Health check

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy get pods -o wide
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "scontrol ping && sinfo -N -l"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmd -c slurmd -- bash -lc "python -c 'import torch; print(torch.cuda.is_available())'"
```

Expected:
- `slurmctld` UP
- node `slurmd` in `idle`
- `torch.cuda.is_available()` is `True`

### 2) Submit job

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "sbatch /workspace/slurm/train_mlflow_local.slurm"
```

Watch queue:

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "squeue"
```

### 3) Verify output and metrics

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmd -c slurmd -- bash -lc "ls -lah /workspace/logs && ls -lah /workspace/energy_metrics/node_exporter"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
```

## Access UIs

Use port-forward:

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/grafana 3000:3000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/mlflow 5000:5000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/prometheus 9090:9090
```

URLs:
- Grafana: `http://localhost:3000` (admin/admin)
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`

## Troubleshooting

### GHCR pull fails with 403
- Recreate `ghcr-pull-secret` with valid PAT
- Ensure PAT has package access
- Ensure package visibility/permissions include cluster pull user

### MLflow `Invalid Host header`
- `MLFLOW_SERVER_ALLOWED_HOSTS` is set to `*` in deployment (already in manifest)

### SLURM node stays `unknown`
- Check `slurmd` logs:
  - CPU topology mismatch
  - `gres.conf` device mismatch
- In this setup, entrypoint creates `/dev/nvidia2` symlink dynamically from first available `/dev/nvidia*`

### Job pending due to DOWN node
- Verify `sinfo -N -l` and `slurmd` logs
- Ensure `slurmd` is on GPU node (`malea-srv01` in current manifest)

## Git and registry documentation points

### Recommended git identity setup

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Branch workflow
- Branch used here: `Rancher-sulrm`
- Commit often with infra + docs together
- Keep `rancherConfigs/main.yaml` out of git (already ignored)

### Package registry notes
- Runtime image is in GHCR
- Keep token only in local `.env`
- Commit only `.env.example`

## Files changed for this setup

- `rancherConfigs/slurm-stack.yaml`
- `Dockerfile.slurm`
- `docs/K8S_RANCHER_RUNBOOK.md`
- `README.md`
- `.env.example`
