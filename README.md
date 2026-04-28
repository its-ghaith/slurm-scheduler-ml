# Rancher / Kubernetes Branch (`Rancher-sulrm`)

This branch contains the Kubernetes/Rancher implementation of:
- SLURM (`slurmctld`, `slurmd` with GPU)
- MLflow
- Prometheus
- Grafana
- node-exporter textfile metrics pipeline

## Branch split

- `Rancher-sulrm`: Rancher/Kubernetes setup (this branch)
- `local-sulrm`: local Docker Compose setup

## Main files

- `rancherConfigs/slurm-stack.yaml`
- `setup_rancher.ps1`
- `docs/K8S_RANCHER_RUNBOOK.md`
- `.env.example`

## 1) Configure `.env`

Copy `.env.example` to `.env` and set at least:

```env
CR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`CR_PAT` is used by `setup_rancher.ps1` to create `ghcr-pull-secret`.

## 2) Bootstrap stack

```powershell
./setup_rancher.ps1 -Action bootstrap
```

This does:
- ensure namespace exists
- create/update GHCR pull secret from `.env` (`CR_PAT`)
- apply `rancherConfigs/slurm-stack.yaml`
- rollout restart + wait for `mlflow`, `slurmctld`, `slurmd`
- basic health checks

## 3) Submit training job

```powershell
./setup_rancher.ps1 -Action submit
```

Optional custom training command:

```powershell
./setup_rancher.ps1 -Action submit -TrainCmd "python train.py"
```

## 4) Test metrics quickly

```powershell
./setup_rancher.ps1 -Action test
```

## 5) Access web UIs

```powershell
./setup_rancher.ps1 -Action portforward
```

Then open:
- Grafana: `http://localhost:3000`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`

## Detailed runbook

See `docs/K8S_RANCHER_RUNBOOK.md` for architecture diagram, troubleshooting, and full E2E verification.
