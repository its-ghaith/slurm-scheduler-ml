param(
    [ValidateSet("bootstrap", "rollout", "submit", "test", "portforward")]
    [string]$Action = "bootstrap",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$Manifest = "rancherConfigs/slurm-stack.yaml",
    [string]$TrainCmd = "python train_with_energy_tracking_mlflow.py"
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path = ".env")
    if (-not (Test-Path $Path)) { return }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { return }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim().Trim("'`\"")
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Ensure-Namespace {
    $ns = kubectl --kubeconfig $Kubeconfig get ns $Namespace --ignore-not-found -o name
    if (-not $ns) {
        kubectl --kubeconfig $Kubeconfig create namespace $Namespace | Out-Null
    }
}

function Ensure-GhcrSecret {
    if (-not $env:CR_PAT) {
        throw "CR_PAT missing. Put CR_PAT in .env or env var."
    }
    kubectl --kubeconfig $Kubeconfig -n $Namespace delete secret ghcr-pull-secret --ignore-not-found | Out-Null
    kubectl --kubeconfig $Kubeconfig -n $Namespace create secret docker-registry ghcr-pull-secret `
      --docker-server=ghcr.io `
      --docker-username=its-ghaith `
      --docker-password=$env:CR_PAT | Out-Null
}

function Apply-Stack {
    kubectl --kubeconfig $Kubeconfig apply -f $Manifest | Out-Null
}

function Restart-And-Wait {
    kubectl --kubeconfig $Kubeconfig -n $Namespace rollout restart deploy/mlflow deploy/slurmctld deploy/slurmd | Out-Null
    kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status deploy/mlflow --timeout=300s
    kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status deploy/slurmctld --timeout=300s
    kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status deploy/slurmd --timeout=300s
}

function Submit-Job {
    kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmctld -- bash -lc "TRAIN_CMD='$TrainCmd' sbatch /workspace/slurm/train_mlflow_local.slurm"
}

function Run-Test {
    kubectl --kubeconfig $Kubeconfig -n $Namespace get pods -o wide
    kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmctld -- bash -lc "scontrol ping && sinfo -N -l"
    kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
}

function Show-PortForward {
    Write-Host "Run in separate terminals:" -ForegroundColor Yellow
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/grafana 3000:3000"
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/mlflow 5000:5000"
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/prometheus 9090:9090"
}

Import-DotEnv

switch ($Action) {
    "bootstrap" {
        Ensure-Namespace
        Ensure-GhcrSecret
        Apply-Stack
        Restart-And-Wait
        Run-Test
    }
    "rollout" { Restart-And-Wait }
    "submit" { Submit-Job }
    "test" { Run-Test }
    "portforward" { Show-PortForward }
}
