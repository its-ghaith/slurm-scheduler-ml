[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipMlflowReset,
    [switch]$SkipRestart,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Invoke-Kubectl {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & kubectl @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: kubectl $($Arguments -join ' ')"
    }
}

function Get-KubectlOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & kubectl @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: kubectl $($Arguments -join ' ')`n$output"
    }
    return ($output | Out-String).Trim()
}

function Wait-Rollout {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Deployment
    )

    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "rollout", "status", "deployment/$Deployment",
        "--timeout=300s"
    )
}

Write-Host "Pruefe Zugriff auf Namespace $Namespace ..." -ForegroundColor Cyan
Invoke-Kubectl -Arguments @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pods")

$summary = @"
Dieses Reset-Skript fuehrt folgende Schritte aus:
1. Laufende oder wartende SLURM-Jobs abbrechen.
2. Energy-/Prometheus-Textdateien und SLURM-Logs im slurmd-Pod loeschen.
3. Optional MLflow-DB und Artefakte im mlflow-Pod loeschen.
4. Optional slurmctld, slurmd, prometheus, grafana und mlflow neu starten.
"@

Write-Host $summary -ForegroundColor Yellow

if ($WhatIfPreference) {
    Write-Host "WhatIf aktiv: Es werden keine Aenderungen ausgefuehrt." -ForegroundColor Yellow
    return
}

if (-not $Force -and -not $PSCmdlet.ShouldProcess("$Namespace", "Reset SLURM jobs, metrics, MLflow data and observability state")) {
    return
}

Write-Host "Ermittle alte SLURM-Jobs ..." -ForegroundColor Cyan
$jobIdsRaw = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmctld",
    "--", "bash", "-lc",
    "squeue -h -o '%A' || true"
)

$jobIds = @(
    $jobIdsRaw -split "[`r`n\s]+" |
    Where-Object { $_ -match '^\d+$' } |
    Sort-Object -Unique
)

if ($jobIds.Count -gt 0) {
    Write-Host "Breche Jobs ab: $($jobIds -join ', ')" -ForegroundColor Yellow
    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/slurmctld",
        "--", "bash", "-lc",
        "scancel $($jobIds -join ' ')"
    )
}
else {
    Write-Host "Keine laufenden oder wartenden Jobs gefunden." -ForegroundColor Green
}

Write-Host "Leere Energy-Metriken und Logs im slurmd-Pod ..." -ForegroundColor Cyan
Invoke-Kubectl -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmd", "-c", "slurmd",
    "--", "bash", "-lc",
    @'
mkdir -p /workspace/energy_metrics/node_exporter /workspace/logs
find /workspace/energy_metrics -mindepth 1 -maxdepth 1 -type f -delete
rm -f /workspace/energy_metrics/node_exporter/*
rm -f /workspace/logs/*
rm -f /workspace/slurm-*.out /workspace/job-*.out
echo 'slurmd cleanup complete'
'@
)

if (-not $SkipMlflowReset) {
    Write-Host "Leere MLflow-Daten und Artefakte ..." -ForegroundColor Cyan
    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/mlflow",
        "--", "bash", "-lc",
        @'
mkdir -p /mlflow /mlruns
rm -rf /mlflow/*
rm -rf /mlruns/*
echo 'mlflow cleanup complete'
'@
    )
}
else {
    Write-Host "MLflow-Reset uebersprungen." -ForegroundColor Yellow
}

if (-not $SkipRestart) {
    $deployments = @("slurmctld", "slurmd", "prometheus", "grafana")
    if (-not $SkipMlflowReset) {
        $deployments += "mlflow"
    }

    foreach ($deployment in $deployments) {
        Write-Host "Starte Deployment neu: $deployment" -ForegroundColor Cyan
        Invoke-Kubectl -Arguments @(
            "--kubeconfig", $Kubeconfig,
            "-n", $Namespace,
            "rollout", "restart", "deployment/$deployment"
        )
    }

    foreach ($deployment in $deployments) {
        Write-Host "Warte auf Rollout: $deployment" -ForegroundColor Cyan
        Wait-Rollout -Deployment $deployment
    }
}
else {
    Write-Host "Deployment-Neustarts uebersprungen." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Reset abgeschlossen." -ForegroundColor Green
Write-Host "Naechste Schritte:" -ForegroundColor Green
Write-Host "1. Grafana neu laden und Zeitfenster auf 'Last 6 hours' pruefen."
Write-Host "2. MLflow neu oeffnen; bei MLflow-Reset ist die Run-Historie jetzt leer."
Write-Host "3. Danach neuen Smoke-Test-Job ueber SLURM starten."
