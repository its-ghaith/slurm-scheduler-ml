param(
    [ValidateSet("all", "services", "submit", "status", "logs", "test", "open")]
    [string]$Action = "all",
    [string]$TrainCmd = "python train_with_energy_tracking_mlflow.py",
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"

function Write-Header {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "   ML Energy PoC Setup (SLURM + Prometheus + Grafana)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Ensure-Folders {
    New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path ".\energy_metrics\node_exporter" | Out-Null
}

function Assert-ContainerRunning {
    param([Parameter(Mandatory = $true)][string]$Name)
    $state = docker inspect -f "{{.State.Running}}" $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or "$state".Trim() -ne "true") {
        throw "Container '$Name' is not running. Start stack first: docker compose up -d --build"
    }
}

function Start-Services {
    Write-Host "[1/4] Starting services with docker compose..." -ForegroundColor Yellow
    docker compose up -d --build
    Write-Host "Waiting for startup..." -ForegroundColor Yellow
    Start-Sleep -Seconds 8
    Write-Host "Services are up." -ForegroundColor Green
}

function Submit-Job {
    Assert-ContainerRunning -Name "slurmctld"
    Write-Host "[2/4] Submitting SLURM job..." -ForegroundColor Yellow
    $cmd = "BATCH_SIZE=$BatchSize TRAIN_CMD='$TrainCmd' sbatch /workspace/slurm/train_mlflow_local.slurm"
    $output = docker exec -i slurmctld bash -lc $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "sbatch command failed"
    }
    $jobId = ($output | Select-String -Pattern "Submitted batch job (\d+)").Matches.Groups[1].Value

    if (-not $jobId) {
        throw "Could not parse job id from sbatch output: $output"
    }

    Write-Host "Submitted job id: $jobId" -ForegroundColor Green
    return $jobId
}

function Show-Status {
    Assert-ContainerRunning -Name "slurmctld"
    Write-Host "[3/4] Current SLURM queue:" -ForegroundColor Yellow
    docker exec -i slurmctld squeue
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read SLURM queue"
    }
}

function Show-LatestLogs {
    Assert-ContainerRunning -Name "slurmctld"
    Write-Host "[4/4] Latest logs in /workspace/logs:" -ForegroundColor Yellow
    docker exec -i slurmctld bash -lc "ls -lah /workspace/logs"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not list /workspace/logs"
    }
    docker exec -i slurmctld bash -lc "latest=\$(ls -1t /workspace/logs/*.out 2>/dev/null | head -n 1); if [ -n \"\$latest\" ]; then echo '---'; echo \"Latest: \$latest\"; tail -n 80 \"\$latest\"; else echo 'No .out logs yet'; fi"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not show latest SLURM log"
    }
}

function Run-Checks {
    Assert-ContainerRunning -Name "prometheus"
    Write-Host "[Test] Checking Prometheus metrics..." -ForegroundColor Yellow
    docker exec -i prometheus sh -lc "wget -qO- 'http://127.0.0.1:9090/api/v1/query?query=slurm_jobs_total'"
    if ($LASTEXITCODE -ne 0) {
        throw "Prometheus query slurm_jobs_total failed"
    }
    docker exec -i prometheus sh -lc "wget -qO- 'http://127.0.0.1:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
    if ($LASTEXITCODE -ne 0) {
        throw "Prometheus query slurm_job_training_energy_kwh failed"
    }
}

function Open-UIs {
    Write-Host "Opening UIs..." -ForegroundColor Yellow
    Start-Process "http://localhost:3000/d/slurm-energy-overview/slurm-energy-overview?orgId=1&from=now-30d&to=now"
    Start-Process "http://localhost:9090"
    Start-Process "http://localhost:5000"
}

Write-Header
Ensure-Folders

switch ($Action) {
    "all" {
        Start-Services
        $jobId = Submit-Job
        Show-Status
        Show-LatestLogs
        Write-Host ""
        Write-Host "Submitted job id: $jobId" -ForegroundColor Cyan
        Write-Host "Dashboard: http://localhost:3000/d/slurm-energy-overview/slurm-energy-overview?orgId=1&from=now-30d&to=now" -ForegroundColor Cyan
    }
    "services" { Start-Services }
    "submit" {
        $jobId = Submit-Job
        Write-Host "Submitted job id: $jobId" -ForegroundColor Cyan
    }
    "status" { Show-Status }
    "logs" { Show-LatestLogs }
    "test" { Run-Checks }
    "open" { Open-UIs }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
