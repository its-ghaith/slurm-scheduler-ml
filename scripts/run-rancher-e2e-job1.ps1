[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$ExperimentName = "ml-energy-poc",
    [ValidateSet("carpk")]
    [string]$Dataset = "carpk",
    [string]$Model = "yolov8",
    [int]$Epochs = 1,
    [int]$BatchSize = 2,
    [int]$ImageSize = 640,
    [int]$Patience = 1,
    [int]$TimeoutSeconds = 1800,
    [switch]$AdaptiveEnabled,
    [int]$AdaptiveMinEpochs = 20,
    [int]$AdaptivePatience = 3,
    [int]$AdaptiveSmoothingWindow = 3,
    [double]$AdaptiveMinDeltaMap50 = 0.001,
    [double]$AdaptiveMinMapeMap50PerWh = 0.0001,
    [switch]$SkipDependencyInstall
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

function Get-PodName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    return Get-KubectlOutput -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "get", "pod",
        "-l", $Label,
        "-o", "jsonpath={.items[0].metadata.name}"
    )
}

function Wait-ForDeployment {
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

function Copy-ToPod {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Pod,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [string]$Container
    )

    $args = @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "cp", $Source, "${Pod}:$Destination"
    )

    if ($Container) {
        $args += @("-c", $Container)
    }

    Invoke-Kubectl -Arguments $args
}

function Normalize-UnixLines {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Deployment,
        [Parameter(Mandatory = $true)]
        [string]$Container,
        [Parameter(Mandatory = $true)]
        [string[]]$Paths
    )

    $joined = ($Paths | ForEach-Object { "'$_'" }) -join " "
    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/$Deployment",
        "-c", $Container,
        "--", "bash", "-lc",
        "for f in $joined; do if [ -f ""`$f"" ]; then sed -i 's/\r$//' ""`$f""; fi; done"
    )
}

function Assert-PathExistsInPod {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Deployment,
        [Parameter(Mandatory = $true)]
        [string]$Container,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/$Deployment",
        "-c", $Container,
        "--", "bash", "-lc",
        "test -e '$Path'"
    )
}

function Test-SlurmdDependencies {
    $pythonCode = "import importlib, sys`nrequired = ['codecarbon', 'mlflow', 'torch', 'ultralytics', 'cv2']`nmissing = []`nfor name in required:`n    try:`n        importlib.import_module(name)`n    except Exception:`n        missing.append(name)`nprint('missing=' + ','.join(missing) if missing else 'deps-ok')`nsys.exit(1 if missing else 0)"
    $output = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmd -c slurmd -- python -c $pythonCode 2>&1
    if ($LASTEXITCODE -eq 0) {
        return $true
    }

    Write-Host ($output | Out-String) -ForegroundColor Yellow
    return $false
}

Write-Host "Starte Rancher E2E-Test mit frischem Reset ..." -ForegroundColor Cyan

$resetScript = Join-Path $PSScriptRoot "reset-rancher-slurm-observability.ps1"
if (-not (Test-Path $resetScript)) {
    throw "Reset-Skript nicht gefunden: $resetScript"
}

& powershell -ExecutionPolicy Bypass -File $resetScript -Force
if ($LASTEXITCODE -ne 0) {
    throw "Reset-Skript ist fehlgeschlagen."
}

Write-Host "Warte auf zentrale Deployments ..." -ForegroundColor Cyan
@("slurmctld", "slurmd", "prometheus", "grafana", "mlflow") | ForEach-Object {
    Wait-ForDeployment -Deployment $_
}

$slurmdPod = Get-PodName -Label "app=slurmd"
$slurmctldPod = Get-PodName -Label "app=slurmctld"

Write-Host "Verwende Pods: slurmd=$slurmdPod, slurmctld=$slurmctldPod" -ForegroundColor Green

Write-Host "Bereite Workspace im slurmd-Pod vor ..." -ForegroundColor Cyan
Invoke-Kubectl -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", $slurmdPod,
    "-c", "slurmd",
    "--", "bash", "-lc",
    @'
mkdir -p /workspace/rotationally-invariant-cnns
mkdir -p /workspace/rotationally-invariant-cnns/data
mkdir -p /workspace/slurm
'@
)

Copy-ToPod -Source "rotationally-invariant-cnns/src" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns/data/carpk" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns/data" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns/pyproject.toml" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns/uv.lock" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns" -Container "slurmd"
Copy-ToPod -Source "slurm" -Pod $slurmdPod -Destination "/workspace" -Container "slurmd"

Invoke-Kubectl -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmd",
    "-c", "slurmd",
    "--", "bash", "-lc",
    "cp -a /workspace/slurm/slurm/. /workspace/slurm/ 2>/dev/null || true"
)

Copy-ToPod -Source "slurm/train_repro_rotacnn_job.slurm" -Pod $slurmctldPod -Destination "/workspace/slurm"

Normalize-UnixLines -Deployment "slurmd" -Container "slurmd" -Paths @(
    "/workspace/slurm/train_repro_rotacnn_job.slurm",
    "/workspace/slurm/gpu_energy_utils.py",
    "/workspace/slurm/epoch_energy_controller.py",
    "/workspace/slurm/export_job_metrics_prom.py",
    "/workspace/slurm/export_job_epoch_metrics_prom.py",
    "/workspace/slurm/export_job_phase_metrics_prom.py",
    "/workspace/slurm/log_job_energy_mlflow.py",
    "/workspace/slurm/summarize_gpu_metrics.py",
    "/workspace/slurm/summarize_gpu_metrics_epochs.py",
    "/workspace/slurm/summarize_gpu_metrics_phases.py",
    "/workspace/slurm/train_repro_phase_tracked.py"
)
Normalize-UnixLines -Deployment "slurmctld" -Container "slurmctld" -Paths @(
    "/workspace/slurm/train_repro_rotacnn_job.slurm"
)

if (-not $SkipDependencyInstall) {
    Write-Host "Pruefe und installiere Laufzeitabhaengigkeiten im slurmd-Pod ..." -ForegroundColor Cyan
    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/slurmd",
        "-c", "slurmd",
        "--", "bash", "-lc",
        @'
apt-get update >/dev/null
apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
python -m pip install -q 'numpy<2' ultralytics codecarbon mlflow hydra-core matplotlib pandas scikit-image seaborn tensorboard torchmetrics transformers tqdm albumentations opencv-python optuna plyer groundingdino-py jwt
'@
    )
}
elseif (-not (Test-SlurmdDependencies)) {
    throw "Im slurmd-Pod fehlen Laufzeitabhaengigkeiten. Starte das Skript ohne -SkipDependencyInstall oder installiere die benoetigten Pakete zuerst."
}

if (-not (Test-SlurmdDependencies)) {
    throw "Abhaengigkeitspruefung nach der Vorbereitung fehlgeschlagen."
}

Write-Host "Validiere vorbereiteten Workspace ..." -ForegroundColor Cyan
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/rotationally-invariant-cnns/src/config/dataset/carpk.yaml"
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/rotationally-invariant-cnns/data/carpk/raw"
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/slurm/train_repro_phase_tracked.py"
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/slurm/train_repro_rotacnn_job.slurm"
Assert-PathExistsInPod -Deployment "slurmctld" -Container "slurmctld" -Path "/workspace/slurm/train_repro_rotacnn_job.slurm"

Write-Host "Reiche E2E-Job ein ..." -ForegroundColor Cyan
$overrideString = @(
    "experiment=$ExperimentName"
    "seeds=0"
    "dataset.path=/workspace/rotationally-invariant-cnns/data/carpk"
    "dataset.data_yaml=/workspace/rotationally-invariant-cnns/data/carpk/data.yaml"
    "model.epochs=$Epochs"
    "model.batch_size=$BatchSize"
    "model.img_size=$ImageSize"
    "model.patience=$Patience"
) -join ";"

$submitPairs = @(
    "REPRO_DATASET=$Dataset",
    "REPRO_MODEL=$Model",
    "REPRO_OVERRIDES='$overrideString'",
    "MLFLOW_JOB_ENERGY_EXPERIMENT=$ExperimentName",
    "ENERGY_ADAPTIVE_ENABLED=$($AdaptiveEnabled.IsPresent.ToString().ToLower())",
    "ENERGY_ADAPTIVE_MIN_EPOCHS=$AdaptiveMinEpochs",
    "ENERGY_ADAPTIVE_PATIENCE=$AdaptivePatience",
    "ENERGY_ADAPTIVE_SMOOTHING_WINDOW=$AdaptiveSmoothingWindow",
    "ENERGY_ADAPTIVE_MIN_DELTA_MAP50=$AdaptiveMinDeltaMap50",
    "ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH=$AdaptiveMinMapeMap50PerWh"
)
$submitCommand = ($submitPairs -join " ") + " sbatch /workspace/slurm/train_repro_rotacnn_job.slurm"
$submitOutput = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmctld",
    "--", "bash", "-lc",
    $submitCommand
)

Write-Host $submitOutput -ForegroundColor Green

if ($submitOutput -notmatch 'Submitted batch job (\d+)') {
    throw "Job-ID konnte nicht aus sbatch-Ausgabe gelesen werden."
}

$jobId = [int]$Matches[1]
if ($jobId -ne 1) {
    throw "Erwartet war Job-ID 1 nach Reset, erhalten: $jobId"
}

Write-Host "Warte auf Abschluss von Job $jobId ..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$finalState = $null

while ((Get-Date) -lt $deadline) {
    $jobInfo = Get-KubectlOutput -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/slurmctld",
        "--", "bash", "-lc",
        "scontrol show job $jobId 2>/dev/null || true"
    )

    if ($jobInfo -match 'JobState=([A-Z_]+)') {
        $finalState = $Matches[1]
        Write-Host "Aktueller Job-Status: $finalState" -ForegroundColor DarkCyan
        if ($finalState -in @("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")) {
            break
        }
    }

    Start-Sleep -Seconds 10
}

if (-not $finalState) {
    throw "Job-Status konnte nicht ermittelt werden."
}

if ($finalState -ne "COMPLETED") {
    $jobLogs = Get-KubectlOutput -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/slurmd",
        "-c", "slurmd",
        "--", "bash", "-lc",
        "echo '--- OUT ---'; tail -n 120 /workspace/logs/rotacnn-repro-energy-$jobId.out 2>/dev/null || true; echo '--- ERR ---'; tail -n 120 /workspace/logs/rotacnn-repro-energy-$jobId.err 2>/dev/null || true"
    )
    throw "E2E-Job $jobId endete mit Status $finalState.`n$jobLogs"
}

Write-Host "Pruefe, dass nur ein Job-Metrics-Satz existiert ..." -ForegroundColor Cyan
$promFiles = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmd",
    "-c", "slurmd",
    "--", "bash", "-lc",
    "find /workspace/energy_metrics/node_exporter -maxdepth 1 -type f -name 'job_*.prom' | sort"
)

$promFileList = @(
    $promFiles -split "[`r`n]+" |
    Where-Object { $_ -match '/job_\d+(_phases)?\.prom$' }
)

$jobPromFiles = @(
    $promFileList | Where-Object { $_ -match '/job_\d+\.prom$' }
)

if ($jobPromFiles.Count -ne 1 -or $jobPromFiles[0] -notmatch "/job_$jobId\.prom$") {
    throw "Es sollte genau eine Job-Prometheus-Datei fuer Job $jobId geben. Gefunden: $($jobPromFiles -join ', ')"
}

Write-Host "Validiere Prometheus-Metriken fuer Job $jobId ..." -ForegroundColor Cyan
$promQuery = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/prometheus",
    "--", "sh", "-lc",
    "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_phase_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_phase_codecarbon_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_epoch_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_compare_rel_diff_pct%7Bjob_id%3D%22$jobId%22%7D'"
)

if ($promQuery -notmatch '"result":\[\{') {
    throw "Prometheus liefert keine Metriken fuer Job $jobId.`n$promQuery"
}

Write-Host "Validiere MLflow-Run ..." -ForegroundColor Cyan
$mlflowPython = "from mlflow.tracking import MlflowClient; client=MlflowClient('http://localhost:5000'); exp=client.get_experiment_by_name('$ExperimentName'); assert exp is not None, 'Experiment not found'; runs=client.search_runs([exp.experiment_id], order_by=['attributes.start_time DESC'], max_results=5); matching=[r for r in runs if r.data.tags.get('mlflow.runName') == 'job-energy-$jobId']; assert len(matching)==1, f'Expected exactly one MLflow run for job-energy-$jobId, found {len(matching)}'; r=matching[0]; print('run_id=' + r.info.run_id); print('status=' + r.info.status); [print(f'{key}=' + str(r.data.metrics.get(key))) for key in ['metrics/mAP50B','metrics/mAP50-95B','training_energy_kwh','estimated_electricity_cost_eur','gpu_util_avg_pct']]"
$mlflowCheck = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/mlflow",
    "--", "python", "-c", $mlflowPython
)

if ($mlflowCheck -notmatch 'status=FINISHED') {
    throw "MLflow-Run fuer Job $jobId ist nicht erfolgreich.`n$mlflowCheck"
}

$summary = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmd",
    "-c", "slurmd",
    "--", "bash", "-lc",
    "cat /workspace/energy_metrics/gpu_summary_job_${jobId}_phases.json"
)

Write-Host ""
Write-Host "E2E-Test erfolgreich abgeschlossen." -ForegroundColor Green
Write-Host "Job-ID: $jobId" -ForegroundColor Green
Write-Host "MLflow Experiment: $ExperimentName" -ForegroundColor Green
Write-Host "Prometheus-Dateien: $($jobPromFiles -join ', ')" -ForegroundColor Green
Write-Host ""
Write-Host "MLflow-Auszug:" -ForegroundColor Cyan
Write-Host $mlflowCheck
Write-Host ""
Write-Host "Energy Summary:" -ForegroundColor Cyan
Write-Host $summary
