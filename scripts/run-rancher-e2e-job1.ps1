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
    [switch]$SkipDependencyInstall,
    [switch]$ForceWorkspaceSync,
    [switch]$ForceDatasetSync
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
        "cp"
    )

    if ($Container) {
        $args += @("-c", $Container)
    }

    $args += @($Source, "${Pod}:$Destination")

    Invoke-Kubectl -Arguments $args
}

function Copy-DirectoryToPod {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Pod,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [string]$Container
    )

    $resolvedSource = (Resolve-Path $Source).Path
    $repoRoot = (Get-Location).Path
    $rootUri = New-Object System.Uri(($repoRoot.TrimEnd('\') + '\'))
    $sourceUri = New-Object System.Uri($resolvedSource)
    $relativeSource = [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($sourceUri).ToString()).Replace('/', '\')

    if ([System.IO.Path]::IsPathRooted($relativeSource)) {
        throw "Konnte keinen relativen Pfad für kubectl cp aus $Source ableiten."
    }

    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "cp",
        "-c", $Container,
        $relativeSource,
        "${Pod}:$Destination"
    )
}

function Get-LocalFileCount {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Get-ChildItem -Path $Path -Recurse -File | Measure-Object).Count
}

function Get-RemoteFileCount {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Deployment,
        [Parameter(Mandatory = $true)]
        [string]$Container,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $output = Get-KubectlOutput -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/$Deployment",
        "-c", $Container,
        "--", "bash", "-lc",
        "if [ -d '$Path' ]; then find '$Path' -type f | wc -l; else echo 0; fi"
    )

    return [int]($output.Trim())
}

function Test-RemotePathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Deployment,
        [Parameter(Mandatory = $true)]
        [string]$Container,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $output = Get-KubectlOutput -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/$Deployment",
        "-c", $Container,
        "--", "bash", "-lc",
        "if [ -e '$Path' ]; then echo exists; else echo missing; fi"
    )

    return $output.Trim() -eq "exists"
}

function New-WorkspaceBundle {
    $bundleRootRelative = Join-Path ".tmp_rancher_workspace" ("rancher-workspace-" + [guid]::NewGuid().ToString("N"))
    $bundleDirRelative = Join-Path $bundleRootRelative "workspace_bundle"
    $repoDir = Join-Path $bundleDirRelative "rotationally-invariant-cnns"
    $slurmDir = Join-Path $bundleDirRelative "slurm"

    New-Item -ItemType Directory -Force -Path $repoDir, $slurmDir | Out-Null
    Copy-Item -Recurse -Force "rotationally-invariant-cnns/src" $repoDir
    if (Test-Path "rotationally-invariant-cnns/pyproject.toml") {
        Copy-Item -Force "rotationally-invariant-cnns/pyproject.toml" $repoDir
    }
    if (Test-Path "rotationally-invariant-cnns/uv.lock") {
        Copy-Item -Force "rotationally-invariant-cnns/uv.lock" $repoDir
    }
    Copy-Item -Recurse -Force "slurm/*" $slurmDir

    return @{
        Root = $bundleRootRelative
        Bundle = $bundleDirRelative
    }
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
    $dependencyCheck = @'
python - <<'PY'
import importlib
import sys

required = ['codecarbon', 'mlflow', 'torch', 'ultralytics', 'cv2']
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

print('missing=' + ','.join(missing) if missing else 'deps-ok')
sys.exit(1 if missing else 0)
PY
status=$?
echo "__DEP_EXIT__${status}"
exit 0
'@
    $output = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmd -c slurmd -- bash -lc $dependencyCheck 2>&1
    $outputText = ($output | Out-String).Trim()
    $exitMatch = [regex]::Match($outputText, "__DEP_EXIT__(\d+)")
    $exitCode = if ($exitMatch.Success) { [int]$exitMatch.Groups[1].Value } else { 1 }
    $cleanOutput = ($outputText -replace "__DEP_EXIT__\d+", "").Trim()

    if ($exitCode -eq 0) {
        return $true
    }

    if ($cleanOutput) {
        Write-Host $cleanOutput -ForegroundColor Yellow
    }
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
mkdir -p /workspace-cache
'@
)

$bundle = New-WorkspaceBundle
try {
    if ($ForceWorkspaceSync) {
        Write-Host "Erzwinge frische Code-/SLURM-Synchronisierung ..." -ForegroundColor Yellow
        Invoke-Kubectl -Arguments @(
            "--kubeconfig", $Kubeconfig,
            "-n", $Namespace,
            "exec", $slurmdPod,
            "-c", "slurmd",
            "--", "bash", "-lc",
            "rm -rf /workspace/workspace_bundle /workspace/rotationally-invariant-cnns/src /workspace/slurm/*"
        )
    }

    $optionalConfigFiles = @("rotationally-invariant-cnns/pyproject.toml", "rotationally-invariant-cnns/uv.lock") |
        Where-Object { Test-Path $_ }
    $codeFileCount = (Get-LocalFileCount "rotationally-invariant-cnns/src") + (Get-LocalFileCount "slurm") + $optionalConfigFiles.Count
    Write-Host "Synchronisiere Code und SLURM-Skripte gebündelt ($codeFileCount Dateien) ..." -ForegroundColor Cyan
    Copy-DirectoryToPod -Source $bundle.Bundle -Pod $slurmdPod -Destination "/workspace" -Container "slurmd"

    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", "deploy/slurmd",
        "-c", "slurmd",
        "--", "bash", "-lc",
        @'
mkdir -p /workspace/rotationally-invariant-cnns /workspace/slurm
cp -a /workspace/workspace_bundle/rotationally-invariant-cnns/. /workspace/rotationally-invariant-cnns/
cp -a /workspace/workspace_bundle/slurm/. /workspace/slurm/
rm -rf /workspace/workspace_bundle
'@
    )
}
finally {
    if ($bundle -and (Test-Path $bundle.Root)) {
        Remove-Item -Recurse -Force $bundle.Root
    }
}

$localDatasetRawFiles = Get-LocalFileCount "rotationally-invariant-cnns/data/carpk/raw"
$remoteDatasetRawFiles = if ($ForceDatasetSync) { 0 } else { Get-RemoteFileCount -Deployment "slurmd" -Container "slurmd" -Path "/workspace-cache/carpk/raw" }
$remoteDatasetYamlExists = if ($ForceDatasetSync) { $false } else { Test-RemotePathExists -Deployment "slurmd" -Container "slurmd" -Path "/workspace-cache/carpk/data.yaml" }

if ($ForceDatasetSync -or $remoteDatasetRawFiles -ne $localDatasetRawFiles -or -not $remoteDatasetYamlExists) {
    if ($ForceDatasetSync) {
        Write-Host "Erzwinge frische Dataset-Synchronisierung ..." -ForegroundColor Yellow
    }
    else {
        Write-Host "Synchronisiere CARPK-Rohdataset in den persistenten slurmd-Cache (raw=$localDatasetRawFiles Dateien; cacheRaw=$remoteDatasetRawFiles; dataYaml=$remoteDatasetYamlExists) ..." -ForegroundColor Cyan
    }

    Invoke-Kubectl -Arguments @(
        "--kubeconfig", $Kubeconfig,
        "-n", $Namespace,
        "exec", $slurmdPod,
        "-c", "slurmd",
        "--", "bash", "-lc",
        "rm -rf /workspace-cache/carpk/raw && rm -f /workspace-cache/carpk/data.yaml && mkdir -p /workspace-cache/carpk"
    )

    Copy-DirectoryToPod -Source "rotationally-invariant-cnns/data/carpk" -Pod $slurmdPod -Destination "/workspace-cache" -Container "slurmd"
}
else {
    Write-Host "CARPK-Rohdataset bereits im persistenten Pod-Cache vorhanden (raw=$remoteDatasetRawFiles Dateien). Überspringe erneutes Kopieren." -ForegroundColor Green
}

Invoke-Kubectl -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", $slurmdPod,
    "-c", "slurmd",
    "--", "bash", "-lc",
    @'
mkdir -p /workspace/rotationally-invariant-cnns/data
rm -rf /workspace/rotationally-invariant-cnns/data/carpk
ln -s /workspace-cache/carpk /workspace/rotationally-invariant-cnns/data/carpk
'@
)

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
    if (Test-SlurmdDependencies) {
        Write-Host "Laufzeitabhängigkeiten bereits vorhanden. Überspringe Installation." -ForegroundColor Green
    }
    else {
        Write-Host "Installiere fehlende Laufzeitabhängigkeiten im slurmd-Pod ..." -ForegroundColor Cyan
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
}
elseif (-not (Test-SlurmdDependencies)) {
    throw "Im slurmd-Pod fehlen Laufzeitabhängigkeiten. Starte das Skript ohne -SkipDependencyInstall oder installiere die benötigten Pakete zuerst."
}

if (-not (Test-SlurmdDependencies)) {
    throw "Abhängigkeitsprüfung nach der Vorbereitung fehlgeschlagen."
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

Write-Host "Prüfe, dass nur ein Job-Metrics-Satz existiert ..." -ForegroundColor Cyan
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
    throw "Es sollte genau eine Job-Prometheus-Datei für Job $jobId geben. Gefunden: $($jobPromFiles -join ', ')"
}

Write-Host "Validiere Prometheus-Metriken für Job $jobId ..." -ForegroundColor Cyan
$promQuery = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/prometheus",
    "--", "sh", "-lc",
    "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_phase_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_phase_codecarbon_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_epoch_energy_kwh%7Bjob_id%3D%22$jobId%22%7D' && echo && wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_compare_rel_diff_pct%7Bjob_id%3D%22$jobId%22%7D'"
)

if ($promQuery -notmatch '"result":\[\{') {
    throw "Prometheus liefert keine Metriken für Job $jobId.`n$promQuery"
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
    throw "MLflow-Run für Job $jobId ist nicht erfolgreich.`n$mlflowCheck"
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
