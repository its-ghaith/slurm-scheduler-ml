[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$ExperimentName = "ml-energy-poc",
    [string]$Dataset = "synthcells",
    [string]$Model = "yolov8",
    [int]$Epochs = 1,
    [int]$BatchSize = 2,
    [int]$ImageSize = 640,
    [int]$Patience = 1,
    [int]$TimeoutSeconds = 1800,
    [switch]$SkipDependencyInstall,
    [switch]$WaitForCompletion
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

Write-Host "Starte Rancher-Job-Submit ohne Reset ..." -ForegroundColor Cyan
Invoke-Kubectl -Arguments @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pods")

$slurmdPod = Get-PodName -Label "app=slurmd"
$slurmctldPod = Get-PodName -Label "app=slurmctld"

Write-Host "Verwende Pods: slurmd=$slurmdPod, slurmctld=$slurmctldPod" -ForegroundColor Green

Write-Host "Bereite Workspace im slurmd-Pod vor ..." -ForegroundColor Cyan
Invoke-Kubectl -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmd",
    "-c", "slurmd",
    "--", "bash", "-lc",
    @'
mkdir -p /workspace/rotationally-invariant-cnns
mkdir -p /workspace/rotationally-invariant-cnns/data
mkdir -p /workspace/rotationally-invariant-cnns-changes
mkdir -p /workspace/slurm
mkdir -p /workspace/logs
'@
)

Copy-ToPod -Source "rotationally-invariant-cnns/src" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns/src" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns/data/synthetic_cells" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns/data/synthetic_cells" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns/pyproject.toml" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns/pyproject.toml" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns/uv.lock" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns/uv.lock" -Container "slurmd"
Copy-ToPod -Source "rotationally-invariant-cnns-changes/train_repro_phase_tracked.py" -Pod $slurmdPod -Destination "/workspace/rotationally-invariant-cnns-changes/train_repro_phase_tracked.py" -Container "slurmd"
Copy-ToPod -Source "slurm" -Pod $slurmdPod -Destination "/workspace/slurm" -Container "slurmd"

Invoke-Kubectl -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmd",
    "-c", "slurmd",
    "--", "bash", "-lc",
    "cp -a /workspace/slurm/slurm/. /workspace/slurm/ 2>/dev/null || true"
)

Copy-ToPod -Source "slurm/train_repro_rotacnn_job.slurm" -Pod $slurmctldPod -Destination "/workspace/slurm/train_repro_rotacnn_job.slurm"

Normalize-UnixLines -Deployment "slurmd" -Container "slurmd" -Paths @(
    "/workspace/slurm/train_repro_rotacnn_job.slurm",
    "/workspace/slurm/export_job_metrics_prom.py",
    "/workspace/slurm/export_job_phase_metrics_prom.py",
    "/workspace/slurm/log_job_energy_mlflow.py",
    "/workspace/rotationally-invariant-cnns-changes/train_repro_phase_tracked.py"
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
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/rotationally-invariant-cnns/src/config/dataset/synthcells.yaml"
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/rotationally-invariant-cnns/data/synthetic_cells/raw"
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/rotationally-invariant-cnns-changes/train_repro_phase_tracked.py"
Assert-PathExistsInPod -Deployment "slurmd" -Container "slurmd" -Path "/workspace/slurm/train_repro_rotacnn_job.slurm"
Assert-PathExistsInPod -Deployment "slurmctld" -Container "slurmctld" -Path "/workspace/slurm/train_repro_rotacnn_job.slurm"

$overrideString = @(
    "experiment=$ExperimentName"
    "seeds=0"
    "dataset.path=/workspace/rotationally-invariant-cnns/data/synthetic_cells"
    "dataset.data_yaml=/workspace/rotationally-invariant-cnns/data/synthetic_cells/data.yaml"
    "model.epochs=$Epochs"
    "model.batch_size=$BatchSize"
    "model.img_size=$ImageSize"
    "model.patience=$Patience"
) -join ";"

Write-Host "Reiche Job ein ..." -ForegroundColor Cyan
$submitOutput = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/slurmctld",
    "--", "bash", "-lc",
    "REPRO_DATASET=$Dataset REPRO_MODEL=$Model REPRO_OVERRIDES='$overrideString' MLFLOW_JOB_ENERGY_EXPERIMENT=$ExperimentName sbatch /workspace/slurm/train_repro_rotacnn_job.slurm"
)

Write-Host $submitOutput -ForegroundColor Green

if ($submitOutput -notmatch 'Submitted batch job (\d+)') {
    throw "Job-ID konnte nicht aus sbatch-Ausgabe gelesen werden."
}

$jobId = [int]$Matches[1]
Write-Host "Job-ID: $jobId" -ForegroundColor Green

if (-not $WaitForCompletion) {
    Write-Host "Job wurde eingereicht. Kein Warten auf Completion, da -WaitForCompletion nicht gesetzt ist." -ForegroundColor Yellow
    return
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
    throw "Job $jobId endete mit Status $finalState.`n$jobLogs"
}

$mlflowCheck = Get-KubectlOutput -Arguments @(
    "--kubeconfig", $Kubeconfig,
    "-n", $Namespace,
    "exec", "deploy/mlflow",
    "--", "bash", "-lc",
    @"
python - <<'PY'
from mlflow.tracking import MlflowClient
client = MlflowClient('http://localhost:5000')
exp = client.get_experiment_by_name('$ExperimentName')
if exp is None:
    raise SystemExit('Experiment not found')
runs = client.search_runs([exp.experiment_id], order_by=['attributes.start_time DESC'], max_results=5)
matching = [r for r in runs if r.data.tags.get('mlflow.runName') == 'job-energy-$jobId']
if not matching:
    raise SystemExit('Matching MLflow run not found')
r = matching[0]
print('run_id=' + r.info.run_id)
print('status=' + r.info.status)
PY
"@
)

Write-Host ""
Write-Host "Job erfolgreich abgeschlossen." -ForegroundColor Green
Write-Host "Job-ID: $jobId" -ForegroundColor Green
Write-Host $mlflowCheck
