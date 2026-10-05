[CmdletBinding()]
param(
    [string]$Manifest = "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json",
    [string]$Controllers = "controller_benchmark/config/live-shadow-controllers.json",
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Manifest)) { $Manifest = Join-Path $RepoRoot $Manifest }
if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }

$configuration = Get-Content $Manifest -Raw | ConvertFrom-Json
$cases = @($configuration.stages | ForEach-Object { $_.cases })
$pretrained = @($cases | Where-Object { $_.pretrained -eq $true })
$scratchControls = @($cases | Where-Object { $_.pretrained -eq $false })
$classification = @($cases | Where-Object { $_.task_type -eq "image_classification" })
$segmentation = @($cases | Where-Object { $_.task_type -eq "semantic_segmentation" })

if ($cases.Count -ne 12) { throw "Expected 12 cases: nine pretrained and three segmentation scratch controls." }
if ($pretrained.Count -ne 9) { throw "Expected exactly nine genuinely pretrained cases." }
if ($scratchControls.Count -ne 3 -or @($scratchControls | Where-Object { $_.task_type -ne "semantic_segmentation" }).Count -ne 0) {
    throw "Only the three architecture-matched segmentation scratch controls may be non-pretrained."
}
if (@($classification | Where-Object { $_.primary_metric -ne "macro_f1" }).Count -ne 0) {
    throw "Every classification case must use Macro-F1 as primary quality."
}
if (@($segmentation | Group-Object dataset_key | Where-Object { $_.Count -ne 2 }).Count -ne 0) {
    throw "Every segmentation dataset requires one scratch and one pretrained ResNet18-U-Net case."
}

if (-not $PlanOnly) {
    $effectiveKubeconfig = $Kubeconfig
    if (-not [IO.Path]::IsPathRooted($effectiveKubeconfig)) {
        $effectiveKubeconfig = Join-Path $RepoRoot $effectiveKubeconfig
    }
    $slurmdPod = ((& kubectl --kubeconfig $effectiveKubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
    if ($LASTEXITCODE -ne 0 -or -not $slurmdPod) { throw "slurmd is not running." }
    $preflight = @'
set -euo pipefail
export TORCH_HOME=/workspace-cache/controller-pretrained-weights/torch
export YOLO_CONFIG_DIR=/workspace-cache/controller-pretrained-weights/ultralytics
mkdir -p "$TORCH_HOME" "$YOLO_CONFIG_DIR" /workspace-cache/controller-pretrained-weights
python - <<'PY'
import shutil
from pathlib import Path
from torchvision import models
from ultralytics.utils.downloads import attempt_download_asset

models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
target = Path('/workspace-cache/controller-pretrained-weights/yolov8n.pt')
if not target.exists():
    source = Path(attempt_download_asset('yolov8n.pt'))
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
if not target.is_file() or target.stat().st_size < 1_000_000:
    raise RuntimeError(f'Invalid YOLO pretrained checkpoint: {target}')
print(f'Pretrained cache ready: {target} ({target.stat().st_size} bytes)')
PY
'@
    $encodedPreflight = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($preflight))
    & kubectl --kubeconfig $effectiveKubeconfig -n $Namespace exec $slurmdPod -c slurmd -- bash -lc "echo '$encodedPreflight' | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) { throw "Could not prepare the persistent pretrained-weight cache." }
}

$arguments = @{
    Manifest = $Manifest
    Controllers = $Controllers
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    ExpectedCases = 12
    BenchmarkVersion = "controller-live-shadow-pretrained-nine-dataset-v1"
    DashboardFile = "live-shadow-pretrained-nine-dataset.json"
    ExperimentName = "controller-live-shadow-pretrained-nine-dataset"
    DashboardTitle = "Live Shadow Controllers - Pretrained Nine-Dataset Study"
    DashboardUid = "controller-live-shadow-pretrained-nine"
    SubmissionLabel = "live-shadow-pretrained-nine-dataset"
}
if (-not [string]::IsNullOrWhiteSpace($RunId)) { $arguments.RunId = $RunId }
if ($PlanOnly) { $arguments.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-live-shadow-nine-dataset-async.ps1") @arguments
