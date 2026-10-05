[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
$generatedRoot = Join-Path $RepoRoot "results\controller-benchmark\generated-manifests"
$manifest = Join-Path $generatedRoot "rapec-g-v2-small-validation-v1.json"
New-Item -ItemType Directory -Force $generatedRoot | Out-Null

& python -m controller_benchmark.rapec_g_v2_small_manifest `
    --vision-source "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json" `
    --classification-source "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json" `
    --output $manifest
if ($LASTEXITCODE -ne 0) { throw "RAPEC-G v2 small manifest generation failed." }

$arguments = @{
    Manifest = $manifest
    Controllers = "controller_benchmark/config/generalized-live-shadow-v2-small-controllers.json"
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    ExpectedCases = 6
    BenchmarkVersion = "rapec-g-v2-small-validation-v1"
    DashboardFile = "rapec-g-v2-small-validation.json"
    ExperimentName = "rapec-g-v2-small-validation-v1"
    DashboardTitle = "RAPEC-G v2 Small Cross-Domain Validation"
    DashboardUid = "rapec-g-v2-small-validation-v1"
    SubmissionLabel = "rapec-g-v2-small"
}
if (-not [string]::IsNullOrWhiteSpace($RunId)) { $arguments.RunId = $RunId }
if ($PlanOnly) { $arguments.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-live-shadow-nine-dataset-async.ps1") @arguments
if ($LASTEXITCODE -ne 0) { throw "RAPEC-G v2 small validation submission failed." }
