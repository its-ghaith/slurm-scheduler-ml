[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$generatedRoot = Join-Path $repoRoot "results\controller-benchmark\generated-manifests"
New-Item -ItemType Directory -Force -Path $generatedRoot | Out-Null
$manifest = Join-Path $generatedRoot "rapec-g-v3-small-validation-v1.json"

Push-Location $repoRoot
try {
    & python -m controller_benchmark.rapec_g_v2_small_manifest `
        --vision-source "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json" `
        --classification-source "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json" `
        --output $manifest
    if ($LASTEXITCODE -ne 0) { throw "RAPEC-G v3 small manifest generation failed." }

    $arguments = @{
        Manifest = $manifest
        Controllers = "controller_benchmark/config/generalized-live-shadow-v3-small-controllers.json"
        Kubeconfig = $Kubeconfig
        Namespace = $Namespace
        ExpectedCases = 6
        BenchmarkVersion = "rapec-g-v3-small-validation-v1"
        DashboardFile = "rapec-g-v3-small-validation.json"
        ExperimentName = "rapec-g-v3-small-validation-v1"
        DashboardTitle = "RAPEC-G v3 Small Cross-Domain Validation"
        DashboardUid = "rapec-g-v3-small-validation-v1"
        SubmissionLabel = "rapec-g-v3-small"
    }
    if ($PlanOnly) { $arguments.PlanOnly = $true }
    & (Join-Path $PSScriptRoot "submit-live-shadow-nine-dataset-async.ps1") @arguments
    if ($LASTEXITCODE -ne 0) { throw "RAPEC-G v3 small validation submission failed." }
}
finally {
    Pop-Location
}
