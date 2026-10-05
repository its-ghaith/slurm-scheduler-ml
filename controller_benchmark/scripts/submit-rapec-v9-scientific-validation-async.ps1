[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OptimizationDir,
    [string]$Manifest = "controller_benchmark/config/benchmark-v4-rapec-v9-nine-dataset.json",
    [object]$FreshSeeds = @(),
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$RefreshBaselines,
    [switch]$AllowInfeasible,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$freshSeedList = @()
foreach ($value in @($FreshSeeds)) {
    foreach ($token in ([string]$value).Split(',')) {
        if (-not $token.Trim()) { continue }
        $parsed = 0
        if (-not [int]::TryParse($token.Trim(), [ref]$parsed) -or $parsed -lt 0) {
            throw "FreshSeeds must contain non-negative comma-separated integers: $token"
        }
        $freshSeedList += $parsed
    }
}
$freshSeedList = @($freshSeedList | Select-Object -Unique)
if (-not [IO.Path]::IsPathRooted($OptimizationDir)) {
    $OptimizationDir = Join-Path $RepoRoot $OptimizationDir
}
if (-not [IO.Path]::IsPathRooted($Manifest)) {
    $Manifest = Join-Path $RepoRoot $Manifest
}
$selectedPath = Join-Path $OptimizationDir "selected-controller.json"
if (-not (Test-Path $selectedPath)) {
    throw "RAPEC-v9 optimization output is incomplete: $OptimizationDir"
}
$selected = Get-Content $selectedPath -Raw | ConvertFrom-Json
if (-not [bool]$selected.quality_feasible -and -not $AllowInfeasible) {
    throw "No quality-feasible RAPEC-v9 configuration was found."
}

$effectiveManifest = $Manifest
$baselineCatalog = ""
$refreshEffectiveBaselines = [bool]$RefreshBaselines
if ($freshSeedList.Count) {
    $effectiveManifest = Join-Path $OptimizationDir "fresh-seed-validation-manifest.json"
    & python -m controller_benchmark.scientific_optimization.build_validation_manifest `
        --source $Manifest --output $effectiveManifest --seeds $freshSeedList
    if ($LASTEXITCODE -ne 0) { throw "Could not build the fresh-seed validation manifest." }
    $refreshEffectiveBaselines = $true
    Write-Host "Fresh Full100/controller pairs will be scheduled for seeds: $($freshSeedList -join ', ')" -ForegroundColor Cyan
}
else {
    $methodManifestPath = @(
        (Join-Path $OptimizationDir "optimizer-benchmark-manifest.json"),
        (Join-Path $OptimizationDir "scientific-optimization-manifest.json")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($methodManifestPath) {
        $method = Get-Content $methodManifestPath -Raw | ConvertFrom-Json
        $sourceDir = [string]$method.source_locks[0].source_dir
        & python -m controller_benchmark.optimization.source_bundle verify --source-dir $sourceDir
        if ($LASTEXITCODE -ne 0) { throw "Immutable baseline source verification failed." }
        $baselineCatalog = Join-Path $sourceDir "baseline-cache/catalog.json"
        Write-Host "Existing checksum-verified Full100 baselines will be reused." -ForegroundColor Cyan
    }
    else {
        $baselineCatalog = "results/controller-baselines/controller-benchmark-v4-rapec-v9-risk-constrained/catalog.json"
    }
}

$submit = @{
    Manifest = $effectiveManifest
    Controller = [string]$selected.controller_plugin
    ControllerId = [string]$selected.controller_id
    ControllerParametersJson = ($selected.parameters | ConvertTo-Json -Depth 30 -Compress)
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    RequireAllStages = $true
}
if ($baselineCatalog) { $submit.BaselineCatalog = $baselineCatalog }
if ($refreshEffectiveBaselines) { $submit.RefreshBaselines = $true }
if ($PlanOnly) { $submit.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1") @submit
if ($LASTEXITCODE -ne 0) { throw "RAPEC-v9 validation submission failed." }

Write-Host "Existing results were not modified." -ForegroundColor Green
if (-not $PlanOnly) {
    Write-Host "The validation is queued on Rancher; this computer may be turned off." -ForegroundColor Green
}
