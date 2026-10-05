[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OptimizationDir,
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    [int[]]$FreshSeeds = @(),
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$AllowInfeasible,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($OptimizationDir)) {
    $OptimizationDir = Join-Path $RepoRoot $OptimizationDir
}
if (-not [IO.Path]::IsPathRooted($Manifest)) { $Manifest = Join-Path $RepoRoot $Manifest }
$selectedPath = Join-Path $OptimizationDir "selected-controller.json"
$methodManifestPath = Join-Path $OptimizationDir "scientific-optimization-manifest.json"
if (-not (Test-Path $selectedPath) -or -not (Test-Path $methodManifestPath)) {
    throw "Scientific optimization output is incomplete: $OptimizationDir"
}
$selected = Get-Content $selectedPath -Raw | ConvertFrom-Json
$method = Get-Content $methodManifestPath -Raw | ConvertFrom-Json
if (-not [bool]$selected.quality_feasible -and -not $AllowInfeasible) {
    throw "The optimizer found no configuration satisfying the quality constraint. Use more trials or repeated Full100 seeds. Use -AllowInfeasible only for diagnosis."
}
$effectiveManifest = $Manifest
$refreshBaselines = $false
$baselineCatalog = ""

if ($FreshSeeds.Count) {
    $effectiveManifest = Join-Path $OptimizationDir "fresh-seed-validation-manifest.json"
    & python -m controller_benchmark.scientific_optimization.build_validation_manifest `
        --source $Manifest --output $effectiveManifest --seeds $FreshSeeds
    if ($LASTEXITCODE -ne 0) { throw "Could not build the fresh-seed validation manifest." }
    $refreshBaselines = $true
    Write-Host "Fresh Full100 and controller pairs will be scheduled for seeds: $($FreshSeeds -join ', ')" -ForegroundColor Cyan
}
else {
    $sourceDir = [string]$method.source_locks[0].source_dir
    & python -m controller_benchmark.optimization.source_bundle verify --source-dir $sourceDir
    if ($LASTEXITCODE -ne 0) { throw "Immutable baseline source verification failed." }
    $baselineCatalog = Join-Path $sourceDir "baseline-cache/catalog.json"
    Write-Host "Existing checksum-verified Full100 baselines will be reused." -ForegroundColor Cyan
}

$parametersJson = $selected.parameters | ConvertTo-Json -Depth 30 -Compress
$submitParameters = @{
    Manifest = $effectiveManifest
    Controller = [string]$selected.controller_plugin
    ControllerId = [string]$selected.controller_id
    ControllerParametersJson = $parametersJson
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    RequireAllStages = $true
}
if ($baselineCatalog) { $submitParameters.BaselineCatalog = $baselineCatalog }
if ($refreshBaselines) { $submitParameters.RefreshBaselines = $true }
if ($PlanOnly) { $submitParameters.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1") @submitParameters
if ($LASTEXITCODE -ne 0) { throw "Scientific Rancher validation submission failed." }

if ($PlanOnly) {
    Write-Host "Validation was planned only; no Rancher job was queued." -ForegroundColor Green
} else {
    Write-Host "Scientific validation was queued independently on Rancher." -ForegroundColor Green
    Write-Host "Your computer may be turned off after the queue confirmation above." -ForegroundColor Green
}
Write-Host "Previous benchmark runs and optimization results were not modified." -ForegroundColor Green
