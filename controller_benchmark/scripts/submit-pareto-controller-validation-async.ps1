[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OptimizationDir,
    [ValidateSet("conservative", "balanced", "aggressive")]
    [string[]]$Profiles = @("conservative", "balanced", "aggressive"),
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($OptimizationDir)) { $OptimizationDir = Join-Path $RepoRoot $OptimizationDir }
$selectedPath = Join-Path $OptimizationDir "selected-controllers.json"
$optimizationManifestPath = Join-Path $OptimizationDir "optimization-manifest.json"
if (-not (Test-Path $selectedPath) -or -not (Test-Path $optimizationManifestPath)) {
    throw "Optimization output is incomplete: $OptimizationDir"
}
$selected = Get-Content $selectedPath -Raw | ConvertFrom-Json
$optimizationManifest = Get-Content $optimizationManifestPath -Raw | ConvertFrom-Json
$sourceDir = [string]$optimizationManifest.source_dir
$baselineCatalog = Join-Path $sourceDir "baseline-cache/catalog.json"
& python -m controller_benchmark.optimization.source_bundle verify --source-dir $sourceDir
if ($LASTEXITCODE -ne 0) { throw "Immutable optimization source verification failed." }

$submitScript = Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1"
foreach ($profile in $Profiles) {
    $configuration = $selected.profiles.$profile
    if (-not $configuration) { throw "Profile not found: $profile" }
    $parametersJson = $configuration.parameters | ConvertTo-Json -Depth 20 -Compress
    $submitParameters = @{
        Manifest = $Manifest
        Controller = [string]$selected.controller_plugin
        ControllerId = [string]$configuration.id
        ControllerParametersJson = $parametersJson
        Kubeconfig = $Kubeconfig
        Namespace = $Namespace
        BaselineCatalog = $baselineCatalog
        RequireAllStages = $true
    }
    if ($PlanOnly) { $submitParameters.PlanOnly = $true }
    if ($PlanOnly) {
        Write-Host "Preparing Rancher validation plan for profile: $profile" -ForegroundColor Cyan
    } else {
        Write-Host "Preparing real Rancher validation for profile: $profile" -ForegroundColor Cyan
    }
    & $submitScript @submitParameters
    if ($LASTEXITCODE -ne 0) { throw "Rancher validation submission failed for $profile." }
}

if ($PlanOnly) {
    Write-Host "Selected profiles were planned only; no Rancher jobs were queued." -ForegroundColor Green
} else {
    Write-Host "Selected profiles were submitted as separate runs." -ForegroundColor Green
}
Write-Host "Existing Full100 baselines are reused by checksum; old runs remain unchanged." -ForegroundColor Green
