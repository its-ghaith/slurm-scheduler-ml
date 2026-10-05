[CmdletBinding()]
param(
    [string[]]$SourceRunIds = @("20260729T170915Z-rapec-v3-v8-nine-dataset"),
    [string]$AnalysisId = "",
    [ValidateSet("surrogate", "direct")][string]$Mode = "surrogate",
    [ValidateRange(32, 65536)][int]$DesignSamples = 2048,
    [ValidateRange(16, 65536)][int]$SobolBaseSamples = 2048,
    [ValidateRange(64, 2000)][int]$Trees = 600,
    [ValidateRange(32, 10000)][int]$BootstrapResamples = 512,
    [ValidateRange(1, 5)][int]$TopParameters = 5,
    [ValidateRange(0, 10)][int]$TopInteractions = 5,
    [ValidateRange(3, 30)][int]$AleBins = 12,
    [int]$Seed = 20260802,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$Resume,
    [switch]$StartInBackground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$SearchSpace = "controller_benchmark/v8_sensitivity/search-space-focused.json"

if (-not $AnalysisId) {
    $AnalysisId = "rapec-v8-focused-sobol-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}

$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "run-rapec-v8-sobol-sensitivity.ps1"),
    "-SourceRunIds", $SourceRunIds,
    "-AnalysisId", $AnalysisId,
    "-Mode", $Mode,
    "-DesignSamples", $DesignSamples,
    "-SobolBaseSamples", $SobolBaseSamples,
    "-Trees", $Trees,
    "-BootstrapResamples", $BootstrapResamples,
    "-TopParameters", $TopParameters,
    "-TopInteractions", $TopInteractions,
    "-AleBins", $AleBins,
    "-Seed", $Seed,
    "-SearchSpace", $SearchSpace,
    "-Kubeconfig", $Kubeconfig,
    "-Namespace", $Namespace
)

if ($SkipDependencyInstall) { $arguments += "-SkipDependencyInstall" }
if ($Resume) { $arguments += "-Resume" }
if ($StartInBackground) { $arguments += "-StartInBackground" }

Write-Host "Focused RAPEC-v8 Sobol sensitivity analysis uses only the five dominant parameters." -ForegroundColor Cyan
Write-Host "Output AnalysisId: $AnalysisId" -ForegroundColor Cyan
Write-Host "Watcher: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v8-sobol-sensitivity.ps1 -AnalysisId '$AnalysisId'" -ForegroundColor Cyan

& powershell @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Focused RAPEC-v8 Sobol sensitivity analysis failed."
}
