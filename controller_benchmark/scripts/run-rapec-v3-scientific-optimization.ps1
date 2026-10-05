[CmdletBinding()]
param(
    [string[]]$SourceRunIds = @("20260729T170915Z-rapec-v3-v8-nine-dataset"),
    [string]$OptimizationId = "",
    [ValidateSet("qnehvi", "sobol")][string]$Backend = "qnehvi",
    [int]$Trials = 96,
    [int]$TrialsPerFold = 64,
    [int]$InitialTrials = 24,
    [int]$SensitivityTrajectories = 8,
    [int]$SensitivityTopK = 8,
    [double]$NoninferiorityProbability = 0.95,
    [int]$BootstrapSamples = 1024,
    [int]$Seed = 20260801,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$SkipOuterFolds,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not $OptimizationId) {
    $OptimizationId = "rapec-v3-ti-scientific-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if ($OptimizationId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
    throw "OptimizationId must contain 1-96 letters, digits, dots, underscores, or hyphens."
}
if (-not $SourceRunIds.Count) { throw "At least one SourceRunId is required." }

$sourceRoot = Join-Path $RepoRoot "results/controller-optimization/sources"
$sourceDirs = @()
foreach ($sourceRunId in $SourceRunIds) {
    $sourceDir = Join-Path $sourceRoot $sourceRunId
    if (-not (Test-Path $sourceDir)) {
        & (Join-Path $PSScriptRoot "export-controller-optimization-source.ps1") `
            -RunId $sourceRunId `
            -DestinationRoot $sourceRoot `
            -Kubeconfig $Kubeconfig `
            -Namespace $Namespace
        if ($LASTEXITCODE -ne 0) { throw "Optimization source export failed: $sourceRunId" }
    }
    & python -m controller_benchmark.optimization.source_bundle verify --source-dir $sourceDir
    if ($LASTEXITCODE -ne 0) { throw "Immutable source verification failed: $sourceRunId" }
    $sourceDirs += $sourceDir
}

$venvRoot = Join-Path $RepoRoot "results/controller-scientific-optimization/.venv"
$venvPython = Join-Path $venvRoot "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    & python -m venv --system-site-packages $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not create the scientific optimization environment." }
}
if (-not $SkipDependencyInstall) {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $venvPython -c "import torch, botorch, gpytorch, SALib, numpy" 2>$null
    $dependencyStatus = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($dependencyStatus -ne 0) {
        & $venvPython -m pip install -r "controller_benchmark/scientific_optimization/requirements.txt"
        if ($LASTEXITCODE -ne 0) { throw "Could not install scientific optimization dependencies." }
    }
}

$outputDir = Join-Path $RepoRoot "results/controller-scientific-optimization/runs/$OptimizationId"
if ((Test-Path $outputDir) -and -not $Resume) {
    throw "Output already exists and is protected from overwrite: $outputDir"
}
$arguments = @(
    "-u",
    "-m", "controller_benchmark.scientific_optimization.optimize",
    "--output-dir", $outputDir,
    "--optimization-id", $OptimizationId,
    "--backend", $Backend,
    "--trials", $Trials,
    "--trials-per-fold", $TrialsPerFold,
    "--initial-trials", $InitialTrials,
    "--sensitivity-trajectories", $SensitivityTrajectories,
    "--sensitivity-top-k", $SensitivityTopK,
    "--noninferiority-probability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--bootstrap-samples", $BootstrapSamples,
    "--seed", $Seed
)
foreach ($sourceDir in $sourceDirs) { $arguments += @("--source-dir", $sourceDir) }
if ($SkipOuterFolds) { $arguments += "--skip-outer-folds" }
if ($Resume) { $arguments += "--resume" }

Write-Host "Scientific optimization ID: $OptimizationId" -ForegroundColor Cyan
Write-Host "Progress output: $outputDir/progress.jsonl" -ForegroundColor Cyan
Write-Host "Live status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v3-scientific-optimization.ps1 -OptimizationId '$OptimizationId'" -ForegroundColor Cyan
Write-Host "The qLogNEHVI pure-Python fallback is valid but slower when Ninja/C++ is unavailable." -ForegroundColor DarkGray
if ($Resume) {
    Write-Host "Resume mode: completed search sections are reused; empty interrupted sections restart safely." -ForegroundColor Yellow
}

& $venvPython @arguments
if ($LASTEXITCODE -ne 0) { throw "Scientific RAPEC-v3 optimization failed." }

Write-Host "Scientific optimization completed: $outputDir" -ForegroundColor Green
Write-Host "Nash-selected controller: $outputDir/selected-controller.json" -ForegroundColor Green
Write-Host "All old controllers, source bundles, baselines, and results remain unchanged." -ForegroundColor Green
Write-Host "No Rancher training job was submitted." -ForegroundColor Green
