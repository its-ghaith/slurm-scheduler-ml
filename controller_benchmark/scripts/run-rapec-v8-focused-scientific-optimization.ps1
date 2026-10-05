[CmdletBinding()]
param(
    [string[]]$SourceRunIds = @("20260729T170915Z-rapec-v3-v8-nine-dataset"),
    [string]$OptimizationId = "",
    [ValidateSet("qnehvi", "sobol")][string]$Backend = "qnehvi",
    [int]$Trials = 96,
    [int]$TrialsPerFold = 64,
    [int]$InitialTrials = 24,
    [int]$SensitivityTrajectories = 12,
    [double]$NoninferiorityProbability = 0.95,
    [int]$BootstrapSamples = 1024,
    [int]$Seed = 20260802,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$SkipOuterFolds,
    [switch]$Resume,
    [switch]$StartInBackground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not $OptimizationId) {
    $OptimizationId = "rapec-v8-focused-scientific-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if ($OptimizationId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
    throw "OptimizationId must contain 1-96 letters, digits, dots, underscores, or hyphens."
}
if (-not $SourceRunIds.Count) { throw "At least one SourceRunId is required." }

$runsRoot = Join-Path $RepoRoot "results/controller-v8-scientific-optimization/runs"
$outputDir = Join-Path $runsRoot $OptimizationId
if ((Test-Path $outputDir) -and -not $Resume) {
    throw "Output already exists and is protected from overwrite: $outputDir"
}

if ($StartInBackground) {
    $logRoot = Join-Path $RepoRoot "results/controller-v8-scientific-optimization/launch-logs"
    New-Item -ItemType Directory -Force $logRoot | Out-Null
    $stdoutPath = Join-Path $logRoot "$OptimizationId.out.log"
    $stderrPath = Join-Path $logRoot "$OptimizationId.err.log"
    $backgroundArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-OptimizationId", $OptimizationId,
        "-Backend", $Backend,
        "-Trials", $Trials,
        "-TrialsPerFold", $TrialsPerFold,
        "-InitialTrials", $InitialTrials,
        "-SensitivityTrajectories", $SensitivityTrajectories,
        "-NoninferiorityProbability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
        "-BootstrapSamples", $BootstrapSamples,
        "-Seed", $Seed,
        "-Kubeconfig", $Kubeconfig,
        "-Namespace", $Namespace,
        "-SourceRunIds"
    )
    $backgroundArguments += $SourceRunIds
    if ($SkipDependencyInstall) { $backgroundArguments += "-SkipDependencyInstall" }
    if ($SkipOuterFolds) { $backgroundArguments += "-SkipOuterFolds" }
    if ($Resume) { $backgroundArguments += "-Resume" }
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $backgroundArguments `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
        -WindowStyle Hidden -PassThru
    Write-Host "Focused RAPEC-v8 optimization started in the background." -ForegroundColor Green
    Write-Host "OptimizationId: $OptimizationId" -ForegroundColor Cyan
    Write-Host "Worker PID: $($process.Id)" -ForegroundColor Cyan
    Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v8-focused-scientific-optimization.ps1 -OptimizationId '$OptimizationId'" -ForegroundColor Cyan
    Write-Host "Logs: $stdoutPath and $stderrPath" -ForegroundColor DarkGray
    return
}

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

$arguments = @(
    "-u",
    "-m", "controller_benchmark.scientific_optimization.optimize",
    "--output-dir", $outputDir,
    "--optimization-id", $OptimizationId,
    "--base-controller-id", "rapec-v8",
    "--controller-plugin", "controller_benchmark.controllers.bayesian_guarded_rapec_v3:BayesianGuardedRapecV3Controller",
    "--controller-id-prefix", "rapec-v8-focused-nash",
    "--search-space", "controller_benchmark/scientific_optimization/search-space-rapec-v8-focused.json",
    "--backend", $Backend,
    "--trials", $Trials,
    "--trials-per-fold", $TrialsPerFold,
    "--initial-trials", $InitialTrials,
    "--sensitivity-trajectories", $SensitivityTrajectories,
    "--sensitivity-top-k", 4,
    "--noninferiority-probability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--bootstrap-samples", $BootstrapSamples,
    "--seed", $Seed
)
foreach ($sourceDir in $sourceDirs) { $arguments += @("--source-dir", $sourceDir) }
if ($SkipOuterFolds) { $arguments += "--skip-outer-folds" }
if ($Resume) { $arguments += "--resume" }

Write-Host "Focused RAPEC-v8 scientific optimization: $OptimizationId" -ForegroundColor Cyan
Write-Host "Active parameters: min_epochs_floor, utility_quantile, noise_multiplier, minimum_quality_floor" -ForegroundColor Cyan
Write-Host "Fixed parameter: trend_window=10" -ForegroundColor Cyan
Write-Host "Output: $outputDir" -ForegroundColor Cyan
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v8-focused-scientific-optimization.ps1 -OptimizationId '$OptimizationId'" -ForegroundColor Cyan

& $venvPython @arguments
if ($LASTEXITCODE -ne 0) { throw "Focused scientific RAPEC-v8 optimization failed." }

Write-Host "Focused RAPEC-v8 optimization completed: $outputDir" -ForegroundColor Green
Write-Host "Nash-selected controller: $outputDir/selected-controller.json" -ForegroundColor Green
Write-Host "All previous controllers, source bundles, baselines, and results remain unchanged." -ForegroundColor Green
Write-Host "No Rancher training job was submitted." -ForegroundColor Green
