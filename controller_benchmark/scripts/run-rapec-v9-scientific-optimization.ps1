[CmdletBinding()]
param(
    [string[]]$SourceRunIds = @("20260729T170915Z-rapec-v3-v8-nine-dataset"),
    [string]$OptimizationId = "",
    [ValidateSet("qnehvi", "sobol", "rf-parego")][string]$Backend = "qnehvi",
    [int]$Trials = 128,
    [int]$TrialsPerFold = 72,
    [int]$InitialTrials = 32,
    [int]$SensitivityTrajectories = 16,
    [double]$NoninferiorityProbability = 0.95,
    [int]$BootstrapSamples = 1024,
    [ValidateRange(1, 64)][int]$ReplayWorkers = 6,
    [ValidateRange(128, 100000)][int]$RfCandidatePool = 2048,
    [ValidateRange(32, 5000)][int]$RfTrees = 200,
    [ValidateRange(1, 100)][int]$RfMinSamplesLeaf = 2,
    [int]$Seed = 20260803,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$RunSensitivity,
    [switch]$SkipOuterFolds,
    [switch]$Resume,
    [switch]$StartInBackground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not $OptimizationId) {
    $OptimizationId = "rapec-v9-risk-constrained-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if ($OptimizationId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
    throw "OptimizationId must contain 1-96 letters, digits, dots, underscores, or hyphens."
}
if (-not $SourceRunIds.Count) { throw "At least one SourceRunId is required." }

$runsRoot = Join-Path $RepoRoot "results/controller-v9-scientific-optimization/runs"
$outputDir = Join-Path $runsRoot $OptimizationId
if ((Test-Path $outputDir) -and -not $Resume) {
    throw "Output already exists and is protected from overwrite: $outputDir"
}

if ($StartInBackground) {
    $logRoot = Join-Path $RepoRoot "results/controller-v9-scientific-optimization/launch-logs"
    New-Item -ItemType Directory -Force $logRoot | Out-Null
    $stdoutPath = Join-Path $logRoot "$OptimizationId.out.log"
    $stderrPath = Join-Path $logRoot "$OptimizationId.err.log"
    $backgroundArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
        "-OptimizationId", $OptimizationId,
        "-Backend", $Backend,
        "-Trials", $Trials,
        "-TrialsPerFold", $TrialsPerFold,
        "-InitialTrials", $InitialTrials,
        "-SensitivityTrajectories", $SensitivityTrajectories,
        "-NoninferiorityProbability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
        "-BootstrapSamples", $BootstrapSamples,
        "-ReplayWorkers", $ReplayWorkers,
        "-RfCandidatePool", $RfCandidatePool,
        "-RfTrees", $RfTrees,
        "-RfMinSamplesLeaf", $RfMinSamplesLeaf,
        "-Seed", $Seed,
        "-Kubeconfig", $Kubeconfig,
        "-Namespace", $Namespace,
        "-SourceRunIds"
    )
    $backgroundArguments += $SourceRunIds
    if ($SkipDependencyInstall) { $backgroundArguments += "-SkipDependencyInstall" }
    if ($RunSensitivity) { $backgroundArguments += "-RunSensitivity" }
    if ($SkipOuterFolds) { $backgroundArguments += "-SkipOuterFolds" }
    if ($Resume) { $backgroundArguments += "-Resume" }
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $backgroundArguments `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
        -WindowStyle Hidden -PassThru
    Write-Host "RAPEC-v9 optimization started in the background." -ForegroundColor Green
    Write-Host "OptimizationId: $OptimizationId" -ForegroundColor Cyan
    Write-Host "Worker PID: $($process.Id)" -ForegroundColor Cyan
    Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v9-scientific-optimization.ps1 -OptimizationId '$OptimizationId'" -ForegroundColor Cyan
    Write-Host "Logs: $stdoutPath and $stderrPath" -ForegroundColor DarkGray
    return
}

$sourceRoot = Join-Path $RepoRoot "results/controller-optimization/sources"
$sourceDirs = @()
foreach ($sourceRunId in $SourceRunIds) {
    $sourceDir = Join-Path $sourceRoot $sourceRunId
    if (-not (Test-Path $sourceDir)) {
        & (Join-Path $PSScriptRoot "export-controller-optimization-source.ps1") `
            -RunId $sourceRunId -DestinationRoot $sourceRoot `
            -Kubeconfig $Kubeconfig -Namespace $Namespace
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
    & $venvPython -c "import torch, botorch, gpytorch, SALib, numpy, sklearn" 2>$null
    $dependencyStatus = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($dependencyStatus -ne 0) {
        & $venvPython -m pip install -r "controller_benchmark/scientific_optimization/requirements.txt"
        if ($LASTEXITCODE -ne 0) { throw "Could not install scientific optimization dependencies." }
    }
}

$arguments = @(
    "-u", "-m", "controller_benchmark.scientific_optimization.optimize",
    "--output-dir", $outputDir,
    "--optimization-id", $OptimizationId,
    "--base-controllers", "controller_benchmark/config/rapec-v9-controller.json",
    "--base-controller-id", "rapec-v9",
    "--controller-plugin", "controller_benchmark.controllers.risk_constrained_bayesian_multi_horizon:RiskConstrainedBayesianMultiHorizonController",
    "--controller-id-prefix", "rapec-v9-risk-constrained",
    "--search-space", "controller_benchmark/scientific_optimization/search-space-rapec-v9.json",
    "--selection-method", "energy-under-quality-constraint",
    "--backend", $Backend,
    "--trials", $Trials,
    "--trials-per-fold", $TrialsPerFold,
    "--initial-trials", $InitialTrials,
    "--sensitivity-trajectories", $SensitivityTrajectories,
    "--sensitivity-top-k", 6,
    "--noninferiority-probability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--bootstrap-samples", $BootstrapSamples,
    "--replay-workers", $ReplayWorkers,
    "--replay-cache-dir", (Join-Path $RepoRoot "results/controller-v9-scientific-optimization/replay-cache"),
    "--evaluation-controller-id", "rapec-v9-scientific-fixed-evaluation",
    "--rf-candidate-pool", $RfCandidatePool,
    "--rf-trees", $RfTrees,
    "--rf-min-samples-leaf", $RfMinSamplesLeaf,
    "--seed", $Seed
)
foreach ($sourceDir in $sourceDirs) { $arguments += @("--source-dir", $sourceDir) }
if (-not $RunSensitivity) { $arguments += "--skip-sensitivity" }
if ($SkipOuterFolds) { $arguments += "--skip-outer-folds" }
if ($Resume) { $arguments += "--resume" }

Write-Host "RAPEC-v9 constraint-first scientific optimization: $OptimizationId" -ForegroundColor Cyan
Write-Host "Quality is a hard non-inferiority constraint; energy Q25 is maximized only among feasible trials." -ForegroundColor Cyan
if (-not $RunSensitivity) {
    Write-Host "Morris is skipped because all seven RAPEC-v9 parameters remain mandatory." -ForegroundColor Cyan
}
Write-Host "Output: $outputDir" -ForegroundColor Cyan
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v9-scientific-optimization.ps1 -OptimizationId '$OptimizationId'" -ForegroundColor Cyan

& $venvPython @arguments
if ($LASTEXITCODE -ne 0) { throw "Scientific RAPEC-v9 optimization failed." }

Write-Host "RAPEC-v9 optimization completed: $outputDir" -ForegroundColor Green
Write-Host "Constraint-first selected controller: $outputDir/selected-controller.json" -ForegroundColor Green
Write-Host "Previous controllers, baselines, and results remain unchanged. No Rancher job was submitted." -ForegroundColor Green
