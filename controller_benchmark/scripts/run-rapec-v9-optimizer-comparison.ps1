[CmdletBinding()]
param(
    [string]$SourceRunIds = "20260729T170915Z-rapec-v3-v8-nine-dataset",
    [string]$BenchmarkId = "",
    [string]$Backends = "sobol,qnehvi,rf-parego",
    [string]$OptimizerSeeds = "0,1,2",
    [ValidateRange(2, 10000)][int]$Trials = 100,
    [ValidateRange(2, 10000)][int]$TrialsPerFold = 100,
    [ValidateRange(1, 1000)][int]$InitialTrials = 16,
    [ValidateRange(1, 64)][int]$ReplayWorkers = 6,
    [ValidateRange(0.5, 0.999)][double]$NoninferiorityProbability = 0.95,
    [ValidateRange(16, 100000)][int]$BootstrapSamples = 1024,
    [int]$Seed = 20260804,
    [ValidateRange(128, 100000)][int]$RfCandidatePool = 2048,
    [ValidateRange(32, 5000)][int]$RfTrees = 200,
    [ValidateRange(1, 100)][int]$RfMinSamplesLeaf = 2,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$PlanOnly,
    [switch]$StartInBackground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not $BenchmarkId) {
    $BenchmarkId = "rapec-v9-optimizer-comparison-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if ($BenchmarkId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
    throw "BenchmarkId must contain 1-96 letters, digits, dots, underscores, or hyphens."
}
$sourceRunIdList = @($SourceRunIds.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$backendList = @($Backends.Split(',') | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
$optimizerSeedList = @()
foreach ($value in $OptimizerSeeds.Split(',')) {
    $parsed = 0
    if (-not [int]::TryParse($value.Trim(), [ref]$parsed)) {
        throw "Invalid optimizer seed: $value"
    }
    $optimizerSeedList += $parsed
}
$allowedBackends = @("sobol", "qnehvi", "rf-parego")
$invalidBackends = @($backendList | Where-Object { $_ -notin $allowedBackends })
if ($invalidBackends.Count) { throw "Unsupported optimizer backends: $($invalidBackends -join ', ')" }
if (-not $sourceRunIdList.Count) { throw "At least one SourceRunId is required." }
if (-not $backendList.Count) { throw "At least one optimizer backend is required." }
if (-not $optimizerSeedList.Count) { throw "At least one optimizer seed is required." }

$runsRoot = Join-Path $RepoRoot "results/controller-optimizer-comparison/runs"
$outputDir = Join-Path $runsRoot $BenchmarkId
$cacheDir = Join-Path $RepoRoot "results/controller-optimizer-comparison/replay-cache"
if (Test-Path $outputDir) {
    throw "Output already exists and is protected from overwrite: $outputDir"
}

if ($StartInBackground) {
    $logRoot = Join-Path $RepoRoot "results/controller-optimizer-comparison/launch-logs"
    New-Item -ItemType Directory -Force $logRoot | Out-Null
    $stdoutPath = Join-Path $logRoot "$BenchmarkId.out.log"
    $stderrPath = Join-Path $logRoot "$BenchmarkId.err.log"
    $backgroundArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
        "-BenchmarkId", $BenchmarkId,
        "-Trials", $Trials,
        "-TrialsPerFold", $TrialsPerFold,
        "-InitialTrials", $InitialTrials,
        "-ReplayWorkers", $ReplayWorkers,
        "-NoninferiorityProbability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
        "-BootstrapSamples", $BootstrapSamples,
        "-Seed", $Seed,
        "-RfCandidatePool", $RfCandidatePool,
        "-RfTrees", $RfTrees,
        "-RfMinSamplesLeaf", $RfMinSamplesLeaf,
        "-Kubeconfig", $Kubeconfig,
        "-Namespace", $Namespace,
        "-Backends", ($backendList -join ','),
        "-OptimizerSeeds", ($optimizerSeedList -join ','),
        "-SourceRunIds", ($sourceRunIdList -join ',')
    )
    if ($SkipDependencyInstall) { $backgroundArguments += "-SkipDependencyInstall" }
    if ($PlanOnly) { $backgroundArguments += "-PlanOnly" }
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $backgroundArguments `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
        -WindowStyle Hidden -PassThru
    Write-Host "RAPEC-v9 optimizer comparison started in the background." -ForegroundColor Green
    Write-Host "BenchmarkId: $BenchmarkId" -ForegroundColor Cyan
    Write-Host "Worker PID: $($process.Id)" -ForegroundColor Cyan
    Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v9-optimizer-comparison.ps1 -BenchmarkId '$BenchmarkId'" -ForegroundColor Cyan
    Write-Host "Logs: $stdoutPath and $stderrPath" -ForegroundColor DarkGray
    return
}

$sourceRoot = Join-Path $RepoRoot "results/controller-optimization/sources"
$sourceDirs = @()
foreach ($sourceRunId in $sourceRunIdList) {
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
if (-not $SkipDependencyInstall -and -not $PlanOnly) {
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
    "-u", "-m", "controller_benchmark.scientific_optimization.optimizer_benchmark",
    "--output-dir", $outputDir,
    "--benchmark-id", $BenchmarkId,
    "--backends", ($backendList -join ','),
    "--optimizer-seeds", (($optimizerSeedList | ForEach-Object { [string]$_ }) -join ','),
    "--trials", $Trials,
    "--trials-per-fold", $TrialsPerFold,
    "--initial-trials", $InitialTrials,
    "--replay-workers", $ReplayWorkers,
    "--replay-cache-dir", $cacheDir,
    "--noninferiority-probability", $NoninferiorityProbability.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--bootstrap-samples", $BootstrapSamples,
    "--seed", $Seed,
    "--rf-candidate-pool", $RfCandidatePool,
    "--rf-trees", $RfTrees,
    "--rf-min-samples-leaf", $RfMinSamplesLeaf
)
foreach ($sourceDir in $sourceDirs) { $arguments += @("--source-dir", $sourceDir) }
if ($PlanOnly) { $arguments += "--plan-only" }

Write-Host "RAPEC-v9 fair optimizer comparison: $BenchmarkId" -ForegroundColor Cyan
Write-Host "Backends: $($backendList -join ', '); optimizer seeds: $($optimizerSeedList -join ', ')" -ForegroundColor Cyan
Write-Host "Morris is skipped; all seven RAPEC-v9 parameters remain active." -ForegroundColor Cyan
Write-Host "No Rancher training job is submitted by this offline comparison." -ForegroundColor Cyan
Write-Host "Output: $outputDir" -ForegroundColor Cyan

& $venvPython @arguments
if ($LASTEXITCODE -ne 0) { throw "RAPEC-v9 optimizer comparison failed." }

if ($PlanOnly) {
    Write-Host "Optimizer comparison planned: $outputDir/plan.json" -ForegroundColor Green
}
else {
    Write-Host "Optimizer comparison completed: $outputDir" -ForegroundColor Green
    Write-Host "Selected controller: $outputDir/selected-controller.json" -ForegroundColor Green
}
Write-Host "All previous controllers, baselines, optimizations, and Rancher results remain unchanged." -ForegroundColor Green
