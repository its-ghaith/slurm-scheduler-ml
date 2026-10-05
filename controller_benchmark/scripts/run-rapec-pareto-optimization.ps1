[CmdletBinding()]
param(
    [string]$SourceRunId = "20260729T170915Z-rapec-v3-v8-nine-dataset",
    [string]$OptimizationId = "",
    [int]$TrialsPerStudy = 500,
    [int]$Jobs = 1,
    [int]$Seed = 20260731,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not $OptimizationId) {
    $OptimizationId = "po-rapec-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if ($OptimizationId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$') {
    throw "OptimizationId must contain 1-64 letters, digits, dots, underscores, or hyphens."
}
if ($TrialsPerStudy -lt 1 -or $Jobs -lt 1) { throw "TrialsPerStudy and Jobs must be positive." }

$sourceRoot = Join-Path $RepoRoot "results/controller-optimization/sources"
$sourceDir = Join-Path $sourceRoot $SourceRunId
if (-not (Test-Path $sourceDir)) {
    & (Join-Path $PSScriptRoot "export-controller-optimization-source.ps1") `
        -RunId $SourceRunId `
        -DestinationRoot $sourceRoot `
        -Kubeconfig $Kubeconfig `
        -Namespace $Namespace
    if ($LASTEXITCODE -ne 0) { throw "Optimization source export failed." }
}
else {
    & python -m controller_benchmark.optimization.source_bundle verify --source-dir $sourceDir
    if ($LASTEXITCODE -ne 0) { throw "Optimization source verification failed." }
}

$venvRoot = Join-Path $RepoRoot "results/controller-optimization/.venv"
$venvPython = Join-Path $venvRoot "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    & python -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not create the optimization virtual environment." }
}
if (-not $SkipDependencyInstall) {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $venvPython -c "import optuna" 2>$null
    $optunaImportExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($optunaImportExitCode -ne 0) {
        & $venvPython -m pip install -r "controller_benchmark/optimization/requirements.txt"
        if ($LASTEXITCODE -ne 0) { throw "Could not install optimization dependencies." }
    }
}

$outputDir = Join-Path $RepoRoot "results/controller-optimization/runs/$OptimizationId"
$arguments = @(
    "-m", "controller_benchmark.optimization.optimize",
    "--source-dir", $sourceDir,
    "--output-dir", $outputDir,
    "--optimization-id", $OptimizationId,
    "--trials-per-study", $TrialsPerStudy,
    "--jobs", $Jobs,
    "--seed", $Seed
)
if ($Resume) { $arguments += "--resume" }
& $venvPython @arguments
if ($LASTEXITCODE -ne 0) { throw "RAPEC Pareto optimization failed." }

Write-Host "Pareto optimization completed: $outputDir" -ForegroundColor Green
Write-Host "Selected profiles: $outputDir/selected-controllers.json" -ForegroundColor Green
Write-Host "No Rancher training job was submitted and no previous result was modified." -ForegroundColor Green
