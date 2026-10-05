[CmdletBinding()]
param(
    [string[]]$SourceRunIds = @("20260729T170915Z-rapec-v3-v8-nine-dataset"),
    [string]$AnalysisId = "",
    [ValidateSet("surrogate", "direct")][string]$Mode = "surrogate",
    [ValidateRange(32, 65536)][int]$DesignSamples = 1024,
    [ValidateRange(16, 65536)][int]$SobolBaseSamples = 1024,
    [ValidateRange(64, 2000)][int]$Trees = 400,
    [ValidateRange(32, 10000)][int]$BootstrapResamples = 256,
    [ValidateRange(1, 23)][int]$TopParameters = 8,
    [ValidateRange(0, 32)][int]$TopInteractions = 5,
    [ValidateRange(3, 30)][int]$AleBins = 10,
    [int]$Seed = 20260802,
    [string]$SearchSpace = "controller_benchmark/v8_sensitivity/search-space.json",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$SkipDependencyInstall,
    [switch]$Resume,
    [switch]$StartInBackground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

function Test-PowerOfTwo([int]$Value) {
    return $Value -gt 0 -and (($Value -band ($Value - 1)) -eq 0)
}

if (-not (Test-PowerOfTwo $DesignSamples)) {
    throw "DesignSamples must be a power of two."
}
if (-not (Test-PowerOfTwo $SobolBaseSamples)) {
    throw "SobolBaseSamples must be a power of two."
}
if (-not $AnalysisId) {
    $AnalysisId = "rapec-v8-sobol-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if ($AnalysisId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$') {
    throw "AnalysisId must contain 1-96 letters, digits, dots, underscores, or hyphens."
}
if (-not $SourceRunIds.Count) {
    throw "At least one SourceRunId is required."
}

$sourceRoot = Join-Path $RepoRoot "results/controller-optimization/sources"
$sourceDirs = @()
foreach ($sourceRunId in $SourceRunIds) {
    $sourceDir = Join-Path $sourceRoot $sourceRunId
    if (-not (Test-Path -LiteralPath $sourceDir)) {
        & (Join-Path $PSScriptRoot "export-controller-optimization-source.ps1") `
            -RunId $sourceRunId `
            -DestinationRoot $sourceRoot `
            -Kubeconfig $Kubeconfig `
            -Namespace $Namespace
        if ($LASTEXITCODE -ne 0) {
            throw "Sensitivity source export failed: $sourceRunId"
        }
    }
    & python -m controller_benchmark.optimization.source_bundle verify --source-dir $sourceDir
    if ($LASTEXITCODE -ne 0) {
        throw "Immutable source verification failed: $sourceRunId"
    }
    $sourceDirs += $sourceDir
}

$environmentRoot = Join-Path $RepoRoot "results/controller-v8-sensitivity/.venv"
$environmentPython = Join-Path $environmentRoot "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & python -m venv --system-site-packages $environmentRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the V8 sensitivity environment."
    }
}
if (-not $SkipDependencyInstall) {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $environmentPython -c "import numpy, scipy, SALib, sklearn, matplotlib" 2>$null
    $dependencyStatus = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($dependencyStatus -ne 0) {
        & $environmentPython -m pip install -r "controller_benchmark/v8_sensitivity/requirements.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install V8 sensitivity dependencies."
        }
    }
}

$outputDir = Join-Path $RepoRoot "results/controller-v8-sensitivity/runs/$AnalysisId"
if ((Test-Path -LiteralPath $outputDir) -and -not $Resume) {
    throw "Output already exists and is protected from overwrite: $outputDir"
}
$arguments = @(
    "-u",
    "-m", "controller_benchmark.v8_sensitivity.analysis",
    "--output-dir", $outputDir,
    "--analysis-id", $AnalysisId,
    "--mode", $Mode,
    "--design-samples", $DesignSamples,
    "--sobol-base-samples", $SobolBaseSamples,
    "--trees", $Trees,
    "--bootstrap-resamples", $BootstrapResamples,
    "--top-parameters", $TopParameters,
    "--top-interactions", $TopInteractions,
    "--ale-bins", $AleBins,
    "--search-space", $SearchSpace,
    "--seed", $Seed
)
foreach ($sourceDir in $sourceDirs) {
    $arguments += @("--source-dir", $sourceDir)
}
if ($Resume) {
    $arguments += "--resume"
}

Write-Host "RAPEC-v8 sensitivity analysis: $AnalysisId" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Cyan
Write-Host "Search space: $SearchSpace" -ForegroundColor Cyan
Write-Host "Output: $outputDir" -ForegroundColor Cyan
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v8-sobol-sensitivity.ps1 -AnalysisId '$AnalysisId'" -ForegroundColor Cyan
Write-Host "No Rancher training job is submitted. Immutable Full100 curves are replayed locally." -ForegroundColor DarkGray

if ($StartInBackground) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $stdoutPath = Join-Path $outputDir "worker.out.log"
    $stderrPath = Join-Path $outputDir "worker.err.log"
    $backgroundArguments = @($arguments)
    if (-not $Resume) {
        # The directory is created above for redirected logs; resume mode allows
        # the new worker to use that otherwise empty directory safely.
        $backgroundArguments += "--resume"
    }
    $process = Start-Process `
        -FilePath $environmentPython `
        -ArgumentList $backgroundArguments `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    Write-Host "Sensitivity worker started in background with PID $($process.Id)." -ForegroundColor Green
    Write-Host "The computer must remain powered on because this analysis runs locally." -ForegroundColor Yellow
    exit 0
}

& $environmentPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "RAPEC-v8 Sobol sensitivity analysis failed."
}

Write-Host "Sensitivity analysis completed: $outputDir" -ForegroundColor Green
Write-Host "Readable summary: $outputDir/SUMMARY.md" -ForegroundColor Green
Write-Host "All previous controllers and results remain unchanged." -ForegroundColor Green
