[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [string]$OutputDir = "",
    [int]$HorizonCheckpoints = 5,
    [int]$MinimumCheckpoints = 10,
    [double]$UsefulGain = 0.002,
    [int]$Patience = 3,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Snapshot)) { $Snapshot = Join-Path $RepoRoot $Snapshot }
if (-not (Test-Path $Snapshot)) { throw "Snapshot not found: $Snapshot" }
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "results\controller-benchmark\lcpfn-comparisons\$(Get-Date -Format 'yyyy-MM-dd_HHmmss')"
} elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}

$versionText = & $PythonExecutable -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { throw "Could not execute Python: $PythonExecutable" }
$version = [Version]$versionText.Trim()
if ($version -lt [Version]'3.9' -or $version -ge [Version]'3.12') {
    throw "LC-PFN 0.1.3 requires Python >=3.9,<3.12; found $version. Use -PythonExecutable with Python 3.9-3.11 or use compare-completed-run-with-lcpfn-rancher.ps1."
}

$environmentRoot = Join-Path $RepoRoot "results\controller-benchmark\lcpfn-environment"
$python = Join-Path $environmentRoot "Scripts\python.exe"
if (-not (Test-Path $python)) {
    & $PythonExecutable -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not create the LC-PFN environment." }
    & $python -m pip install -r "controller_benchmark/lcpfn-requirements.txt"
    if ($LASTEXITCODE -ne 0) { throw "Could not install LC-PFN 0.1.3." }
}

& $python -m controller_benchmark.compare_lcpfn `
    --snapshot $Snapshot `
    --output-dir $OutputDir `
    --horizon-checkpoints $HorizonCheckpoints `
    --minimum-checkpoints $MinimumCheckpoints `
    --useful-gain $UsefulGain `
    --patience $Patience
if ($LASTEXITCODE -ne 0) { throw "LC-PFN comparison failed." }
Write-Host "LC-PFN comparison created: $OutputDir" -ForegroundColor Green
