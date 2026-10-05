[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$DestinationRoot = "results/controller-optimization/sources",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($DestinationRoot)) { $DestinationRoot = Join-Path $RepoRoot $DestinationRoot }
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$target = Join-Path $DestinationRoot $RunId

if (Test-Path $target) {
    & python -m controller_benchmark.optimization.source_bundle verify --source-dir $target
    if ($LASTEXITCODE -ne 0) { throw "Existing optimization source failed verification: $target" }
    Write-Host "Verified existing immutable optimization source: $target" -ForegroundColor Green
    return
}

New-Item -ItemType Directory -Force $DestinationRoot | Out-Null
$temporary = "$target.incomplete-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $temporary | Out-Null
$pod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=benchmark-orchestrator -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "benchmark-orchestrator is not running." }
$remote = "/workspace-cache/controller-benchmarks/$RunId"

try {
    Push-Location $temporary
    try {
        & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:${remote}/manifest.json" "manifest.json"
        if ($LASTEXITCODE -ne 0) { throw "Could not copy manifest.json." }
        & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:${remote}/run-matrix.json" "run-matrix.json"
        if ($LASTEXITCODE -ne 0) { throw "Could not copy run-matrix.json." }
        & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:${remote}/baseline-cache" "baseline-cache"
        if ($LASTEXITCODE -ne 0) { throw "Could not copy the Full100 baseline cache." }
    }
    finally {
        Pop-Location
    }
    & python -m controller_benchmark.optimization.source_bundle create --source-dir $temporary --run-id $RunId
    if ($LASTEXITCODE -ne 0) { throw "Could not validate and lock the optimization source." }
    Move-Item -LiteralPath $temporary -Destination $target
}
catch {
    Write-Warning "The incomplete source was retained for diagnosis: $temporary"
    throw
}

Write-Host "Immutable Full100 optimization source created: $target" -ForegroundColor Green
Write-Host "The Rancher run and all previous benchmark results were read only." -ForegroundColor Green
