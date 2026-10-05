[CmdletBinding()]
param(
    [string]$BenchmarkId = "",
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 5,
    [switch]$Once
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$parameters = @{
    RunsRoot = Join-Path $RepoRoot "results/controller-optimizer-comparison/runs"
    ProcessModule = "controller_benchmark.scientific_optimization.optimizer_benchmark"
    IntervalSeconds = $IntervalSeconds
}
if ($BenchmarkId) { $parameters.OptimizationId = $BenchmarkId }
if ($Once) { $parameters.Once = $true }

& (Join-Path $PSScriptRoot "watch-rapec-v3-scientific-optimization.ps1") @parameters
