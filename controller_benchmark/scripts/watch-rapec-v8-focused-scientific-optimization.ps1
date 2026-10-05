[CmdletBinding()]
param(
    [string]$OptimizationId = "",
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 5,
    [switch]$Once
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$parameters = @{
    RunsRoot = Join-Path $RepoRoot "results/controller-v8-scientific-optimization/runs"
    IntervalSeconds = $IntervalSeconds
}
if ($OptimizationId) { $parameters.OptimizationId = $OptimizationId }
if ($Once) { $parameters.Once = $true }

& (Join-Path $PSScriptRoot "watch-rapec-v3-scientific-optimization.ps1") @parameters
