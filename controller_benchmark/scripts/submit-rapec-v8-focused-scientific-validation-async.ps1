[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OptimizationDir,
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    [int[]]$FreshSeeds = @(),
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$AllowInfeasible,
    [switch]$PlanOnly
)

$parameters = @{
    OptimizationDir = $OptimizationDir
    Manifest = $Manifest
    FreshSeeds = $FreshSeeds
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
}
if ($AllowInfeasible) { $parameters.AllowInfeasible = $true }
if ($PlanOnly) { $parameters.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-rapec-v3-scientific-validation-async.ps1") @parameters
