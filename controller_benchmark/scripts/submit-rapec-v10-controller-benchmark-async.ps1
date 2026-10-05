[CmdletBinding()]
param(
    [string]$ControllerId = "rapec-v10",
    [string]$ControllerParametersJson = "",
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-v10.json",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$BaselineCatalog = "results/controller-baselines/controller-benchmark-v3-nine-dataset-fresh/catalog.json",
    [switch]$RefreshBaselines,
    [switch]$RequireAllStages,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ControllerParametersJson)) {
    $configuration = Get-Content (Join-Path $PSScriptRoot "..\config\rapec-v10-controller.json") -Raw | ConvertFrom-Json
    $ControllerParametersJson = $configuration.controllers[0].controller_parameters | ConvertTo-Json -Depth 10 -Compress
}

$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.rapec_v10:RAPECV10Controller"
    ControllerId = $ControllerId
    ControllerParametersJson = $ControllerParametersJson
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    BaselineCatalog = $BaselineCatalog
}
if ($RefreshBaselines) { $arguments.RefreshBaselines = $true }
if ($RequireAllStages) { $arguments.RequireAllStages = $true }
if ($PlanOnly) { $arguments.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1") @arguments
