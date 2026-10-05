[CmdletBinding()]
param(
    [string]$ControllerId = "rapec-v4",
    [string]$ControllerParametersJson = "",
    [string]$Manifest = "controller_benchmark/config/benchmark-v2-shared-seed.json",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$BaselineCatalog = "D:\Masterarbeit_Vergleichssicherung\ControllerBaselines\controller-benchmark-v2-shared-seed\catalog.json",
    [switch]$RefreshBaselines,
    [switch]$RequireAllStages,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ControllerParametersJson)) {
    $ControllerParametersJson = @"
{
  "adapter_strength": 1.0,
  "min_fit_points": 10,
  "bootstrap_samples": 96,
  "knn_neighbors": 9,
  "min_epochs_floor": 12,
  "late_stage_relaxation": 0.55,
  "min_meaningful_gain": 0.0001
}
"@
}

$submit = Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1"
$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.task_aware_dynamic_threshold:TaskAwareDynamicThresholdController"
    ControllerId = $ControllerId
    ControllerParametersJson = $ControllerParametersJson
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    BaselineCatalog = $BaselineCatalog
}
if ($RefreshBaselines) { $arguments.RefreshBaselines = $true }
if ($RequireAllStages) { $arguments.RequireAllStages = $true }
if ($PlanOnly) { $arguments.PlanOnly = $true }

& $submit @arguments
