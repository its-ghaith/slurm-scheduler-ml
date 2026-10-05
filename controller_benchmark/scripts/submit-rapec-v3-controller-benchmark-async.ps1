[CmdletBinding()]
param(
    [string]$ControllerId = "rapec-v3",
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
  "horizon_epochs": 5,
  "min_fit_points": 10,
  "patience": 3,
  "trend_window": 10,
  "bootstrap_samples": 96,
  "knn_neighbors": 9,
  "min_epochs_floor": 12,
  "max_probability_gain_gt_threshold": 0.25,
  "max_uncertainty_to_gain_ratio": 4.0,
  "minimum_quality_floor": 0.0,
  "noise_multiplier": 1.25,
  "recent_gain_fraction": 0.35,
  "remaining_room_fraction": 0.015,
  "late_stage_relaxation": 0.55,
  "min_meaningful_gain": 0.0001,
  "utility_quantile": 0.25
}
"@
}

$submit = Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1"
$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.risk_aware_predictive_energy:RiskAwarePredictiveEnergyController"
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
