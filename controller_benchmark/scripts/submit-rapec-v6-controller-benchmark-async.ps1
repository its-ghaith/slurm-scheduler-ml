[CmdletBinding()]
param(
    [string]$ControllerId = "rapec-v6",
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
  "horizons": [1, 3, 5, 10, 20],
  "min_fit_points": 12,
  "trend_window": 20,
  "bootstrap_samples": 128,
  "min_epochs_floor": 15,
  "min_patience": 2,
  "max_patience": 6,
  "base_risk_alpha": 0.10,
  "epsilon_noise_multiplier": 1.25,
  "epsilon_remaining_gain_fraction": 0.10,
  "epsilon_floor": 0.00001,
  "epsilon_quality_fraction_min": 0.005,
  "epsilon_quality_fraction_max": 0.020,
  "utility_quantile": 0.25,
  "minimum_feature_coverage": 0.75,
  "calibration_quantile": 0.90
}
"@
}

$submit = Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1"
$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.online_multi_horizon_pareto:OnlineMultiHorizonParetoController"
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
