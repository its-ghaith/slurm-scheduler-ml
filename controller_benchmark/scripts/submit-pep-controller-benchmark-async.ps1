[CmdletBinding()]
param(
    [string]$ControllerId = "pep-v1",
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
  "min_epochs": 30,
  "horizon_epochs": 5,
  "min_fit_points": 10,
  "patience": 3,
  "trend_window": 10,
  "bootstrap_samples": 64,
  "knn_neighbors": 7,
  "useful_gain": 0.002,
  "min_expected_gain": 0.002,
  "max_probability_gain_gt_threshold": 0.20,
  "min_quality_per_wh": 0.0002,
  "minimum_quality": 0.0,
  "target_energy_saving_fraction": 0.20,
  "post_training_reserve_wh": 0.0,
  "uncertainty_weight": 1.0
}
"@
}

$submit = Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1"
$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.probabilistic_energy_pareto:ProbabilisticEnergyParetoController"
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
