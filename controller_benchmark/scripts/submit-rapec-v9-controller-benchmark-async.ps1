[CmdletBinding()]
param(
    [string]$ControllerId = "rapec-v9",
    [string]$ControllerParametersJson = "",
    [string]$Manifest = "controller_benchmark/config/benchmark-v4-rapec-v9-nine-dataset.json",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$BaselineCatalog = "results/controller-baselines/controller-benchmark-v4-rapec-v9-risk-constrained/catalog.json",
    [switch]$RefreshBaselines,
    [switch]$RequireAllStages,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ControllerParametersJson)) {
    $ControllerParametersJson = @"
{
  "horizons": [1, 5, 10, 20],
  "posterior_samples": 512,
  "model_window": 60,
  "minimum_observations_floor": 10,
  "minimum_calibration_points": 8,
  "calibration_window": 40,
  "quality_risk_alpha": 0.05,
  "evidence_decay": 0.5,
  "evidence_stop_probability": 0.95,
  "numerical_quality_floor": 0.000001
}
"@
}

$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.risk_constrained_bayesian_multi_horizon:RiskConstrainedBayesianMultiHorizonController"
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

