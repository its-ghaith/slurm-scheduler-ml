[CmdletBinding()]
param(
    [ValidateRange(1, 10)][int]$EnergyPriority = 5,
    [ValidateRange(0, 10)][int]$TaskDifficulty = 0,
    [string]$ControllerId = "",
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-v3-pf.json",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$BaselineCatalog = "results/controller-baselines/controller-benchmark-v3-nine-dataset-fresh/catalog.json",
    [switch]$RefreshBaselines,
    [switch]$RequireAllStages,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$configuration = Get-Content (
    Join-Path $PSScriptRoot "..\config\rapec-v3-pf-preference-controller.json"
) -Raw | ConvertFrom-Json
$parameters = $configuration.controllers[0].controller_parameters
$parameters.energy_priority_level = $EnergyPriority
$parameters.task_difficulty_level = if ($TaskDifficulty -eq 0) {
    $null
} else {
    $TaskDifficulty
}

if ([string]::IsNullOrWhiteSpace($ControllerId)) {
    $difficultyLabel = if ($TaskDifficulty -eq 0) { "auto" } else { "$TaskDifficulty" }
    $ControllerId = "rapec-v3-pf-pref-e$EnergyPriority-d$difficultyLabel"
}

$arguments = @{
    Manifest = $Manifest
    Controller = "controller_benchmark.controllers.preference_conditioned_rapec_v3_pf:PreferenceConditionedRapecV3PfController"
    ControllerId = $ControllerId
    ControllerParametersJson = ($parameters | ConvertTo-Json -Depth 10 -Compress)
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    BaselineCatalog = $BaselineCatalog
}
if ($RefreshBaselines) { $arguments.RefreshBaselines = $true }
if ($RequireAllStages) { $arguments.RequireAllStages = $true }
if ($PlanOnly) { $arguments.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-controller-benchmark-async.ps1") @arguments
