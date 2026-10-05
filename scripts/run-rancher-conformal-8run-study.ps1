[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int[]]$TrainingSeeds = @(10, 11),
    [int]$SplitSeed = 42,
    [int]$StartAtRun = 1,
    [int]$EndAtRun = 8,
    [int]$CooldownSeconds = 30,
    [string]$PlanPath = "results/conformal-controller-study/run-matrix.csv",
    [string]$SnapshotRoot = "D:\Masterarbeit_Vergleichssicherung",
    [switch]$PlanOnly,
    [switch]$SkipReset,
    [switch]$SkipBackupAndDashboard,
    [switch]$ForceWorkspaceSync
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($TrainingSeeds.Count -ne 2) {
    throw "The 8-run design requires exactly two unseen training seeds."
}
if (-not (Test-Path -LiteralPath "slurm/conformal_calibration.json")) {
    throw "Missing calibration file: slurm/conformal_calibration.json"
}

function Get-ScenarioConfig {
    param([Parameter(Mandatory = $true)][string]$Scenario)
    switch ($Scenario) {
        "train128-scratch" {
            return @{ ModelVersion = "yolov8n.yaml"; Pretrained = "false"; TrainSize = 128; MinEpochs = 40; MinQuality = 0.60 }
        }
        "train256-pretrained" {
            return @{ ModelVersion = "yolov8n.pt"; Pretrained = "true"; TrainSize = 256; MinEpochs = 35; MinQuality = 0.78 }
        }
        default { throw "Unknown scenario: $Scenario" }
    }
}

function New-RunMatrix {
    $rows = [System.Collections.Generic.List[object]]::new()
    $run = 0
    foreach ($scenario in @("train128-scratch", "train256-pretrained")) {
        foreach ($seed in $TrainingSeeds) {
            # Alternate order within a seed to reduce systematic thermal/cache bias.
            $strategies = if ((($seed + $(if ($scenario -eq "train128-scratch") { 0 } else { 1 })) % 2) -eq 0) {
                @("full100", "conformal_energy_aware_controller")
            } else {
                @("conformal_energy_aware_controller", "full100")
            }
            foreach ($strategy in $strategies) {
                $run++
                $rows.Add([pscustomobject]@{
                    run = $run
                    scenario = $scenario
                    training_seed = $seed
                    split_seed = $SplitSeed
                    strategy = $strategy
                    controller = if ($strategy -eq "full100") { "none" } else { "conformal_energy_aware" }
                    max_epochs = 100
                })
            }
        }
    }
    return @($rows)
}

$matrix = New-RunMatrix
$planDirectory = Split-Path -Parent $PlanPath
if ($planDirectory) { New-Item -ItemType Directory -Force -Path $planDirectory | Out-Null }
$matrix | Export-Csv -LiteralPath $PlanPath -NoTypeInformation -Encoding UTF8

if ($StartAtRun -lt 1 -or $StartAtRun -gt 8 -or $EndAtRun -lt $StartAtRun -or $EndAtRun -gt 8) {
    throw "Valid run range is 1..8."
}
$selected = @($matrix[($StartAtRun - 1)..($EndAtRun - 1)])
Write-Host "Conformal controller study: runs $StartAtRun-$EndAtRun of 8" -ForegroundColor Green
$selected | Format-Table run, scenario, training_seed, strategy, controller -AutoSize
if ($PlanOnly) { return }

if (-not $SkipReset -and $StartAtRun -eq 1) {
    & powershell -ExecutionPolicy Bypass -File .\scripts\reset-rancher-slurm-observability.ps1 `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -Force
    if ($LASTEXITCODE -ne 0) { throw "Observability reset failed." }
}

$first = $true
foreach ($row in $selected) {
    if (-not $first -and $CooldownSeconds -gt 0) { Start-Sleep -Seconds $CooldownSeconds }
    $scenario = Get-ScenarioConfig -Scenario $row.scenario
    $arguments = @(
        "-ExecutionPolicy", "Bypass", "-File", ".\scripts\submit-rancher-job.ps1",
        "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace,
        "-ExperimentName", "carpk-conformal-$($row.scenario)",
        "-ScenarioName", $row.scenario, "-Dataset", "carpk", "-Model", "yolov8",
        "-ModelVersion", $scenario.ModelVersion, "-Epochs", "100", "-BatchSize", "8",
        "-ImageSize", "640", "-Patience", "100", "-TrainSize", "$($scenario.TrainSize)",
        "-Pretrained", $scenario.Pretrained, "-TrainingSeed", "$($row.training_seed)",
        "-SplitSeed", "$SplitSeed", "-CachePolicy", "ram",
        "-ControllerMode", $row.controller, "-ComparisonStrategy", $row.strategy,
        "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "$($scenario.MinEpochs)",
        "-AdaptivePatience", "3", "-UncertaintyTargetEpoch", "100",
        "-UncertaintyAlpha", "0.05", "-UncertaintyBootstrapSamples", "16",
        "-UncertaintyMinFitPoints", "12", "-UncertaintyMinQuality", "$($scenario.MinQuality)",
        "-ConformalCalibrationPath", "/workspace/slurm/conformal_calibration.json",
        "-ConformalRegretTolerance", "0.015", "-ConformalFutureEfficiencyThreshold", "0.05",
        "-ConformalMaxIntervalWidth", "0.08", "-ConformalEnergyWindow", "5",
        "-ConformalRequireCalibration", "true", "-IdleSampleSeconds", "10",
        "-SkipDependencyInstall", "-WaitForCompletion"
    )
    if ($first -or $ForceWorkspaceSync) { $arguments += "-ForceWorkspaceSync" }
    Write-Host "Run $($row.run)/8: $($row.scenario), seed $($row.training_seed), $($row.strategy)" -ForegroundColor Cyan
    & powershell @arguments
    if ($LASTEXITCODE -ne 0) { throw "Run $($row.run) failed." }
    $first = $false
}

if ($EndAtRun -lt 8 -or $SkipBackupAndDashboard) {
    Write-Host "Selected runs completed; backup/dashboard deployment was skipped." -ForegroundColor Yellow
    return
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$snapshot = Join-Path $SnapshotRoot "CARPK_ConformalController_8Jobs_$stamp"
& powershell -ExecutionPolicy Bypass -File .\scripts\backup-rancher-comparison-v2.ps1 `
    -Kubeconfig $Kubeconfig -Namespace $Namespace -OutputPath $snapshot `
    -SnapshotName "carpk-conformal-controller-8run" `
    -DashboardSource "grafana/dashboards/slurm-energy-overview.json"
if ($LASTEXITCODE -ne 0) { throw "8-run snapshot failed." }

& powershell -ExecutionPolicy Bypass -File .\scripts\deploy-rancher-comparison-dashboards.ps1 `
    -Kubeconfig $Kubeconfig -Namespace $Namespace -EightRunSnapshot $snapshot
if ($LASTEXITCODE -ne 0) { throw "Comparison dashboard deployment failed." }

Write-Host "8-run study completed and preserved: $snapshot" -ForegroundColor Green
