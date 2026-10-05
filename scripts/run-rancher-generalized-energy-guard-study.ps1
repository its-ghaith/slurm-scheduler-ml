[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [ValidateSet("A", "B", "Both")]
    [string]$Stage = "Both",
    [int]$CooldownSeconds = 5,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}
$Kubeconfig = [System.IO.Path]::GetFullPath($Kubeconfig)
$submitScript = Join-Path $RepoRoot "scripts\submit-rancher-job.ps1"

$runs = @()
if ($Stage -in @("A", "Both")) {
    foreach ($seed in 0..4) {
        $runs += [pscustomobject]@{
            Stage = "A"; Scenario = "train128-scratch"; TrainingSeed = $seed
            TrainSize = 128; ModelVersion = "yolov8n.yaml"; Pretrained = "false"
        }
    }
}
if ($Stage -in @("B", "Both")) {
    $runs += @(
        [pscustomobject]@{ Stage = "B"; Scenario = "train128-scratch-yolov8n"; TrainingSeed = 0; TrainSize = 128; ModelVersion = "yolov8n.yaml"; Pretrained = "false" },
        [pscustomobject]@{ Stage = "B"; Scenario = "train256-scratch-yolov8n"; TrainingSeed = 0; TrainSize = 256; ModelVersion = "yolov8n.yaml"; Pretrained = "false" },
        [pscustomobject]@{ Stage = "B"; Scenario = "trainfull-scratch-yolov8n"; TrainingSeed = 0; TrainSize = 0; ModelVersion = "yolov8n.yaml"; Pretrained = "false" },
        [pscustomobject]@{ Stage = "B"; Scenario = "train128-pretrained-yolov8n"; TrainingSeed = 0; TrainSize = 128; ModelVersion = "yolov8n.pt"; Pretrained = "true" },
        [pscustomobject]@{ Stage = "B"; Scenario = "train256-pretrained-yolov8n"; TrainingSeed = 0; TrainSize = 256; ModelVersion = "yolov8n.pt"; Pretrained = "true" },
        [pscustomobject]@{ Stage = "B"; Scenario = "trainfull-pretrained-yolov8n"; TrainingSeed = 0; TrainSize = 0; ModelVersion = "yolov8n.pt"; Pretrained = "true" },
        [pscustomobject]@{ Stage = "B"; Scenario = "train256-pretrained-yolov8s"; TrainingSeed = 0; TrainSize = 256; ModelVersion = "yolov8s.pt"; Pretrained = "true" }
    )
}

Write-Host "Generalized Energy Guard study: $($runs.Count) controller runs" -ForegroundColor Green
$runs | Format-Table Stage, Scenario, TrainingSeed, TrainSize, ModelVersion, Pretrained -AutoSize
if ($PlanOnly) { return }

for ($index = 0; $index -lt $runs.Count; $index++) {
    $run = $runs[$index]
    if ($index -gt 0 -and $CooldownSeconds -gt 0) { Start-Sleep -Seconds $CooldownSeconds }
    $experiment = if ($run.Stage -eq "A") {
        "carpk-stage-a-generalized-energy-guard"
    } else {
        "carpk-stage-b-generalized-energy-guard"
    }
    $submit = @(
        "-ExecutionPolicy", "Bypass", "-File", $submitScript,
        "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace,
        "-ExperimentName", $experiment,
        "-ScenarioName", $run.Scenario, "-Dataset", "carpk", "-Model", "yolov8",
        "-ModelVersion", $run.ModelVersion, "-Epochs", "100", "-BatchSize", "8", "-ImageSize", "640",
        "-Patience", "100", "-TrainSize", "$($run.TrainSize)", "-Pretrained", $run.Pretrained,
        "-TrainingSeed", "$($run.TrainingSeed)", "-SplitSeed", "0", "-CachePolicy", "ram",
        "-TimeoutSeconds", "7200", "-ControllerMode", "generalized_energy_guard",
        "-ComparisonStrategy", "generalized_energy_guard_20pct",
        "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "30",
        "-UncertaintyTargetEpoch", "100", "-UncertaintyAlpha", "0.05",
        "-UncertaintyBootstrapSamples", "24", "-UncertaintyMinFitPoints", "12",
        "-UncertaintyMinQuality", "0.55", "-ConformalRegretTolerance", "0.015",
        "-ControllerEvaluationInterval", "1",
        "-EnergyGuardTargetSavingFraction", "0.20",
        "-EnergyGuardSafetyMarginFraction", "0.01",
        "-EnergyGuardPostTrainingReserveWh", "0.90",
        "-EnergyGuardWindow", "5",
        "-EnergyGuardFallbackEpochFraction", "0.77",
        "-EnergyGuardStrict", "true",
        "-IdleSampleSeconds", "10", "-SkipDependencyInstall", "-WaitForCompletion", "-ValidateMlflowViaSlurmd"
    )
    if ($index -eq 0) { $submit += "-ForceWorkspaceSync" }
    Write-Host "Run $($index + 1)/$($runs.Count): Stage $($run.Stage), $($run.Scenario), seed=$($run.TrainingSeed)" -ForegroundColor Cyan
    & powershell @submit
    if ($LASTEXITCODE -ne 0) { throw "Controller run failed: $($run.Scenario), seed=$($run.TrainingSeed)" }
}

Write-Host "All generalized energy guard jobs completed." -ForegroundColor Green
