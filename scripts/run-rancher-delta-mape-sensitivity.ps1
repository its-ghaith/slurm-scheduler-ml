[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int[]]$TrainingSeeds = @(0, 1, 2),
    [int]$SplitSeed = 42,
    [string]$ExperimentName = "carpk-delta-mape-sensitivity",
    [string]$PlanPath = "results/delta-mape-sensitivity/run_matrix.csv",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$profiles = @(
    [pscustomobject]@{ profile = "aggressive"; min_epochs = 15; patience = 2; window = 3; min_delta = 0.0010; min_mape = 0.00010 },
    [pscustomobject]@{ profile = "balanced"; min_epochs = 20; patience = 3; window = 5; min_delta = 0.0005; min_mape = 0.00005 },
    [pscustomobject]@{ profile = "conservative"; min_epochs = 30; patience = 5; window = 5; min_delta = 0.0002; min_mape = 0.00002 }
)

$matrix = foreach ($seed in $TrainingSeeds) {
    foreach ($profile in $profiles) {
        [pscustomobject]@{
            scenario = "train128-scratch"
            training_seed = $seed
            split_seed = $SplitSeed
            profile = $profile.profile
            min_epochs = $profile.min_epochs
            patience = $profile.patience
            smoothing_window = $profile.window
            min_delta = $profile.min_delta
            min_mape = $profile.min_mape
        }
    }
}
$parent = Split-Path -Parent $PlanPath
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$matrix | Export-Csv $PlanPath -NoTypeInformation -Encoding UTF8

if ($PlanOnly) {
    $matrix | Format-Table -AutoSize
    Write-Host "PlanOnly: no SLURM job was submitted." -ForegroundColor Yellow
    return
}

foreach ($row in $matrix) {
    $arguments = @(
        "-ExecutionPolicy", "Bypass", "-File", ".\scripts\submit-rancher-job.ps1",
        "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace,
        "-ExperimentName", $ExperimentName, "-ScenarioName", $row.scenario,
        "-Model", "yolov8", "-ModelVersion", "yolov8n.yaml", "-Pretrained", "false",
        "-Epochs", "100", "-BatchSize", "8", "-ImageSize", "640", "-Patience", "100", "-TrainSize", "128",
        "-TrainingSeed", "$($row.training_seed)", "-SplitSeed", "$($row.split_seed)", "-CachePolicy", "ram",
        "-ControllerMode", "delta_mape", "-ComparisonStrategy", "delta_mape_$($row.profile)",
        "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "$($row.min_epochs)",
        "-AdaptivePatience", "$($row.patience)", "-AdaptiveSmoothingWindow", "$($row.smoothing_window)",
        "-AdaptiveMinDeltaMap50", "$($row.min_delta)", "-AdaptiveMinMapeMap50PerWh", "$($row.min_mape)",
        "-SkipDependencyInstall", "-WaitForCompletion"
    )
    Write-Host "Delta-MAPE sensitivity: seed=$($row.training_seed), profile=$($row.profile)" -ForegroundColor Cyan
    & powershell @arguments
    if ($LASTEXITCODE -ne 0) { throw "Sensitivity run failed for seed $($row.training_seed), profile $($row.profile)." }
}
