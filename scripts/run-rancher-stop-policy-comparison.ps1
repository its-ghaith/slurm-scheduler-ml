[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [ValidateSet("train128-scratch", "train256-pretrained")]
    [string[]]$Scenarios = @("train128-scratch", "train256-pretrained"),
    [int[]]$TrainingSeeds = @(0, 1, 2, 3, 4),
    [int]$SplitSeed = 42,
    [string]$ExperimentPrefix = "carpk-stop-policy-v2",
    [int]$MaxEpochs = 100,
    [int]$StandardEarlyStoppingPatience = 20,
    [double]$StandardEarlyStoppingMinDelta = 0.0005,
    [ValidateSet("none", "ram", "disk")]
    [string]$CachePolicy = "ram",
    [int]$IdleSampleSeconds = 10,
    [int]$CooldownSeconds = 30,
    [string]$PlanPath = "results/stop-policy-study/run_matrix.csv",
    [switch]$PlanOnly,
    [switch]$ForceWorkspaceSync,
    [switch]$ForceDatasetSync
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path = (Join-Path (Split-Path -Parent $PSScriptRoot) ".env"))
    if (-not (Test-Path $Path)) { return }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $separator = $line.IndexOf("=")
        if ($separator -lt 1) { return }
        $name = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim().Trim("'`"")
        Set-Item -Path "Env:$name" -Value $value
    }
}

Import-DotEnv
if (-not $PSBoundParameters.ContainsKey("Scenarios") -and $env:STUDY_SCENARIOS) {
    $Scenarios = @($env:STUDY_SCENARIOS -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
if (-not $PSBoundParameters.ContainsKey("TrainingSeeds") -and $env:TRAINING_SEEDS) {
    $TrainingSeeds = @($env:TRAINING_SEEDS -split "," | ForEach-Object { [int]$_.Trim() })
}
if (-not $PSBoundParameters.ContainsKey("SplitSeed") -and $env:SPLIT_SEED) { $SplitSeed = [int]$env:SPLIT_SEED }
if (-not $PSBoundParameters.ContainsKey("CachePolicy") -and $env:CACHE_POLICY) { $CachePolicy = $env:CACHE_POLICY }
if (-not $PSBoundParameters.ContainsKey("IdleSampleSeconds") -and $env:GPU_IDLE_SAMPLE_SECONDS) {
    $IdleSampleSeconds = [int]$env:GPU_IDLE_SAMPLE_SECONDS
}

function Get-ScenarioConfig {
    param([Parameter(Mandatory = $true)][string]$Name)

    switch ($Name) {
        "train128-scratch" {
            return @{
                Model = "yolov8"
                ModelVersion = "yolov8n.yaml"
                Pretrained = "false"
                TrainSize = 128
                BatchSize = 8
                ImageSize = 640
                DeltaMinEpochs = 20
                DeltaPatience = 3
                DeltaWindow = 5
                DeltaMinDelta = 0.0005
                DeltaMinMape = 0.00005
                UncertaintyWarmup = 35
                UncertaintyPatience = 5
                UncertaintyEpsilon = 0.005
                UncertaintyAlpha = 0.05
                UncertaintyBootstrapSamples = 128
                UncertaintyMinQuality = 0.55
            }
        }
        "train256-pretrained" {
            return @{
                Model = "yolov8"
                ModelVersion = "yolov8n.pt"
                Pretrained = "true"
                TrainSize = 256
                BatchSize = 8
                ImageSize = 640
                DeltaMinEpochs = 20
                DeltaPatience = 3
                DeltaWindow = 5
                DeltaMinDelta = 0.0003
                DeltaMinMape = 0.00003
                UncertaintyWarmup = 30
                UncertaintyPatience = 5
                UncertaintyEpsilon = 0.003
                UncertaintyAlpha = 0.05
                UncertaintyBootstrapSamples = 128
                UncertaintyMinQuality = 0.70
            }
        }
        default { throw "Unknown scenario: $Name" }
    }
}

function Get-Strategies {
    param([Parameter(Mandatory = $true)][hashtable]$ScenarioConfig)

    return @(
        [pscustomobject]@{ Name = "full100"; Epochs = $MaxEpochs; Patience = $MaxEpochs; Controller = "none"; MinEpochs = $MaxEpochs; ControllerPatience = $MaxEpochs }
        [pscustomobject]@{ Name = "standard_early_stopping"; Epochs = $MaxEpochs; Patience = $MaxEpochs; Controller = "metric_early_stopping"; MinEpochs = 1; ControllerPatience = $StandardEarlyStoppingPatience }
        [pscustomobject]@{ Name = "delta_mape_controller"; Epochs = $MaxEpochs; Patience = $MaxEpochs; Controller = "delta_mape"; MinEpochs = $ScenarioConfig.DeltaMinEpochs; ControllerPatience = $ScenarioConfig.DeltaPatience }
        [pscustomobject]@{ Name = "uncertainty_aware_controller"; Epochs = $MaxEpochs; Patience = $MaxEpochs; Controller = "uncertainty_aware"; MinEpochs = $ScenarioConfig.UncertaintyWarmup; ControllerPatience = $ScenarioConfig.UncertaintyPatience }
        [pscustomobject]@{ Name = "fixed_epoch_30"; Epochs = 30; Patience = 30; Controller = "none"; MinEpochs = 30; ControllerPatience = 30 }
        [pscustomobject]@{ Name = "fixed_epoch_50"; Epochs = 50; Patience = 50; Controller = "none"; MinEpochs = 50; ControllerPatience = 50 }
    )
}

function New-StudyPlan {
    $rows = [System.Collections.Generic.List[object]]::new()
    foreach ($scenario in $Scenarios) {
        $config = Get-ScenarioConfig -Name $scenario
        foreach ($seed in $TrainingSeeds) {
            # A stable shuffle avoids systematic cache, temperature, or cluster-load bias.
            $scenarioOffset = if ($scenario -eq "train128-scratch") { 128 } else { 256 }
            $random = [System.Random]::new(($SplitSeed * 1009) + ($seed * 97) + $scenarioOffset)
            $ordered = Get-Strategies -ScenarioConfig $config | Sort-Object { $random.Next() }
            $order = 0
            foreach ($strategy in $ordered) {
                $order++
                $rows.Add([pscustomobject]@{
                    scenario = $scenario
                    training_seed = $seed
                    split_seed = $SplitSeed
                    run_order = $order
                    strategy = $strategy.Name
                    controller = $strategy.Controller
                    epochs = $strategy.Epochs
                    patience = $strategy.Patience
                    controller_min_epochs = $strategy.MinEpochs
                    controller_patience = $strategy.ControllerPatience
                    cache_policy = $CachePolicy
                    experiment = "$ExperimentPrefix-$scenario"
                })
            }
        }
    }
    return $rows
}

function Invoke-StudyRun {
    param(
        [Parameter(Mandatory = $true)]$Row,
        [Parameter(Mandatory = $true)][hashtable]$ScenarioConfig,
        [switch]$FirstRun
    )

    $arguments = @(
        "-ExecutionPolicy", "Bypass",
        "-File", ".\scripts\submit-rancher-job.ps1",
        "-Kubeconfig", $Kubeconfig,
        "-Namespace", $Namespace,
        "-ExperimentName", $Row.experiment,
        "-ScenarioName", $Row.scenario,
        "-Dataset", "carpk",
        "-Model", $ScenarioConfig.Model,
        "-ModelVersion", $ScenarioConfig.ModelVersion,
        "-Epochs", "$($Row.epochs)",
        "-BatchSize", "$($ScenarioConfig.BatchSize)",
        "-ImageSize", "$($ScenarioConfig.ImageSize)",
        "-Patience", "$($Row.patience)",
        "-TrainSize", "$($ScenarioConfig.TrainSize)",
        "-Pretrained", $ScenarioConfig.Pretrained,
        "-TrainingSeed", "$($Row.training_seed)",
        "-SplitSeed", "$($Row.split_seed)",
        "-CachePolicy", $Row.cache_policy,
        "-ControllerMode", $Row.controller,
        "-ComparisonStrategy", $Row.strategy,
        "-AdaptiveMonitorMetric", "map50_95",
        "-AdaptiveMinEpochs", "$($Row.controller_min_epochs)",
        "-AdaptivePatience", "$($Row.controller_patience)",
        "-AdaptiveSmoothingWindow", "$($ScenarioConfig.DeltaWindow)",
        "-AdaptiveMinDeltaMap50", "$($ScenarioConfig.DeltaMinDelta)",
        "-AdaptiveMinMapeMap50PerWh", "$($ScenarioConfig.DeltaMinMape)",
        "-StandardMinDelta", "$StandardEarlyStoppingMinDelta",
        "-UncertaintyTargetEpoch", "$MaxEpochs",
        "-UncertaintyEpsilon", "$($ScenarioConfig.UncertaintyEpsilon)",
        "-UncertaintyAlpha", "$($ScenarioConfig.UncertaintyAlpha)",
        "-UncertaintyBootstrapSamples", "$($ScenarioConfig.UncertaintyBootstrapSamples)",
        "-UncertaintyMinFitPoints", "12",
        "-UncertaintyMinQuality", "$($ScenarioConfig.UncertaintyMinQuality)",
        "-UncertaintyRequireLowMape", "true",
        "-IdleSampleSeconds", "$IdleSampleSeconds",
        "-SkipDependencyInstall",
        "-WaitForCompletion"
    )
    if ($FirstRun -and $ForceWorkspaceSync) { $arguments += "-ForceWorkspaceSync" }
    if ($FirstRun -and $ForceDatasetSync) { $arguments += "-ForceDatasetSync" }

    Write-Host "[$($Row.scenario), seed $($Row.training_seed), order $($Row.run_order)] $($Row.strategy)" -ForegroundColor Cyan
    & powershell @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Study run failed: scenario=$($Row.scenario), seed=$($Row.training_seed), strategy=$($Row.strategy)"
    }
}

$plan = @(New-StudyPlan)
$planDirectory = Split-Path -Parent $PlanPath
if ($planDirectory) { New-Item -ItemType Directory -Force -Path $planDirectory | Out-Null }
$plan | Export-Csv -Path $PlanPath -NoTypeInformation -Encoding UTF8

Write-Host "CARPK stop-policy study: $($plan.Count) sequential runs" -ForegroundColor Green
Write-Host "Scenarios: $($Scenarios -join ', '); training seeds: $($TrainingSeeds -join ', '); split seed: $SplitSeed" -ForegroundColor Green
Write-Host "Run matrix: $PlanPath" -ForegroundColor Green

if ($PlanOnly) {
    $plan | Format-Table scenario, training_seed, run_order, strategy, epochs, controller -AutoSize
    Write-Host "PlanOnly: no SLURM job was submitted." -ForegroundColor Yellow
    return
}

$firstRun = $true
foreach ($row in $plan) {
    if (-not $firstRun -and $CooldownSeconds -gt 0) {
        Write-Host "Cooldown for $CooldownSeconds seconds before the next measured run ..." -ForegroundColor DarkCyan
        Start-Sleep -Seconds $CooldownSeconds
    }
    Invoke-StudyRun -Row $row -ScenarioConfig (Get-ScenarioConfig -Name $row.scenario) -FirstRun:$firstRun
    $firstRun = $false
}

Write-Host "Study completed. Analyze results with slurm/analyze_stop_policy_study.py." -ForegroundColor Green
