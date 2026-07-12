[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [ValidateSet("train64-scratch", "train128-scratch", "train256-pretrained", "yolov8s-train128", "yolo11n-train128")]
    [string]$Scenario = "train128-scratch",
    [string]$ExperimentPrefix = "carpk-stop-policy",
    [int]$MaxEpochs = 100,
    [int]$StandardEarlyStoppingPatience = 20,
    [int[]]$FixedEpochBudgets = @(20, 30, 50),
    [int]$Seed = 0,
    [switch]$WaitForCompletion,
    [switch]$SkipDependencyInstall,
    [switch]$ForceWorkspaceSync,
    [switch]$ForceDatasetSync
)

$ErrorActionPreference = "Stop"

function New-ScenarioConfig {
    param([string]$Name)

    switch ($Name) {
        "train64-scratch" {
            return @{
                Description = "Sehr hart: nur 64 Trainingsbilder, YOLOv8n ohne Pretraining."
                Model = "yolov8"
                ModelVersion = "yolov8n.yaml"
                Pretrained = "false"
                TrainSize = 64
                BatchSize = 8
                ImageSize = 640
                MonitorMetric = "map50_95"
                DeltaMinEpochs = 12
                DeltaPatience = 3
                DeltaSmoothingWindow = 3
                DeltaMinDelta = 0.0005
                DeltaMinMape = 0.00005
                UncertaintyWarmup = 15
                UncertaintyPatience = 3
                UncertaintyEpsilon = 0.02
                UncertaintyAlpha = 0.10
                UncertaintyBootstrapSamples = 28
            }
        }
        "train128-scratch" {
            return @{
                Description = "Empfohlen: 128 Trainingsbilder, YOLOv8n ohne Pretraining."
                Model = "yolov8"
                ModelVersion = "yolov8n.yaml"
                Pretrained = "false"
                TrainSize = 128
                BatchSize = 8
                ImageSize = 640
                MonitorMetric = "map50_95"
                DeltaMinEpochs = 15
                DeltaPatience = 3
                DeltaSmoothingWindow = 3
                DeltaMinDelta = 0.0005
                DeltaMinMape = 0.00005
                UncertaintyWarmup = 18
                UncertaintyPatience = 3
                UncertaintyEpsilon = 0.02
                UncertaintyAlpha = 0.10
                UncertaintyBootstrapSamples = 32
            }
        }
        "train256-pretrained" {
            return @{
                Description = "Mittel: 256 Trainingsbilder, YOLOv8n mit Pretraining."
                Model = "yolov8"
                ModelVersion = "yolov8n.pt"
                Pretrained = "true"
                TrainSize = 256
                BatchSize = 8
                ImageSize = 640
                MonitorMetric = "map50_95"
                DeltaMinEpochs = 15
                DeltaPatience = 3
                DeltaSmoothingWindow = 3
                DeltaMinDelta = 0.0003
                DeltaMinMape = 0.00003
                UncertaintyWarmup = 20
                UncertaintyPatience = 3
                UncertaintyEpsilon = 0.01
                UncertaintyAlpha = 0.05
                UncertaintyBootstrapSamples = 32
            }
        }
        "yolov8s-train128" {
            return @{
                Description = "Groesseres Modell: 128 Trainingsbilder, YOLOv8s mit Pretraining."
                Model = "yolov8"
                ModelVersion = "yolov8s.pt"
                Pretrained = "true"
                TrainSize = 128
                BatchSize = 6
                ImageSize = 640
                MonitorMetric = "map50_95"
                DeltaMinEpochs = 15
                DeltaPatience = 3
                DeltaSmoothingWindow = 3
                DeltaMinDelta = 0.0004
                DeltaMinMape = 0.00004
                UncertaintyWarmup = 20
                UncertaintyPatience = 3
                UncertaintyEpsilon = 0.01
                UncertaintyAlpha = 0.05
                UncertaintyBootstrapSamples = 32
            }
        }
        "yolo11n-train128" {
            return @{
                Description = "Alternatives Modell: 128 Trainingsbilder, YOLO11n mit Pretraining."
                Model = "yolov11"
                ModelVersion = "yolo11n.pt"
                Pretrained = "true"
                TrainSize = 128
                BatchSize = 8
                ImageSize = 640
                MonitorMetric = "map50_95"
                DeltaMinEpochs = 15
                DeltaPatience = 3
                DeltaSmoothingWindow = 3
                DeltaMinDelta = 0.0004
                DeltaMinMape = 0.00004
                UncertaintyWarmup = 20
                UncertaintyPatience = 3
                UncertaintyEpsilon = 0.01
                UncertaintyAlpha = 0.05
                UncertaintyBootstrapSamples = 32
            }
        }
        default {
            throw "Unbekanntes Szenario: $Name"
        }
    }
}

function Invoke-ComparisonRun {
    param(
        [hashtable]$ScenarioConfig,
        [string]$StrategyName,
        [int]$Epochs,
        [int]$Patience,
        [string]$ControllerMode = "none",
        [double]$DeltaMinDelta = 0.001,
        [double]$DeltaMinMape = 0.0001,
        [int]$DeltaMinEpochs = 20,
        [int]$DeltaPatience = 3,
        [int]$DeltaSmoothingWindow = 3,
        [double]$UncertaintyEpsilon = 0.01,
        [double]$UncertaintyAlpha = 0.05,
        [int]$UncertaintyWarmup = 20,
        [int]$UncertaintyPatience = 3,
        [int]$UncertaintyBootstrapSamples = 24
    )

    $experimentName = "$ExperimentPrefix-$Scenario-$StrategyName"
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", ".\scripts\submit-rancher-job.ps1",
        "-Kubeconfig", $Kubeconfig,
        "-Namespace", $Namespace,
        "-ExperimentName", $experimentName,
        "-Dataset", "carpk",
        "-Model", $ScenarioConfig.Model,
        "-ModelVersion", $ScenarioConfig.ModelVersion,
        "-Epochs", "$Epochs",
        "-BatchSize", "$($ScenarioConfig.BatchSize)",
        "-ImageSize", "$($ScenarioConfig.ImageSize)",
        "-Patience", "$Patience",
        "-TrainSize", "$($ScenarioConfig.TrainSize)",
        "-Pretrained", $ScenarioConfig.Pretrained,
        "-Seed", "$Seed",
        "-ControllerMode", $ControllerMode,
        "-ComparisonStrategy", $StrategyName,
        "-AdaptiveMonitorMetric", "$($ScenarioConfig.MonitorMetric)",
        "-AdaptiveMinEpochs", "$DeltaMinEpochs",
        "-AdaptivePatience", "$DeltaPatience",
        "-AdaptiveSmoothingWindow", "$DeltaSmoothingWindow",
        "-AdaptiveMinDeltaMap50", "$DeltaMinDelta",
        "-AdaptiveMinMapeMap50PerWh", "$DeltaMinMape",
        "-UncertaintyTargetEpoch", "$MaxEpochs",
        "-UncertaintyEpsilon", "$UncertaintyEpsilon",
        "-UncertaintyAlpha", "$UncertaintyAlpha",
        "-UncertaintyBootstrapSamples", "$UncertaintyBootstrapSamples",
        "-UncertaintyMinFitPoints", "8"
    )

    if ($SkipDependencyInstall) { $args += "-SkipDependencyInstall" }
    if ($ForceWorkspaceSync) { $args += "-ForceWorkspaceSync" }
    if ($ForceDatasetSync) { $args += "-ForceDatasetSync" }
    if ($WaitForCompletion) { $args += "-WaitForCompletion" }

    Write-Host ""
    Write-Host "Starte Vergleichslauf: $StrategyName" -ForegroundColor Cyan
    Write-Host "  Epochen=$Epochs, Patience=$Patience, ControllerMode=$ControllerMode" -ForegroundColor DarkCyan
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Vergleichslauf $StrategyName fehlgeschlagen."
    }
}

$scenarioConfig = New-ScenarioConfig -Name $Scenario

Write-Host "Stop-Policy-Vergleich fuer CARPK" -ForegroundColor Green
Write-Host "Szenario: $Scenario" -ForegroundColor Green
Write-Host "Beschreibung: $($scenarioConfig.Description)" -ForegroundColor Green

Invoke-ComparisonRun -ScenarioConfig $scenarioConfig -StrategyName "full100" -Epochs $MaxEpochs -Patience $MaxEpochs -ControllerMode "none"
Invoke-ComparisonRun -ScenarioConfig $scenarioConfig -StrategyName "standard_early_stopping" -Epochs $MaxEpochs -Patience $StandardEarlyStoppingPatience -ControllerMode "none"
Invoke-ComparisonRun `
    -ScenarioConfig $scenarioConfig `
    -StrategyName "delta_mape_controller" `
    -Epochs $MaxEpochs `
    -Patience $MaxEpochs `
    -ControllerMode "delta_mape" `
    -DeltaMinDelta $scenarioConfig.DeltaMinDelta `
    -DeltaMinMape $scenarioConfig.DeltaMinMape `
    -DeltaMinEpochs $scenarioConfig.DeltaMinEpochs `
    -DeltaPatience $scenarioConfig.DeltaPatience `
    -DeltaSmoothingWindow $scenarioConfig.DeltaSmoothingWindow
Invoke-ComparisonRun `
    -ScenarioConfig $scenarioConfig `
    -StrategyName "uncertainty_aware_controller" `
    -Epochs $MaxEpochs `
    -Patience $MaxEpochs `
    -ControllerMode "uncertainty_aware" `
    -DeltaMinEpochs $scenarioConfig.UncertaintyWarmup `
    -DeltaPatience $scenarioConfig.UncertaintyPatience `
    -UncertaintyEpsilon $scenarioConfig.UncertaintyEpsilon `
    -UncertaintyAlpha $scenarioConfig.UncertaintyAlpha `
    -UncertaintyWarmup $scenarioConfig.UncertaintyWarmup `
    -UncertaintyPatience $scenarioConfig.UncertaintyPatience `
    -UncertaintyBootstrapSamples $scenarioConfig.UncertaintyBootstrapSamples

foreach ($budget in $FixedEpochBudgets) {
    Invoke-ComparisonRun -ScenarioConfig $scenarioConfig -StrategyName "fixed_epoch_$budget" -Epochs $budget -Patience $budget -ControllerMode "none"
}

Write-Host ""
Write-Host "Stop-Policy-Vergleich abgeschlossen." -ForegroundColor Green
