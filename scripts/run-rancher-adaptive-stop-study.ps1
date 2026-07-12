[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [ValidateSet("train64-scratch", "train128-scratch", "train256-pretrained", "yolov8s-train128", "yolo11n-train128")]
    [string]$Scenario = "train128-scratch",
    [string]$ExperimentPrefix = "carpk-adaptive-stop",
    [ValidateSet("both", "baseline", "adaptive")]
    [string]$Mode = "both",
    [int]$Epochs = 100,
    [int]$Patience = 20,
    [int]$BatchSize = 8,
    [int]$ImageSize = 640,
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
                Description  = "Sehr hart: nur 64 Trainingsbilder, YOLOv8n ohne Pretraining."
                Model        = "yolov8"
                ModelVersion = "yolov8n.yaml"
                Pretrained   = "false"
                TrainSize    = 64
                BatchSize    = 8
                ImageSize    = 640
                AdaptiveMonitorMetric = "map50_95"
                AdaptiveMinEpochs = 12
                AdaptivePatience = 2
                AdaptiveSmoothingWindow = 3
                AdaptiveMinDeltaMap50 = 0.0005
                AdaptiveMinMapeMap50PerWh = 0.00005
            }
        }
        "train128-scratch" {
            return @{
                Description  = "Empfohlen: 128 Trainingsbilder, YOLOv8n ohne Pretraining."
                Model        = "yolov8"
                ModelVersion = "yolov8n.yaml"
                Pretrained   = "false"
                TrainSize    = 128
                BatchSize    = 8
                ImageSize    = 640
                AdaptiveMonitorMetric = "map50_95"
                AdaptiveMinEpochs = 15
                AdaptivePatience = 2
                AdaptiveSmoothingWindow = 3
                AdaptiveMinDeltaMap50 = 0.0005
                AdaptiveMinMapeMap50PerWh = 0.00005
            }
        }
        "train256-pretrained" {
            return @{
                Description  = "Mittel: 256 Trainingsbilder, YOLOv8n mit Pretraining."
                Model        = "yolov8"
                ModelVersion = "yolov8n.pt"
                Pretrained   = "true"
                TrainSize    = 256
                BatchSize    = 8
                ImageSize    = 640
                AdaptiveMonitorMetric = "map50_95"
                AdaptiveMinEpochs = 15
                AdaptivePatience = 2
                AdaptiveSmoothingWindow = 3
                AdaptiveMinDeltaMap50 = 0.0003
                AdaptiveMinMapeMap50PerWh = 0.00003
            }
        }
        "yolov8s-train128" {
            return @{
                Description  = "Groesseres Modell: 128 Trainingsbilder, YOLOv8s mit Pretraining."
                Model        = "yolov8"
                ModelVersion = "yolov8s.pt"
                Pretrained   = "true"
                TrainSize    = 128
                BatchSize    = 6
                ImageSize    = 640
                AdaptiveMonitorMetric = "map50_95"
                AdaptiveMinEpochs = 15
                AdaptivePatience = 2
                AdaptiveSmoothingWindow = 3
                AdaptiveMinDeltaMap50 = 0.0004
                AdaptiveMinMapeMap50PerWh = 0.00004
            }
        }
        "yolo11n-train128" {
            return @{
                Description  = "Alternatives Modell: 128 Trainingsbilder, YOLO11n mit Pretraining."
                Model        = "yolov11"
                ModelVersion = "yolo11n.pt"
                Pretrained   = "true"
                TrainSize    = 128
                BatchSize    = 8
                ImageSize    = 640
                AdaptiveMonitorMetric = "map50_95"
                AdaptiveMinEpochs = 15
                AdaptivePatience = 2
                AdaptiveSmoothingWindow = 3
                AdaptiveMinDeltaMap50 = 0.0004
                AdaptiveMinMapeMap50PerWh = 0.00004
            }
        }
        default {
            throw "Unbekanntes Szenario: $Name"
        }
    }
}

function Invoke-StudyRun {
    param(
        [hashtable]$ScenarioConfig,
        [string]$Mode
    )

    $experimentName = "$ExperimentPrefix-$Scenario-$Mode"
    $adaptiveEnabled = $Mode -eq "adaptive"
    $argList = @(
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
        "-AdaptiveMonitorMetric", "$($ScenarioConfig.AdaptiveMonitorMetric)"
    )

    if ($SkipDependencyInstall) { $argList += "-SkipDependencyInstall" }
    if ($ForceWorkspaceSync) { $argList += "-ForceWorkspaceSync" }
    if ($ForceDatasetSync) { $argList += "-ForceDatasetSync" }
    if ($WaitForCompletion) { $argList += "-WaitForCompletion" }

    if ($adaptiveEnabled) {
        $argList += @(
            "-AdaptiveEnabled",
            "-AdaptiveMinEpochs", "$($ScenarioConfig.AdaptiveMinEpochs)",
            "-AdaptivePatience", "$($ScenarioConfig.AdaptivePatience)",
            "-AdaptiveSmoothingWindow", "$($ScenarioConfig.AdaptiveSmoothingWindow)",
            "-AdaptiveMinDeltaMap50", "$($ScenarioConfig.AdaptiveMinDeltaMap50)",
            "-AdaptiveMinMapeMap50PerWh", "$($ScenarioConfig.AdaptiveMinMapeMap50PerWh)"
        )
    }

    Write-Host ""
    Write-Host "Starte $Mode-Lauf: $experimentName" -ForegroundColor Cyan
    Write-Host "  Beschreibung: $($ScenarioConfig.Description)" -ForegroundColor DarkCyan
    Write-Host "  ModelVersion: $($ScenarioConfig.ModelVersion), Pretrained: $($ScenarioConfig.Pretrained), TrainSize: $($ScenarioConfig.TrainSize), MonitorMetric: $($ScenarioConfig.AdaptiveMonitorMetric)" -ForegroundColor DarkCyan

    & powershell @argList
    if ($LASTEXITCODE -ne 0) {
        throw "Submit fuer $experimentName fehlgeschlagen."
    }
}

$scenarioConfig = New-ScenarioConfig -Name $Scenario

Write-Host "Adaptive-Stop-Studie fuer CARPK" -ForegroundColor Green
Write-Host "Szenario: $Scenario" -ForegroundColor Green
Write-Host "Beschreibung: $($scenarioConfig.Description)" -ForegroundColor Green
Write-Host "Ziel: YOLO auf CARPK schwieriger machen, damit Adaptive Stop einen echten Trade-off sieht." -ForegroundColor Green

if ($Mode -in @("both", "baseline")) {
    Invoke-StudyRun -ScenarioConfig $scenarioConfig -Mode "baseline"
}

if ($Mode -in @("both", "adaptive")) {
    Invoke-StudyRun -ScenarioConfig $scenarioConfig -Mode "adaptive"
}

Write-Host ""
Write-Host "Studienlauf abgeschlossen." -ForegroundColor Green
Write-Host "Vergleiche in MLflow dieselben Szenarien zwischen '-baseline' und '-adaptive'." -ForegroundColor Green
