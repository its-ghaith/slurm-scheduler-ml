[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$SnapshotRoot = "D:\Masterarbeit_Vergleichssicherung",
    [string]$SevenRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_7Jobs_2026-07-13",
    [string]$SixtyRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125",
    [string]$NineRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_9Jobs_CompleteRerun_2026-07-19_155800",
    [string]$StageASnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StageA_HybridGeneralisation_Min30Eval1_10Jobs_2026-07-20_234033",
    [string]$StageBSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StageB_ScenarioGeneralisation_Min30Eval1_14Jobs_2026-07-21_204649",
    [int]$CooldownSeconds = 5,
    [switch]$PlanOnly,
    [switch]$SkipHistoricalDashboardRestore
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$Kubeconfig = [System.IO.Path]::GetFullPath($Kubeconfig)

function Invoke-Native([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}
function Get-NativeText([string]$Command, [string[]]$Arguments) {
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

$cases = @()
foreach ($seed in 0..4) {
    $cases += [pscustomobject]@{ Stage = "A"; Scenario = "train128-scratch"; TrainingSeed = $seed; TrainSize = 128; ModelVersion = "yolov8n.yaml"; Pretrained = "false" }
}
$cases += @(
    [pscustomobject]@{ Stage = "B"; Scenario = "train128-scratch-yolov8n"; TrainingSeed = 0; TrainSize = 128; ModelVersion = "yolov8n.yaml"; Pretrained = "false" },
    [pscustomobject]@{ Stage = "B"; Scenario = "train256-scratch-yolov8n"; TrainingSeed = 0; TrainSize = 256; ModelVersion = "yolov8n.yaml"; Pretrained = "false" },
    [pscustomobject]@{ Stage = "B"; Scenario = "trainfull-scratch-yolov8n"; TrainingSeed = 0; TrainSize = 0; ModelVersion = "yolov8n.yaml"; Pretrained = "false" },
    [pscustomobject]@{ Stage = "B"; Scenario = "train128-pretrained-yolov8n"; TrainingSeed = 0; TrainSize = 128; ModelVersion = "yolov8n.pt"; Pretrained = "true" },
    [pscustomobject]@{ Stage = "B"; Scenario = "train256-pretrained-yolov8n"; TrainingSeed = 0; TrainSize = 256; ModelVersion = "yolov8n.pt"; Pretrained = "true" },
    [pscustomobject]@{ Stage = "B"; Scenario = "trainfull-pretrained-yolov8n"; TrainingSeed = 0; TrainSize = 0; ModelVersion = "yolov8n.pt"; Pretrained = "true" },
    [pscustomobject]@{ Stage = "B"; Scenario = "train256-pretrained-yolov8s"; TrainingSeed = 0; TrainSize = 256; ModelVersion = "yolov8s.pt"; Pretrained = "true" }
)
$runs = @()
for ($caseIndex = 0; $caseIndex -lt $cases.Count; $caseIndex++) {
    $case = $cases[$caseIndex]
    # Alternate order to reduce systematic cache, temperature, and cluster-time bias.
    $strategies = if (($caseIndex % 2) -eq 0) { @("full100", "generalized_energy_guard_20pct") } else { @("generalized_energy_guard_20pct", "full100") }
    foreach ($strategy in $strategies) {
        $runs += [pscustomobject]@{
            Id = $runs.Count + 1; Stage = $case.Stage; Scenario = $case.Scenario
            TrainingSeed = $case.TrainingSeed; TrainSize = $case.TrainSize
            ModelVersion = $case.ModelVersion; Pretrained = $case.Pretrained
            Strategy = $strategy
            Controller = $(if ($strategy -eq "full100") { "none" } else { "generalized_energy_guard" })
            MinEpochs = $(if ($strategy -eq "full100") { 100 } else { 30 })
        }
    }
}

Write-Host "Paired Generalized Energy Guard study: 12 cases, 24 jobs" -ForegroundColor Green
$runs | Format-Table Id, Stage, Scenario, TrainingSeed, Strategy, Controller -AutoSize
if ($PlanOnly) { return }

$slurmdPod = Get-NativeText "kubectl" @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd",
    "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}"
)
$slurmdPod = @($slurmdPod -split '\s+' | Where-Object { $_ })[0]
if (-not $slurmdPod) { throw "No running slurmd pod found." }

Write-Host "Reset active study data; historical snapshots remain untouched ..." -ForegroundColor Cyan
$queued = Get-NativeText "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--", "bash", "-lc", "squeue -h -o '%A'")
if ($queued) { Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--", "bash", "-lc", "scancel $($queued -replace '\s+', ' ')") }
Invoke-Native "kubectl" @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc",
    "rm -rf /workspace/energy_metrics/* /workspace/logs/* /workspace-cache/model_registry/job_*; mkdir -p /workspace/energy_metrics/node_exporter /workspace/logs /workspace-cache/model_registry"
)
$mlflowReset = "pid=`$(pgrep -f '^/opt/conda/bin/python .*mlflow server.*--port 5000' | head -1 || true); if [ -n `"`$pid`" ]; then kill `"`$pid`" || true; sleep 2; fi; rm -rf /workspace/mlflow-fallback/backend /workspace/mlflow-fallback/artifacts; mkdir -p /workspace/mlflow-fallback/backend /workspace/mlflow-fallback/artifacts; nohup mlflow server --backend-store-uri sqlite:////workspace/mlflow-fallback/backend/mlflow.db --default-artifact-root /workspace/mlflow-fallback/artifacts --host 0.0.0.0 --port 5000 --allowed-hosts '*' >/workspace/mlflow-fallback/mlflow.log 2>&1 &"
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc", $mlflowReset)
for ($attempt = 1; $attempt -le 30; $attempt++) {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $slurmdPod -c slurmd -- `
        python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2).read()" `
        2>&1 | Out-Null
    $healthExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($healthExitCode -eq 0) { break }
    if ($attempt -eq 30) { throw "MLflow fallback did not become healthy." }
    Start-Sleep 1
}
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "python", "-c", "from ultralytics import YOLO; YOLO('/workspace/yolov8n.pt'); YOLO('/workspace/yolov8s.pt')")
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deployment/slurmctld")
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/slurmctld", "--timeout=300s")

foreach ($run in $runs) {
    if ($run.Id -gt 1 -and $CooldownSeconds -gt 0) { Start-Sleep $CooldownSeconds }
    $submit = @(
        "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\submit-rancher-job.ps1"),
        "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace,
        "-ExperimentName", "carpk-generalized-energy-guard-20pct",
        "-ScenarioName", $run.Scenario, "-Dataset", "carpk", "-Model", "yolov8",
        "-ModelVersion", $run.ModelVersion, "-Epochs", "100", "-BatchSize", "8", "-ImageSize", "640",
        "-Patience", "100", "-TrainSize", "$($run.TrainSize)", "-Pretrained", $run.Pretrained,
        "-TrainingSeed", "$($run.TrainingSeed)", "-SplitSeed", "0", "-CachePolicy", "ram", "-TimeoutSeconds", "7200",
        "-ControllerMode", $run.Controller, "-ComparisonStrategy", $run.Strategy,
        "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "$($run.MinEpochs)",
        "-AdaptivePatience", "3", "-UncertaintyTargetEpoch", "100", "-UncertaintyAlpha", "0.05",
        "-UncertaintyBootstrapSamples", "24", "-UncertaintyMinFitPoints", "12", "-UncertaintyMinQuality", "0.55",
        "-ConformalRegretTolerance", "0.015", "-ControllerEvaluationInterval", "1",
        "-EnergyGuardTargetSavingFraction", "0.20", "-EnergyGuardSafetyMarginFraction", "0.01",
        "-EnergyGuardPostTrainingReserveWh", "0.90", "-EnergyGuardWindow", "5",
        "-EnergyGuardFallbackEpochFraction", "0.77", "-EnergyGuardStrict", "true",
        "-IdleSampleSeconds", "10", "-SkipDependencyInstall", "-WaitForCompletion", "-ValidateMlflowViaSlurmd"
    )
    if ($run.Id -eq 1) { $submit += "-ForceWorkspaceSync" }
    Write-Host "Job $($run.Id)/24: Stage $($run.Stage), $($run.Scenario), seed=$($run.TrainingSeed), $($run.Strategy)" -ForegroundColor Cyan
    & powershell @submit
    if ($LASTEXITCODE -ne 0) { throw "Study job $($run.Id) failed." }
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$snapshot = Join-Path $SnapshotRoot "CARPK_GeneralizedEnergyGuard_20pct_24Jobs_$stamp"
Invoke-Native "powershell" @(
    "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\backup-rancher-comparison-v2.ps1"),
    "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-OutputPath", $snapshot,
    "-SnapshotName", "carpk-generalized-energy-guard-20pct", "-DashboardSource", (Join-Path $RepoRoot "grafana\dashboards\energy-guard-study.json"),
    "-MlflowPod", $slurmdPod, "-MlflowContainer", "slurmd", "-MlflowBackendPath", "/workspace/mlflow-fallback/backend", "-MlflowArtifactsPath", "/workspace/mlflow-fallback/artifacts"
)
Invoke-Native "python" @((Join-Path $RepoRoot "scripts\analyze-generalized-energy-guard-study.py"), "--snapshot", $snapshot, "--target-saving", "0.20")

if (-not $SkipHistoricalDashboardRestore) {
    Invoke-Native "powershell" @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\deploy-rancher-comparison-dashboards.ps1"), "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-SevenRunSnapshot", $SevenRunSnapshot, "-SixtyRunSnapshot", $SixtyRunSnapshot, "-EightRunSnapshot", $NineRunSnapshot, "-EightRunExpectedJobs", "9", "-EightRunDashboardTemplate", (Join-Path $RepoRoot "grafana\dashboards\8-run.json"), "-EightRunTitle", "9 Run - Complete Re-Run")
    Invoke-Native "powershell" @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\deploy-rancher-stage-a-dashboard.ps1"), "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-Snapshot", $StageASnapshot, "-Dashboard", (Join-Path $StageASnapshot "config\slurm-energy-overview.json"))
    Invoke-Native "powershell" @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\deploy-rancher-stage-b-dashboard.ps1"), "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-Snapshot", $StageBSnapshot, "-Dashboard", (Join-Path $StageBSnapshot "config\slurm-energy-overview.json"))
}
Invoke-Native "powershell" @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\deploy-rancher-energy-guard-dashboard.ps1"), "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-Snapshot", $snapshot)
Write-Host "Paired study completed and saved: $snapshot" -ForegroundColor Green
