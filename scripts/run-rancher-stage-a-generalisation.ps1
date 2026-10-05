[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$SevenRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_7Jobs_2026-07-13",
    [string]$SixtyRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125",
    [string]$NineRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_9Jobs_CompleteRerun_2026-07-19_155800",
    [string]$SnapshotRoot = "D:\Masterarbeit_Vergleichssicherung",
    [int]$CooldownSeconds = 5,
    [switch]$PlanOnly,
    [switch]$SkipBackupAndDashboard
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$Kubeconfig = [System.IO.Path]::GetFullPath($Kubeconfig)

function Get-NativeText {
    param([string]$Command, [string[]]$Arguments)
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

function Invoke-Native {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}

$runs = @(
    [pscustomobject]@{ Id = 1; Seed = 0; Strategy = "full100"; Controller = "none"; MinEpochs = 100; ControllerPatience = 100 },
    [pscustomobject]@{ Id = 2; Seed = 0; Strategy = "hybrid_pareto_uncertainty_controller"; Controller = "hybrid_pareto_energy_aware"; MinEpochs = 30; ControllerPatience = 3 },
    [pscustomobject]@{ Id = 3; Seed = 1; Strategy = "hybrid_pareto_uncertainty_controller"; Controller = "hybrid_pareto_energy_aware"; MinEpochs = 30; ControllerPatience = 3 },
    [pscustomobject]@{ Id = 4; Seed = 1; Strategy = "full100"; Controller = "none"; MinEpochs = 100; ControllerPatience = 100 },
    [pscustomobject]@{ Id = 5; Seed = 2; Strategy = "full100"; Controller = "none"; MinEpochs = 100; ControllerPatience = 100 },
    [pscustomobject]@{ Id = 6; Seed = 2; Strategy = "hybrid_pareto_uncertainty_controller"; Controller = "hybrid_pareto_energy_aware"; MinEpochs = 30; ControllerPatience = 3 },
    [pscustomobject]@{ Id = 7; Seed = 3; Strategy = "hybrid_pareto_uncertainty_controller"; Controller = "hybrid_pareto_energy_aware"; MinEpochs = 30; ControllerPatience = 3 },
    [pscustomobject]@{ Id = 8; Seed = 3; Strategy = "full100"; Controller = "none"; MinEpochs = 100; ControllerPatience = 100 },
    [pscustomobject]@{ Id = 9; Seed = 4; Strategy = "full100"; Controller = "none"; MinEpochs = 100; ControllerPatience = 100 },
    [pscustomobject]@{ Id = 10; Seed = 4; Strategy = "hybrid_pareto_uncertainty_controller"; Controller = "hybrid_pareto_energy_aware"; MinEpochs = 30; ControllerPatience = 3 }
)

Write-Host "Stage A: paired CARPK Full100 vs Hybrid-Pareto across five seeds" -ForegroundColor Green
$runs | Format-Table Id, Seed, Strategy, Controller, MinEpochs, ControllerPatience -AutoSize
if ($PlanOnly) { return }

$slurmdPod = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd",
    "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}"
)
$slurmdPod = @($slurmdPod -split "\s+" | Where-Object { $_ })[0]
if (-not $slurmdPod) { throw "No running slurmd pod found." }

Write-Host "Reset active metrics, models, logs, MLflow, and SLURM IDs ..." -ForegroundColor Cyan
$queued = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--",
    "bash", "-lc", "squeue -h -o '%A'"
)
if ($queued) {
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--",
        "bash", "-lc", "scancel $($queued -replace '\s+', ' ')"
    )
}
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--",
    "bash", "-lc",
    "rm -rf /workspace/energy_metrics/* /workspace/logs/* /workspace-cache/model_registry/job_*; mkdir -p /workspace/energy_metrics/node_exporter /workspace/logs /workspace-cache/model_registry"
)

$mlflowReset = "pid=`$(pgrep -f '^/opt/conda/bin/python .*mlflow server.*--port 5000' | head -1 || true); if [ -n `"`$pid`" ]; then kill `"`$pid`" || true; sleep 2; fi; rm -rf /workspace/mlflow-fallback/backend /workspace/mlflow-fallback/artifacts; mkdir -p /workspace/mlflow-fallback/backend /workspace/mlflow-fallback/artifacts; nohup mlflow server --backend-store-uri sqlite:////workspace/mlflow-fallback/backend/mlflow.db --default-artifact-root /workspace/mlflow-fallback/artifacts --host 0.0.0.0 --port 5000 --allowed-hosts '*' >/workspace/mlflow-fallback/mlflow.log 2>&1 &"
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--",
    "bash", "-lc", $mlflowReset
)
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
    Start-Sleep -Seconds 1
}
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deployment/slurmctld"
)
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/slurmctld", "--timeout=300s"
)

foreach ($run in $runs) {
    if ($run.Id -gt 1 -and $CooldownSeconds -gt 0) { Start-Sleep -Seconds $CooldownSeconds }
    $submit = @(
        "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\submit-rancher-job.ps1"),
        "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace,
        "-ExperimentName", "carpk-stage-a-hybrid-generalisation",
        "-ScenarioName", "train128-scratch", "-Dataset", "carpk", "-Model", "yolov8",
        "-ModelVersion", "yolov8n.yaml", "-Epochs", "100", "-BatchSize", "8", "-ImageSize", "640",
        "-Patience", "100", "-TrainSize", "128", "-Pretrained", "false",
        "-TrainingSeed", "$($run.Seed)", "-SplitSeed", "0", "-CachePolicy", "ram",
        "-ControllerMode", $run.Controller, "-ComparisonStrategy", $run.Strategy,
        "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "$($run.MinEpochs)",
        "-AdaptivePatience", "$($run.ControllerPatience)", "-AdaptiveSmoothingWindow", "3",
        "-AdaptiveMinDeltaMap50", "0.0005", "-AdaptiveMinMapeMap50PerWh", "0.00005",
        "-StandardMinDelta", "0.0005", "-UncertaintyTargetEpoch", "100",
        "-UncertaintyEpsilon", "0.02", "-UncertaintyAlpha", "0.10",
        "-UncertaintyBootstrapSamples", "32", "-UncertaintyMinFitPoints", "8",
        "-UncertaintyMinQuality", "0", "-UncertaintyRequireLowMape", "false",
        "-ConformalCalibrationPath", "/workspace/slurm/conformal_calibration_seed0_holdout.json",
        "-ConformalRegretTolerance", "0.015", "-ConformalFutureEfficiencyThreshold", "0.05",
        "-ConformalMaxIntervalWidth", "0.05", "-ConformalEnergyWindow", "5",
        "-ConformalRequireCalibration", "true", "-ControllerEvaluationInterval", "1",
        "-HybridQualityTarget", "0.70", "-HybridPlateauWindow", "12",
        "-HybridPlateauSlopeThreshold", "0.0015", "-HybridThresholdGrowth", "0.25",
        "-HybridCandidateDecay", "0.5", "-HybridLateEpochFraction", "0.90",
        "-HybridLatePatience", "2", "-HybridMaxEpochBudget", "95",
        "-HybridMaxNetEnergyWh", "0", "-HybridRegretWeight", "1.0", "-HybridEnergyWeight", "0.05",
        "-IdleSampleSeconds", "10", "-SkipDependencyInstall", "-WaitForCompletion", "-ValidateMlflowViaSlurmd"
    )
    if ($run.Id -eq 1) { $submit += "-ForceWorkspaceSync" }
    if ($run.Controller -in @("conformal_energy_aware", "hybrid_pareto_energy_aware")) {
        $submit[$submit.IndexOf("-UncertaintyAlpha") + 1] = "0.05"
        $submit[$submit.IndexOf("-UncertaintyBootstrapSamples") + 1] = "8"
        $submit[$submit.IndexOf("-UncertaintyMinFitPoints") + 1] = "12"
        $submit[$submit.IndexOf("-UncertaintyMinQuality") + 1] = "0.60"
    }

    Write-Host "Run $($run.Id)/10: seed=$($run.Seed), strategy=$($run.Strategy)" -ForegroundColor Cyan
    & powershell @submit
    if ($LASTEXITCODE -ne 0) { throw "Run $($run.Id) failed: $($run.Strategy)" }
    $summary = Get-NativeText -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--",
        "cat", "/workspace/energy_metrics/epoch_summary_job_$($run.Id).json"
    ) | ConvertFrom-Json
    if ([string]$summary.comparison_strategy -ne $run.Strategy) {
        throw "Job $($run.Id) has strategy '$($summary.comparison_strategy)', expected '$($run.Strategy)'."
    }
    if ([int]$summary.training_seed -ne $run.Seed) {
        throw "Job $($run.Id) has training seed '$($summary.training_seed)', expected '$($run.Seed)'."
    }
}

if ($SkipBackupAndDashboard) { return }
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$snapshot = Join-Path $SnapshotRoot "CARPK_StageA_HybridGeneralisation_Min30Eval1_10Jobs_$stamp"
$stageDashboard = Join-Path $env:TEMP "stage-a-dashboard-$PID.json"
try {
    & python (Join-Path $RepoRoot "scripts\build-stage-a-dashboard.py") --source (Join-Path $RepoRoot "grafana\dashboards\8-run.json") --output $stageDashboard
    if ($LASTEXITCODE -ne 0) { throw "Dashboard preparation failed." }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\backup-rancher-comparison-v2.ps1") `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -OutputPath $snapshot `
        -SnapshotName "carpk-stage-a-hybrid-generalisation" -DashboardSource $stageDashboard `
        -MlflowPod $slurmdPod -MlflowContainer "slurmd" `
        -MlflowBackendPath "/workspace/mlflow-fallback/backend" -MlflowArtifactsPath "/workspace/mlflow-fallback/artifacts"
    if ($LASTEXITCODE -ne 0) { throw "Stage A snapshot failed." }
    & python (Join-Path $RepoRoot "scripts\analyze-stage-a-generalisation.py") --snapshot $snapshot --regret-budget-pp 1.5
    if ($LASTEXITCODE -ne 0) { throw "Stage A paired analysis failed." }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\deploy-rancher-comparison-dashboards.ps1") `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -SevenRunSnapshot $SevenRunSnapshot `
        -SixtyRunSnapshot $SixtyRunSnapshot -EightRunSnapshot $NineRunSnapshot -EightRunExpectedJobs 9 `
        -EightRunDashboardTemplate (Join-Path $RepoRoot "grafana\dashboards\8-run.json") -EightRunTitle "9 Run - Complete Re-Run"
    if ($LASTEXITCODE -ne 0) { throw "Historical dashboard deployment failed." }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\deploy-rancher-stage-a-dashboard.ps1") `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -Snapshot $snapshot -Dashboard $stageDashboard
    if ($LASTEXITCODE -ne 0) { throw "Stage A dashboard deployment failed." }
}
finally {
    Remove-Item -LiteralPath $stageDashboard -Force -ErrorAction SilentlyContinue
}
Write-Host "Stage A comparison finished: $snapshot" -ForegroundColor Green
