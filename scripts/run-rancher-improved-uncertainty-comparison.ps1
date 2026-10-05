[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$SevenRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_7Jobs_2026-07-13",
    [string]$SixtyRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125",
    [string]$SnapshotRoot = "D:\Masterarbeit_Vergleichssicherung",
    [ValidateRange(1, 2147483647)]
    [int]$ExpectedJobId = 8,
    [switch]$SkipRestore,
    [switch]$SkipSlurmControllerRestart,
    [switch]$SkipJobIdInitialization,
    [string]$BackupMlflowPod = "",
    [string]$BackupMlflowContainer = "",
    [string]$BackupMlflowBackendPath = "/mlflow",
    [string]$BackupMlflowArtifactsPath = "/mlruns",
    [switch]$ValidateMlflowViaSlurmd,
    [switch]$SkipBackupAndDashboard
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$Kubeconfig = [System.IO.Path]::GetFullPath($Kubeconfig)
$SevenRunSnapshot = [System.IO.Path]::GetFullPath($SevenRunSnapshot)
$SixtyRunSnapshot = [System.IO.Path]::GetFullPath($SixtyRunSnapshot)

function Invoke-Native {
    param([Parameter(Mandatory = $true)][string]$Command, [Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}

function Get-NativeText {
    param([Parameter(Mandatory = $true)][string]$Command, [Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

foreach ($required in @(
    $Kubeconfig,
    (Join-Path $SevenRunSnapshot "snapshot-manifest.json"),
    (Join-Path $SixtyRunSnapshot "snapshot-manifest.json"),
    (Join-Path $RepoRoot "slurm\conformal_calibration_seed0_holdout.json")
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required input is missing: $required" }
}

Write-Host "CARPK comparison: historical seven runs plus improved uncertainty-aware controller" -ForegroundColor Green
Write-Host "Job 8 configuration: train128-scratch, seed=0, split=0, YOLOv8n from scratch, max=100 epochs" -ForegroundColor Green

if (-not $SkipRestore) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\restore-rancher-comparison.ps1") `
        -SnapshotPath $SevenRunSnapshot -Kubeconfig $Kubeconfig -Namespace $Namespace `
        -SkipDashboard -Force
    if ($LASTEXITCODE -ne 0) { throw "The historical seven-run snapshot could not be restored." }

    # The historical restore script replaces MLflow data in the existing pod.
    # Do not roll the deployment: this stack intentionally uses emptyDir here.
}

Write-Host "Initialize a clean SLURM sequence so that the measured run is Job 8 ..." -ForegroundColor Cyan
if (-not $SkipSlurmControllerRestart) {
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deployment/slurmctld"
    )
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/slurmctld", "--timeout=300s"
    )
    Start-Sleep -Seconds 5
}
if (-not $SkipJobIdInitialization) {
    for ($expectedId = 1; $expectedId -lt $ExpectedJobId; $expectedId++) {
        $actualId = Get-NativeText -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--",
            "bash", "-lc", "sbatch --parsable --job-name=history-slot-$expectedId --wrap='/bin/true'"
        )
        $actualId = ($actualId -split ";")[0].Trim()
        if ($actualId -ne "$expectedId") { throw "Expected placeholder Job $expectedId, but SLURM returned Job $actualId." }
    }
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        $queued = Get-NativeText -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--",
            "bash", "-lc", "squeue -h | wc -l"
        )
        if ([int]$queued -eq 0) { break }
        if ($attempt -eq 60) { throw "SLURM placeholder jobs did not finish." }
        Start-Sleep -Seconds 1
    }
}

$submitArgs = @(
    "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\submit-rancher-job.ps1"),
    "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace,
    "-ExperimentName", "carpk-stop-policy-improved-uncertainty",
    "-ScenarioName", "train128-scratch", "-Dataset", "carpk", "-Model", "yolov8",
    "-ModelVersion", "yolov8n.yaml", "-Epochs", "100", "-BatchSize", "8", "-ImageSize", "640",
    "-Patience", "100", "-TrainSize", "128", "-Pretrained", "false",
    "-TrainingSeed", "0", "-SplitSeed", "0", "-CachePolicy", "ram",
    "-ControllerMode", "conformal_energy_aware",
    "-ComparisonStrategy", "improved_uncertainty_aware_controller",
    "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "40", "-AdaptivePatience", "3",
    "-AdaptiveSmoothingWindow", "5", "-AdaptiveMinMapeMap50PerWh", "0.00005",
    "-UncertaintyTargetEpoch", "100", "-UncertaintyAlpha", "0.05",
    "-UncertaintyBootstrapSamples", "8", "-UncertaintyMinFitPoints", "12", "-UncertaintyMinQuality", "0.60",
    "-ConformalCalibrationPath", "/workspace/slurm/conformal_calibration_seed0_holdout.json",
    "-ConformalRegretTolerance", "0.015", "-ConformalFutureEfficiencyThreshold", "0.05",
    "-ConformalMaxIntervalWidth", "0.05", "-ConformalEnergyWindow", "5",
    "-ConformalRequireCalibration", "true", "-ControllerEvaluationInterval", "3",
    "-IdleSampleSeconds", "10", "-SkipDependencyInstall", "-ForceWorkspaceSync", "-WaitForCompletion"
)
if ($ValidateMlflowViaSlurmd) { $submitArgs += "-ValidateMlflowViaSlurmd" }

Write-Host "Submit improved controller as SLURM Job $ExpectedJobId ..." -ForegroundColor Cyan
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$submitOutput = & powershell @submitArgs 2>&1 | Tee-Object -Variable capturedOutput
$submitExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($submitExitCode -ne 0) { throw "Improved uncertainty-aware run failed.`n$($capturedOutput | Out-String)" }
$submitted = [regex]::Match(($capturedOutput | Out-String), "Submitted batch job\s+(\d+)")
if (-not $submitted.Success -or $submitted.Groups[1].Value -ne "$ExpectedJobId") {
    throw "The measured run was expected to be Job $ExpectedJobId. Submit output:`n$($capturedOutput | Out-String)"
}

$summaryJson = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmd", "-c", "slurmd", "--",
    "cat", "/workspace/energy_metrics/epoch_summary_job_$ExpectedJobId.json"
) | ConvertFrom-Json
if ($summaryJson.comparison_strategy -ne "improved_uncertainty_aware_controller") {
    throw "Job $ExpectedJobId has an unexpected strategy: $($summaryJson.comparison_strategy)"
}
Write-Host "Job $ExpectedJobId completed $($summaryJson.epochs.Count) epochs; stopped=$($summaryJson.adaptive_stopped)." -ForegroundColor Green

if ($SkipBackupAndDashboard) { return }

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$snapshot = Join-Path $SnapshotRoot "CARPK_StopPolicy_8Jobs_ImprovedUncertainty_$stamp"
$dashboardTemplate = Join-Path $env:TEMP "improved-uncertainty-dashboard-$PID.json"
try {
    & python (Join-Path $RepoRoot "scripts\prepare-improved-uncertainty-dashboard.py") `
        --source (Join-Path $SevenRunSnapshot "config\slurm-energy-overview.json") `
        --output $dashboardTemplate
    if ($LASTEXITCODE -ne 0) { throw "Dashboard template generation failed." }

    $backupArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\backup-rancher-comparison-v2.ps1"),
        "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-OutputPath", $snapshot,
        "-SnapshotName", "carpk-stop-policy-improved-uncertainty-8run", "-DashboardSource", $dashboardTemplate
    )
    if ($BackupMlflowPod) {
        $backupArgs += @(
            "-MlflowPod", $BackupMlflowPod, "-MlflowContainer", $BackupMlflowContainer,
            "-MlflowBackendPath", $BackupMlflowBackendPath, "-MlflowArtifactsPath", $BackupMlflowArtifactsPath
        )
    }
    & powershell @backupArgs
    if ($LASTEXITCODE -ne 0) { throw "Eight-run snapshot failed." }

    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\deploy-rancher-comparison-dashboards.ps1") `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -SevenRunSnapshot $SevenRunSnapshot `
        -SixtyRunSnapshot $SixtyRunSnapshot -EightRunSnapshot $snapshot `
        -EightRunDashboardTemplate $dashboardTemplate -EightRunTitle "8 Run - Improved Uncertainty"
    if ($LASTEXITCODE -ne 0) { throw "Dashboard deployment failed." }
}
finally {
    Remove-Item -LiteralPath $dashboardTemplate -Force -ErrorAction SilentlyContinue
}

Write-Host "Comparison completed and preserved: $snapshot" -ForegroundColor Green
Write-Host "Grafana dashboard: 8 Run - Improved Uncertainty" -ForegroundColor Green
