[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$SevenRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_7Jobs_2026-07-13",
    [string]$SixtyRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125",
    [string]$SnapshotRoot = "D:\Masterarbeit_Vergleichssicherung",
    [int]$ExpectedJobId = 10,
    [switch]$SkipJobIdInitialization,
    [switch]$ResumeAfterTraining,
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

foreach ($required in @(
    $Kubeconfig,
    (Join-Path $RepoRoot "slurm\conformal_calibration_seed0_holdout.json"),
    (Join-Path $RepoRoot "grafana\dashboards\8-run.json")
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required input is missing: $required" }
}

Write-Host "Validate the preserved 7+1 comparison before submitting Job 10 ..." -ForegroundColor Cyan
$existingIds = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmd", "-c", "slurmd", "--",
    "bash", "-lc", "for f in /workspace/energy_metrics/epoch_summary_job_*.json; do [ -f `"`$f`" ] || continue; basename `"`$f`" | sed -E 's/[^0-9]//g'; done | sort -n | tr '\n' ' '"
)
if (-not $ResumeAfterTraining -and $existingIds.Trim() -ne "1 2 3 4 5 6 7 9") {
    throw "Expected preserved complete jobs '1 2 3 4 5 6 7 9', found '$($existingIds.Trim())'."
}
if ($ResumeAfterTraining -and $existingIds.Trim() -ne "1 2 3 4 5 6 7 9 10") {
    throw "Expected completed jobs '1 2 3 4 5 6 7 9 10', found '$($existingIds.Trim())'."
}

if (-not $ResumeAfterTraining -and -not $SkipJobIdInitialization) {
    Write-Host "Reserve SLURM IDs 1-9 so the measured run receives ID 10 ..." -ForegroundColor Cyan
    for ($expected = 1; $expected -lt $ExpectedJobId; $expected++) {
        $actual = Get-NativeText -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--",
            "bash", "-lc", "sbatch --parsable --job-name=history-slot-$expected --wrap='/bin/true'"
        )
        $actual = ($actual -split ";")[0].Trim()
        if ($actual -ne "$expected") { throw "Expected placeholder Job $expected, but SLURM returned Job $actual." }
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
    "-ExperimentName", "carpk-stop-policy-hybrid-controller",
    "-ScenarioName", "train128-scratch", "-Dataset", "carpk", "-Model", "yolov8",
    "-ModelVersion", "yolov8n.yaml", "-Epochs", "100", "-BatchSize", "8", "-ImageSize", "640",
    "-Patience", "100", "-TrainSize", "128", "-Pretrained", "false",
    "-TrainingSeed", "0", "-SplitSeed", "0", "-CachePolicy", "ram",
    "-ControllerMode", "hybrid_pareto_energy_aware",
    "-ComparisonStrategy", "hybrid_pareto_uncertainty_controller",
    "-AdaptiveMonitorMetric", "map50_95", "-AdaptiveMinEpochs", "30", "-AdaptivePatience", "3",
    "-AdaptiveSmoothingWindow", "5", "-AdaptiveMinMapeMap50PerWh", "0.00005",
    "-UncertaintyTargetEpoch", "100", "-UncertaintyAlpha", "0.05",
    "-UncertaintyBootstrapSamples", "8", "-UncertaintyMinFitPoints", "12", "-UncertaintyMinQuality", "0.60",
    "-ConformalCalibrationPath", "/workspace/slurm/conformal_calibration_seed0_holdout.json",
    "-ConformalRegretTolerance", "0.015", "-ConformalFutureEfficiencyThreshold", "0.05",
    "-ConformalMaxIntervalWidth", "0.05", "-ConformalEnergyWindow", "5",
    "-ConformalRequireCalibration", "true", "-ControllerEvaluationInterval", "1",
    "-HybridQualityTarget", "0.70", "-HybridPlateauWindow", "12",
    "-HybridPlateauSlopeThreshold", "0.0015", "-HybridThresholdGrowth", "0.25",
    "-HybridCandidateDecay", "0.5", "-HybridLateEpochFraction", "0.90",
    "-HybridLatePatience", "2", "-HybridMaxEpochBudget", "95",
    "-HybridMaxNetEnergyWh", "0", "-HybridRegretWeight", "1.0", "-HybridEnergyWeight", "0.05",
    "-IdleSampleSeconds", "10", "-SkipDependencyInstall", "-ForceWorkspaceSync",
    "-WaitForCompletion", "-ValidateMlflowViaSlurmd"
)

if (-not $ResumeAfterTraining) {
    Write-Host "Submit hybrid Pareto uncertainty controller as SLURM Job $ExpectedJobId ..." -ForegroundColor Cyan
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $submitOutput = & powershell @submitArgs 2>&1 | Tee-Object -Variable capturedOutput
    $submitExitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($submitExitCode -ne 0) { throw "Hybrid controller run failed.`n$($capturedOutput | Out-String)" }
    $submitted = [regex]::Match(($capturedOutput | Out-String), "Submitted batch job\s+(\d+)")
    if (-not $submitted.Success -or $submitted.Groups[1].Value -ne "$ExpectedJobId") {
        throw "Expected Job $ExpectedJobId. Submit output:`n$($capturedOutput | Out-String)"
    }
}

$summary = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmd", "-c", "slurmd", "--",
    "cat", "/workspace/energy_metrics/epoch_summary_job_$ExpectedJobId.json"
) | ConvertFrom-Json
if ($summary.controller_mode -ne "hybrid_pareto_energy_aware") {
    throw "Job $ExpectedJobId used unexpected controller '$($summary.controller_mode)'."
}
Write-Host "Job $ExpectedJobId completed $(@($summary.epochs).Count) epochs; stopped=$($summary.adaptive_stopped)." -ForegroundColor Green
if ($SkipBackupAndDashboard) { return }

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$snapshot = Join-Path $SnapshotRoot "CARPK_StopPolicy_9Jobs_HybridController_Min30Eval1_$stamp"
$dashboardTemplate = Join-Path $env:TEMP "hybrid-controller-dashboard-$PID.json"
$slurmdPod = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd",
    "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}"
)
$slurmdPod = @($slurmdPod -split "\s+" | Where-Object { $_ })[0]
if (-not $slurmdPod) { throw "No running slurmd pod found for MLflow backup." }
try {
    & python (Join-Path $RepoRoot "scripts\prepare-improved-uncertainty-dashboard.py") `
        --source (Join-Path $RepoRoot "grafana\dashboards\8-run.json") --output $dashboardTemplate
    if ($LASTEXITCODE -ne 0) { throw "Dashboard generation failed." }

    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\backup-rancher-comparison-v2.ps1") `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -OutputPath $snapshot `
        -SnapshotName "carpk-stop-policy-hybrid-controller-9run" -DashboardSource $dashboardTemplate `
        -MlflowPod $slurmdPod -MlflowContainer "slurmd" `
        -MlflowBackendPath "/workspace/mlflow-fallback/backend" `
        -MlflowArtifactsPath "/workspace/mlflow-fallback/artifacts"
    if ($LASTEXITCODE -ne 0) { throw "Nine-run snapshot failed." }

    # The active exporter directory also contains the separately provisioned
    # 7/60/8-run series. Rebuild a canonical, unlabelled nine-job textfile set
    # inside the new snapshot before using it as a restore source.
    $metrics = Join-Path $snapshot "data\energy_metrics"
    $nodeExporter = Join-Path $metrics "node_exporter"
    Remove-Item -LiteralPath $nodeExporter -Recurse -Force
    New-Item -ItemType Directory -Force -Path $nodeExporter | Out-Null
    foreach ($epochFile in Get-ChildItem -LiteralPath $metrics -Filter "epoch_summary_job_*.json" | Sort-Object Name) {
        $jobId = [regex]::Match($epochFile.Name, "(\d+)").Groups[1].Value
        $phaseSummary = Join-Path $metrics "gpu_summary_job_${jobId}_phases.json"
        & python (Join-Path $RepoRoot "slurm\export_job_metrics_prom.py") `
            --summary-json $phaseSummary --output-dir $nodeExporter --aggregate-dir $metrics | Out-Null
        & python (Join-Path $RepoRoot "slurm\export_job_phase_metrics_prom.py") `
            --summary-json $phaseSummary --output-prom (Join-Path $nodeExporter "job_${jobId}_phases.prom") | Out-Null
        & python (Join-Path $RepoRoot "slurm\export_job_epoch_metrics_prom.py") `
            --summary-json $epochFile.FullName --output-prom (Join-Path $nodeExporter "job_${jobId}_epochs.prom") | Out-Null
    }
    & python (Join-Path $RepoRoot "slurm\export_study_metrics_prom.py") `
        --metrics-dir $metrics --output-prom (Join-Path $nodeExporter "study.prom") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Canonical metric regeneration failed." }

    $checksums = Get-ChildItem -LiteralPath $snapshot -Recurse -File |
        Where-Object { $_.FullName -notlike "*\checksums\sha256.txt" } |
        ForEach-Object {
            $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
            $relative = $_.FullName.Substring($snapshot.Length + 1).Replace("\", "/")
            "$($hash.Hash.ToLowerInvariant())  $relative"
        }
    $checksums | Set-Content -LiteralPath (Join-Path $snapshot "checksums\sha256.txt") -Encoding ascii

    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\deploy-rancher-comparison-dashboards.ps1") `
        -Kubeconfig $Kubeconfig -Namespace $Namespace -SevenRunSnapshot $SevenRunSnapshot `
        -SixtyRunSnapshot $SixtyRunSnapshot -EightRunSnapshot $snapshot -EightRunExpectedJobs 9 `
        -EightRunDashboardTemplate $dashboardTemplate -EightRunTitle "9 Run - Hybrid Min30 Eval1"
    if ($LASTEXITCODE -ne 0) { throw "Dashboard deployment failed." }
}
finally {
    Remove-Item -LiteralPath $dashboardTemplate -Force -ErrorAction SilentlyContinue
}

Write-Host "Comparison snapshot: $snapshot" -ForegroundColor Green
Write-Host "Grafana dashboard: 9 Run - Hybrid Min30 Eval1 (same UID as previous 8 Run)" -ForegroundColor Green
