[CmdletBinding()]
param(
    [string]$Manifest = "controller_benchmark/config/benchmark-v2-shared-seed.json",
    [Parameter(Mandatory = $true)][string]$Controller,
    [Parameter(Mandatory = $true)][string]$ControllerId,
    [string]$ControllerParametersJson = "{}",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$SnapshotRoot = "D:\Masterarbeit_Vergleichssicherung\ControllerBenchmarks",
    [string]$BaselineCatalog = "D:\Masterarbeit_Vergleichssicherung\ControllerBaselines\controller-benchmark-v2-shared-seed\catalog.json",
    [switch]$PlanOnly,
    [switch]$RequireAllStages,
    [switch]$RefreshBaselines
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
function Invoke-Native([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}
function Get-NativeText([string]$Command, [string[]]$Arguments) {
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}
if (-not [System.IO.Path]::IsPathRooted($Manifest)) { $Manifest = Join-Path $RepoRoot $Manifest }
if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
if ([System.IO.Path]::GetFileName($BaselineCatalog) -ne "catalog.json") {
    throw "BaselineCatalog must end with catalog.json."
}
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$planDir = Join-Path $RepoRoot "results\controller-benchmark\plans\$stamp-$ControllerId"
$controllerParametersBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ControllerParametersJson))
$planArguments = @("-m", "controller_benchmark.cli", "plan", "--manifest", $Manifest, "--controller", $Controller, "--controller-id", $ControllerId, "--parameters-base64", $controllerParametersBase64, "--output-dir", $planDir)
if (-not $RefreshBaselines -and (Test-Path -LiteralPath $BaselineCatalog)) {
    $planArguments += @("--baseline-catalog", ([System.IO.Path]::GetFullPath($BaselineCatalog)))
}
Invoke-Native "python" $planArguments
$planPath = Join-Path $planDir "run-matrix.json"
$plan = Get-Content $planPath -Raw | ConvertFrom-Json
$plan.runs | Select-Object sequence, benchmark_stage, benchmark_case_id, strategy, controller_mode | Format-Table -AutoSize
Write-Host "Cases: $($plan.planned_cases); candidate runs: $($plan.planned_candidate_runs); Full100 runs: $($plan.planned_baseline_runs); cached Full100: $($plan.baseline_cache.cached_case_ids.Count)" -ForegroundColor Cyan
if ($plan.blocked_stages.Count -gt 0) {
    Write-Host "Blocked benchmark stages:" -ForegroundColor Yellow
    $plan.blocked_stages | Format-Table stage, reason -Wrap
    if ($RequireAllStages) { throw "Full benchmark is blocked because dataset/task extensions are missing." }
}
if ($PlanOnly) { return }

$slurmdPod = (Get-NativeText "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd", "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}")) -split '\s+' | Select-Object -First 1
Write-Host "Reset active benchmark files while preserving versioned historical metrics ..." -ForegroundColor Cyan
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc", "rm -f /workspace/energy_metrics/*.json /workspace/energy_metrics/*.jsonl /workspace/energy_metrics/*.csv /workspace/energy_metrics/*.prom /workspace/logs/* 2>/dev/null || true; find /workspace/energy_metrics/node_exporter -maxdepth 1 -type f \( -name 'job_*.prom' -o -name 'slurm_jobs.prom' \) -delete; mkdir -p /workspace/energy_metrics/node_exporter /workspace/logs")
$mlflowReset = "pid=`$(pgrep -f '^/opt/conda/bin/python .*mlflow server.*--port 5000' | head -1 || true); if [ -n `"`$pid`" ]; then kill `"`$pid`" || true; sleep 2; fi; rm -rf /workspace/mlflow-fallback/backend /workspace/mlflow-fallback/artifacts; mkdir -p /workspace/mlflow-fallback/backend /workspace/mlflow-fallback/artifacts; nohup mlflow server --backend-store-uri sqlite:////workspace/mlflow-fallback/backend/mlflow.db --default-artifact-root /workspace/mlflow-fallback/artifacts --host 0.0.0.0 --port 5000 --allowed-hosts '*' >/workspace/mlflow-fallback/mlflow.log 2>&1 &"
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc", $mlflowReset)
Start-Sleep 5
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deployment/slurmctld")
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/slurmctld", "--timeout=300s")

$needsVedai = @($plan.runs | Where-Object { $_.runner -eq "vedai_yolo" }).Count -gt 0
if ($needsVedai) {
    Write-Host "Prepare versioned VEDAI 512 cache for dataset generalisation ..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "prepare-vedai.ps1") -Kubeconfig $Kubeconfig -Namespace $Namespace
}

$execution = $plan.execution
foreach ($run in $plan.runs) {
    $case = $run.case
    $params = $run.controller_parameters | ConvertTo-Json -Compress
    Write-Host "Run $($run.sequence)/$($plan.runs.Count): $($run.benchmark_case_id) / $($run.strategy)" -ForegroundColor Cyan
    if ($run.runner -eq "carpk_yolo") {
      $submit = @{
        Kubeconfig = $Kubeconfig
        Namespace = $Namespace
        ExperimentName = $execution.experiment_name
        ScenarioName = $run.scenario
        Dataset = "carpk"
        Model = "yolov8"
        ModelVersion = $case.model_version
        Epochs = [int]$execution.max_epochs
        BatchSize = [int]$execution.batch_size
        ImageSize = [int]$execution.image_size
        Patience = [int]$execution.max_epochs
        TrainSize = [int]$case.train_size
        Pretrained = $case.pretrained.ToString().ToLower()
        TrainingSeed = [int]$run.training_seed
        SplitSeed = [int]$execution.split_seed
        CachePolicy = $execution.cache_policy
        TimeoutSeconds = [int]$execution.timeout_seconds
        ControllerMode = $run.controller_mode
        ComparisonStrategy = $run.strategy
        AdaptiveMonitorMetric = "map50_95"
        AdaptiveMinEpochs = 1
        UncertaintyTargetEpoch = [int]$execution.max_epochs
        ControllerPlugin = $run.controller_plugin
        ControllerParametersJson = $params
        ControllerId = $run.controller_id
        BenchmarkVersion = $run.benchmark_version
        BenchmarkRunId = $run.benchmark_run_id
        BenchmarkCaseId = $run.benchmark_case_id
        BenchmarkStage = $run.benchmark_stage
        TaskType = $run.task_type
        QualityMetric = $run.quality_metric
        SkipDependencyInstall = $true
        WaitForCompletion = $true
        ValidateMlflowViaSlurmd = $true
      }
      if ([int]$run.sequence -eq 1) { $submit.ForceWorkspaceSync = $true }
      & (Join-Path $RepoRoot "scripts\submit-rancher-job.ps1") @submit
    } elseif ($run.runner -in @("vedai_yolo", "carpk_task_generalisation")) {
      $runSpec = [ordered]@{
        benchmark_run_id = $run.benchmark_run_id
        benchmark_version = $run.benchmark_version
        benchmark_stage = $run.benchmark_stage
        benchmark_case_id = $run.benchmark_case_id
        runner = $run.runner
        task_type = $run.task_type
        quality_metric = $run.quality_metric
        scenario = $run.scenario
        training_seed = [int]$run.training_seed
        strategy = $run.strategy
        controller_id = $run.controller_id
        controller_mode = $run.controller_mode
        controller_plugin = $run.controller_plugin
        controller_parameters = $run.controller_parameters
        case = $run.case
        execution = $execution
      } | ConvertTo-Json -Depth 20 -Compress
      & (Join-Path $PSScriptRoot "submit-generic-run.ps1") -RunSpecJson $runSpec -Runner $run.runner -Kubeconfig $Kubeconfig -Namespace $Namespace -TimeoutSeconds ([int]$execution.timeout_seconds)
    } else {
      throw "Unsupported runner: $($run.runner)"
    }
    if ($LASTEXITCODE -ne 0) { throw "Benchmark run $($run.sequence) failed." }
    Start-Sleep ([int]$execution.cooldown_seconds)
}

$snapshot = Join-Path $SnapshotRoot "$($plan.benchmark_run_id)"
$dashboard = Join-Path $planDir "controller-benchmark.json"
Invoke-Native "python" @("-m", "controller_benchmark.dashboard.build_dashboard", "--output", $dashboard)
Invoke-Native "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\backup-rancher-comparison-v2.ps1"), "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-OutputPath", $snapshot, "-SnapshotName", $plan.benchmark_run_id, "-DashboardSource", $dashboard, "-MlflowPod", $slurmdPod, "-MlflowContainer", "slurmd", "-MlflowBackendPath", "/workspace/mlflow-fallback/backend", "-MlflowArtifactsPath", "/workspace/mlflow-fallback/artifacts")
New-Item -ItemType Directory -Force (Join-Path $snapshot "controller_benchmark") | Out-Null
Copy-Item $planPath (Join-Path $snapshot "controller_benchmark\run-matrix.json")
Copy-Item $dashboard (Join-Path $snapshot "controller_benchmark\controller-benchmark.json")
if ([int]$plan.planned_baseline_runs -gt 0) {
    Invoke-Native "python" @("-m", "controller_benchmark.baseline_cache", "--manifest", $Manifest, "--snapshot", $snapshot, "--output-dir", (Split-Path -Parent $BaselineCatalog))
}
if (Test-Path -LiteralPath $BaselineCatalog) {
    Copy-Item -Recurse -Force (Split-Path -Parent $BaselineCatalog) (Join-Path $snapshot "controller_benchmark\baseline-cache")
}
Invoke-Native "python" @("-m", "controller_benchmark.analyze", "--snapshot", $snapshot, "--plan", $planPath)
Invoke-Native "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "deploy-dashboard.ps1"), "-Kubeconfig", $Kubeconfig, "-Namespace", $Namespace, "-Snapshot", $snapshot, "-Dashboard", $dashboard)
Write-Host "Benchmark complete: $snapshot" -ForegroundColor Green
