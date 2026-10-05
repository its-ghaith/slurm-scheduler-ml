[CmdletBinding()]
param(
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    [string]$Controllers = "controller_benchmark/config/live-shadow-controllers.json",
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int]$ExpectedCases = 9,
    [string]$BenchmarkVersion = "controller-live-shadow-nine-dataset-v1",
    [string]$DashboardFile = "live-shadow-nine-dataset.json",
    [string]$ExperimentName = "controller-live-shadow-development",
    [string]$DashboardTitle = "",
    [string]$DashboardUid = "",
    [string]$DashboardBuilder = "controller_benchmark.dashboard.build_live_shadow_dashboard",
    [string]$SubmissionLabel = "live-shadow-dev",
    [string]$PretrainingJobId = "",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Manifest)) { $Manifest = Join-Path $RepoRoot $Manifest }
if (-not [IO.Path]::IsPathRooted($Controllers)) { $Controllers = Join-Path $RepoRoot $Controllers }
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
if (-not (Test-Path $Controllers)) { throw "Shadow controller configuration not found: $Controllers" }

function Get-RunningPodName([string]$LabelSelector) {
    $podJson = & kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l $LabelSelector -o json
    if ($LASTEXITCODE -ne 0) { throw "Could not list pods matching $LabelSelector." }
    $podDocument = $podJson | ConvertFrom-Json
    return @(
        $podDocument.items |
            Where-Object { $_.status.phase -eq "Running" } |
            ForEach-Object { $_.metadata.name }
    ) | Select-Object -First 1
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$stageRoot = Join-Path $RepoRoot "results\controller-benchmark\async-submissions\$stamp-$SubmissionLabel"
$planDir = Join-Path $stageRoot "plan"
New-Item -ItemType Directory -Force $stageRoot | Out-Null

$planArguments = @(
    "-m", "controller_benchmark.live_shadow_plan",
    "--manifest", $Manifest,
    "--controllers", $Controllers,
    "--output-dir", $planDir,
    "--expected-cases", "$ExpectedCases",
    "--benchmark-version", $BenchmarkVersion,
    "--dashboard-file", $DashboardFile,
    "--experiment-name", $ExperimentName
)
if (-not [string]::IsNullOrWhiteSpace($RunId)) { $planArguments += @("--run-id", $RunId) }
if (-not [string]::IsNullOrWhiteSpace($PretrainingJobId)) {
    if ($PretrainingJobId -notmatch '^\d+$') { throw "Invalid SLURM pretraining job id: $PretrainingJobId" }
    $planArguments += @("--pretraining-job-id", $PretrainingJobId)
}
& python @planArguments
if ($LASTEXITCODE -ne 0) { throw "Live-shadow planning failed." }

$planPath = Join-Path $planDir "run-matrix.json"
$plan = Get-Content $planPath -Raw | ConvertFrom-Json
$plan.runs | Select-Object sequence, benchmark_stage, benchmark_case_id, strategy, training_seed | Format-Table -AutoSize
Write-Host "Planned real training jobs: $($plan.runs.Count)" -ForegroundColor Cyan
Write-Host "Live shadow controllers per job: $($plan.controllers.Count)" -ForegroundColor Cyan
Write-Host "Training seeds: $((@($plan.runs.training_seed | Sort-Object -Unique) -join ', '))" -ForegroundColor Cyan
if ($plan.runs.Count -ne $ExpectedCases) { throw "Expected exactly $ExpectedCases Full100 development jobs." }
if (@($plan.runs.training_seed | Sort-Object -Unique).Count -ne 1 -or $plan.runs[0].training_seed -ne 0) {
    throw "Development shadow benchmark must use only training seed 0."
}
if ($PlanOnly) {
    Write-Host "Plan only: no Rancher job was queued." -ForegroundColor Green
    return
}

$orchestratorPod = Get-RunningPodName "app=benchmark-orchestrator"
if (-not $orchestratorPod) {
    throw "benchmark-orchestrator is not running. Run controller_benchmark/scripts/install-server-orchestrator.ps1 first."
}

$effectiveRunId = $plan.benchmark_run_id
$runStage = Join-Path $stageRoot $effectiveRunId
$workspace = Join-Path $runStage "workspace"
New-Item -ItemType Directory -Force $workspace | Out-Null
New-Item -ItemType Directory -Force `
    (Join-Path $workspace "rotationally-invariant-cnns"), `
    (Join-Path $workspace "slurm"), `
    (Join-Path $workspace "containers"), `
    (Join-Path $workspace "controller_benchmark") | Out-Null

Copy-Item -Recurse -Force "rotationally-invariant-cnns/src" (Join-Path $workspace "rotationally-invariant-cnns")
foreach ($file in @("rotationally-invariant-cnns/pyproject.toml", "rotationally-invariant-cnns/uv.lock")) {
    if (Test-Path $file) { Copy-Item -Force $file (Join-Path $workspace "rotationally-invariant-cnns") }
}
Copy-Item -Recurse -Force "slurm/*" (Join-Path $workspace "slurm")
Copy-Item -Force "containers/runtime-requirements.txt" (Join-Path $workspace "containers")
$benchmarkWorkspace = Join-Path $workspace "controller_benchmark"
Get-ChildItem "controller_benchmark" -File -Filter "*.py" | Copy-Item -Destination $benchmarkWorkspace -Force
foreach ($directory in @("config", "controllers", "dashboard", "runners", "server", "slurm")) {
    Copy-Item -Recurse -Force (Join-Path "controller_benchmark" $directory) $benchmarkWorkspace
}

Copy-Item -Force $Manifest (Join-Path $runStage "manifest.json")
Copy-Item -Force $Controllers (Join-Path $runStage "live-shadow-controllers.json")
Copy-Item -Force $planPath (Join-Path $runStage "run-matrix.json")
$dashboardArguments = @(
    "-m", $DashboardBuilder,
    "--output", (Join-Path $runStage $DashboardFile)
)
if (-not [string]::IsNullOrWhiteSpace($DashboardTitle)) { $dashboardArguments += @("--title", $DashboardTitle) }
if (-not [string]::IsNullOrWhiteSpace($DashboardUid)) { $dashboardArguments += @("--uid", $DashboardUid) }
& python @dashboardArguments
if ($LASTEXITCODE -ne 0) { throw "Live-shadow dashboard generation failed." }

$clusterRunDir = "/workspace-cache/controller-benchmarks/$effectiveRunId"
$request = @{
    schema_version = 1
    benchmark_run_id = $effectiveRunId
    run_dir = $clusterRunDir
    submitted_at = (Get-Date).ToUniversalTime().ToString("o")
    submitted_from = $env:COMPUTERNAME
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText(
    (Join-Path $runStage "request.json"),
    ($request | ConvertTo-Json),
    $utf8
)

& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- mkdir -p $clusterRunDir
if ($LASTEXITCODE -ne 0) { throw "Could not create the server run directory." }
$parent = Split-Path -Parent $runStage
$leaf = Split-Path -Leaf $runStage
Push-Location $parent
try {
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "$leaf/." "${orchestratorPod}:$clusterRunDir"
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw "Could not upload the live-shadow bundle." }

$prepare = @"
set -e
mkdir -p '$clusterRunDir/workspace/rotationally-invariant-cnns/data' /workspace-cache/controller-benchmark-queue/pending
ln -sfn /workspace-cache/carpk '$clusterRunDir/workspace/rotationally-invariant-cnns/data/carpk'
find '$clusterRunDir/workspace' -type f \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} +
cp '$clusterRunDir/request.json' '/workspace-cache/controller-benchmark-queue/pending/$effectiveRunId.json.tmp'
mv '/workspace-cache/controller-benchmark-queue/pending/$effectiveRunId.json.tmp' '/workspace-cache/controller-benchmark-queue/pending/$effectiveRunId.json'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($prepare))
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- bash -lc "echo '$encoded' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "Could not enqueue the live-shadow benchmark." }

Write-Host "Live-shadow benchmark queued on Rancher: $effectiveRunId" -ForegroundColor Green
Write-Host "$ExpectedCases Full100 jobs will run with seed 0; $($plan.controllers.Count) controllers observe each job live." -ForegroundColor Green
Write-Host "Your computer may now be turned off. Rancher continues independently." -ForegroundColor Green
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 -RunId '$effectiveRunId'"
