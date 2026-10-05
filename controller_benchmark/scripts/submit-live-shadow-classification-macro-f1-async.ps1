[CmdletBinding()]
param(
    [string]$Manifest = "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json",
    [string]$Controllers = "controller_benchmark/config/live-shadow-controllers.json",
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
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

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$stageRoot = Join-Path $RepoRoot "results\controller-benchmark\async-submissions\$stamp-live-shadow-classification-macro-f1"
$planDir = Join-Path $stageRoot "plan"
New-Item -ItemType Directory -Force $stageRoot | Out-Null
$dashboardFile = "live-shadow-classification-macro-f1.json"
$benchmarkVersion = "controller-live-shadow-classification-macro-f1-v1"

$planArguments = @(
    "-m", "controller_benchmark.live_shadow_plan",
    "--manifest", $Manifest,
    "--controllers", $Controllers,
    "--output-dir", $planDir,
    "--expected-cases", "3",
    "--benchmark-version", $benchmarkVersion,
    "--dashboard-file", $dashboardFile,
    "--experiment-name", "controller-live-shadow-classification-macro-f1"
)
if (-not [string]::IsNullOrWhiteSpace($RunId)) { $planArguments += @("--run-id", $RunId) }
& python @planArguments
if ($LASTEXITCODE -ne 0) { throw "Macro-F1 live-shadow planning failed." }

$planPath = Join-Path $planDir "run-matrix.json"
$plan = Get-Content $planPath -Raw | ConvertFrom-Json
$plan.runs | Select-Object sequence, benchmark_case_id, strategy, training_seed | Format-Table -AutoSize
if ($plan.runs.Count -ne 3) { throw "Expected exactly 3 classification jobs." }
if (@($plan.runs | Where-Object { $_.task_type -ne "image_classification" }).Count -ne 0) {
    throw "The Macro-F1 campaign may contain only image-classification jobs."
}
if (@($plan.runs | Where-Object { $_.case.primary_metric -ne "macro_f1" }).Count -ne 0) {
    throw "Every classification case must use primary_metric=macro_f1."
}
if (@($plan.runs.training_seed | Sort-Object -Unique).Count -ne 1 -or $plan.runs[0].training_seed -ne 0) {
    throw "The development benchmark must use only training seed 0."
}
Write-Host "Planned jobs: 3 classification Full100 runs with Macro-F1 and seed 0." -ForegroundColor Cyan
if ($PlanOnly) {
    Write-Host "Plan only: no Rancher job was queued." -ForegroundColor Green
    return
}

$orchestratorPod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=benchmark-orchestrator -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $orchestratorPod) {
    throw "benchmark-orchestrator is not running."
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
& python -m controller_benchmark.dashboard.build_live_shadow_dashboard `
    --output (Join-Path $runStage $dashboardFile) `
    --title "Live Shadow Controllers - Classification Macro-F1" `
    --uid "controller-live-shadow-classification-f1" `
    --tag "controller-benchmark" --tag "live-shadow" --tag "classification" --tag "macro-f1"
if ($LASTEXITCODE -ne 0) { throw "Macro-F1 dashboard generation failed." }

$clusterRunDir = "/workspace-cache/controller-benchmarks/$effectiveRunId"
$request = @{
    schema_version = 1
    benchmark_run_id = $effectiveRunId
    run_dir = $clusterRunDir
    submitted_at = (Get-Date).ToUniversalTime().ToString("o")
    submitted_from = $env:COMPUTERNAME
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $runStage "request.json"), ($request | ConvertTo-Json), $utf8)

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
if ($LASTEXITCODE -ne 0) { throw "Could not upload the Macro-F1 benchmark bundle." }

$prepare = @"
set -e
mkdir -p '$clusterRunDir/workspace/rotationally-invariant-cnns/data' /workspace-cache/controller-benchmark-queue/pending
find '$clusterRunDir/workspace' -type f \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} +
cp '$clusterRunDir/request.json' '/workspace-cache/controller-benchmark-queue/pending/$effectiveRunId.json.tmp'
mv '/workspace-cache/controller-benchmark-queue/pending/$effectiveRunId.json.tmp' '/workspace-cache/controller-benchmark-queue/pending/$effectiveRunId.json'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($prepare))
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- bash -lc "echo '$encoded' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "Could not enqueue the Macro-F1 benchmark." }

Write-Host "Macro-F1 classification benchmark queued on Rancher: $effectiveRunId" -ForegroundColor Green
Write-Host "Your computer may now be turned off. Rancher continues independently." -ForegroundColor Green
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 -RunId '$effectiveRunId'"
