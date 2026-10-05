[CmdletBinding()]
param(
    [string]$Manifest = "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    [string]$Controllers = "controller_benchmark/config/rapec-v3-v8-controllers.json",
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

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$stageRoot = Join-Path $RepoRoot "results\controller-benchmark\nine-dataset-submissions\$stamp-rapec-v3-v8"
$planDir = Join-Path $stageRoot "plan"
New-Item -ItemType Directory -Force $stageRoot | Out-Null

$orchestratorPod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=benchmark-orchestrator -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $orchestratorPod) {
    throw "benchmark-orchestrator is not running. Run controller_benchmark/scripts/install-server-orchestrator.ps1 first."
}

& python -m controller_benchmark.campaign `
    --manifest $Manifest `
    --controllers $Controllers `
    --output-dir $planDir
if ($LASTEXITCODE -ne 0) { throw "Nine-dataset campaign planning failed." }

$planPath = Join-Path $planDir "run-matrix.json"
$plan = Get-Content $planPath -Raw | ConvertFrom-Json
$plan.runs |
    Select-Object sequence, role, benchmark_stage, benchmark_case_id, strategy |
    Format-Table -AutoSize
Write-Host "Planned jobs: $($plan.runs.Count) (Full100=$($plan.planned_baseline_runs), candidates=$($plan.planned_candidate_runs))"
Write-Host "All Full100 baselines are fresh. No existing baseline catalog is used."
if ($PlanOnly) {
    Write-Host "Plan only: nothing was uploaded or queued." -ForegroundColor Green
    return
}

$runId = $plan.benchmark_run_id
$runStage = Join-Path $stageRoot $runId
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
Copy-Item -Force $Controllers (Join-Path $runStage "controllers.json")
Copy-Item -Force $planPath (Join-Path $runStage "run-matrix.json")
& python -m controller_benchmark.dashboard.build_nine_dataset_dashboard --output-dir $runStage
if ($LASTEXITCODE -ne 0) { throw "Nine-dataset dashboard generation failed." }

$controllerHashes = [ordered]@{
    captured_at = (Get-Date).ToUniversalTime().ToString("o")
    rapec_v3 = (Get-FileHash "controller_benchmark/controllers/risk_aware_predictive_energy.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    rapec_v8 = (Get-FileHash "controller_benchmark/controllers/bayesian_guarded_rapec_v3.py" -Algorithm SHA256).Hash.ToLowerInvariant()
    note = "The campaign records the unchanged controller source hashes."
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText(
    (Join-Path $runStage "controller-source-hashes.json"),
    ($controllerHashes | ConvertTo-Json),
    $utf8
)
$preservation = [ordered]@{
    benchmark_version = $plan.benchmark_version
    benchmark_run_id = $runId
    uses_existing_baselines = $false
    overwrites_previous_results = $false
    local_result_root = $runStage
    cluster_result_root = "/workspace-cache/controller-benchmarks/$runId"
    cluster_baseline_root = "/workspace-cache/controller-baselines/$($plan.benchmark_version)"
}
[IO.File]::WriteAllText(
    (Join-Path $runStage "preservation.json"),
    ($preservation | ConvertTo-Json),
    $utf8
)

$clusterRunDir = "/workspace-cache/controller-benchmarks/$runId"
$request = @{
    schema_version = 1
    benchmark_run_id = $runId
    run_dir = $clusterRunDir
    submitted_at = (Get-Date).ToUniversalTime().ToString("o")
    submitted_from = $env:COMPUTERNAME
}
[IO.File]::WriteAllText((Join-Path $runStage "request.json"), ($request | ConvertTo-Json), $utf8)

& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- mkdir -p $clusterRunDir
if ($LASTEXITCODE -ne 0) { throw "Could not create the server campaign directory." }
$parent = Split-Path -Parent $runStage
$leaf = Split-Path -Leaf $runStage
Push-Location $parent
try {
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "$leaf/." "${orchestratorPod}:$clusterRunDir"
}
finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw "Could not upload the nine-dataset campaign bundle." }

$prepare = @"
set -e
mkdir -p '$clusterRunDir/workspace/rotationally-invariant-cnns/data' /workspace-cache/controller-benchmark-queue/pending
ln -sfn /workspace-cache/carpk '$clusterRunDir/workspace/rotationally-invariant-cnns/data/carpk'
find '$clusterRunDir/workspace' -type f \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} +
cp '$clusterRunDir/request.json' '/workspace-cache/controller-benchmark-queue/pending/$runId.json.tmp'
mv '/workspace-cache/controller-benchmark-queue/pending/$runId.json.tmp' '/workspace-cache/controller-benchmark-queue/pending/$runId.json'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($prepare))
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- bash -lc "echo '$encoded' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "Could not enqueue the nine-dataset campaign." }

Write-Host "Fresh 9-dataset campaign queued on Rancher: $runId" -ForegroundColor Green
Write-Host "Previous runs, snapshots, baseline catalogs, and dashboards were not changed." -ForegroundColor Green
Write-Host "Your computer may now be turned off. Rancher continues independently." -ForegroundColor Green
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 -RunId '$runId'"
Write-Host "Live: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-controller-benchmark-progress.ps1 -RunId '$runId' -ShowLogs"
