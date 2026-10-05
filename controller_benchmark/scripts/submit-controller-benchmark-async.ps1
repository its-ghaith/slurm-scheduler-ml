[CmdletBinding()]
param(
    [string]$Manifest = "controller_benchmark/config/benchmark-v2-shared-seed.json",
    [Parameter(Mandatory = $true)][string]$Controller,
    [Parameter(Mandatory = $true)][string]$ControllerId,
    [string]$ControllerParametersJson = "{}",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$BaselineCatalog = "D:\Masterarbeit_Vergleichssicherung\ControllerBaselines\controller-benchmark-v2-shared-seed\catalog.json",
    [switch]$RefreshBaselines,
    [switch]$RequireAllStages,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Manifest)) { $Manifest = Join-Path $RepoRoot $Manifest }
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$stageRoot = Join-Path $RepoRoot "results\controller-benchmark\async-submissions\$stamp-$ControllerId"
$planDir = Join-Path $stageRoot "plan"
New-Item -ItemType Directory -Force $stageRoot | Out-Null

$orchestratorPod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=benchmark-orchestrator -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $orchestratorPod) { throw "benchmark-orchestrator is not running. Run controller_benchmark/scripts/install-server-orchestrator.ps1 first." }

$manifestDocument = Get-Content $Manifest -Raw | ConvertFrom-Json
$EffectiveBaselineCatalog = $BaselineCatalog
$serverBaselineDir = "/workspace-cache/controller-baselines/$($manifestDocument.benchmark_version)"
if (-not $RefreshBaselines -and -not $PlanOnly) {
    $serverCatalogStatus = (& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- bash -lc "test -f '$serverBaselineDir/catalog.json' && echo exists || echo missing")
    if ($LASTEXITCODE -ne 0) { throw "Could not check the persistent server baseline catalog." }
    if (($serverCatalogStatus | Select-Object -Last 1) -eq "exists") {
        $downloadedBaseline = Join-Path $stageRoot "server-baseline-cache"
        New-Item -ItemType Directory -Force $downloadedBaseline | Out-Null
        Push-Location $stageRoot
        try { & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${orchestratorPod}:$serverBaselineDir/." "server-baseline-cache" } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "Could not download the persistent server baseline catalog." }
        $EffectiveBaselineCatalog = Join-Path $downloadedBaseline "catalog.json"
        Write-Host "Using persistent Rancher baseline catalog: $serverBaselineDir"
    } elseif (Test-Path $BaselineCatalog) {
        $localBaselineRoot = Split-Path -Parent ([IO.Path]::GetFullPath($BaselineCatalog))
        & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- mkdir -p $serverBaselineDir
        if ($LASTEXITCODE -ne 0) { throw "Could not initialize the persistent server baseline directory." }
        $localBaselineParent = Split-Path -Parent $localBaselineRoot
        $localBaselineLeaf = Split-Path -Leaf $localBaselineRoot
        Push-Location $localBaselineParent
        try { & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "$localBaselineLeaf/." "${orchestratorPod}:$serverBaselineDir" } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "Could not upload the initial baseline catalog to Rancher." }
        Write-Host "Initialized persistent Rancher baseline catalog: $serverBaselineDir"
    }
}

$parametersBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ControllerParametersJson))
$arguments = @("-m", "controller_benchmark.cli", "plan", "--manifest", $Manifest, "--controller", $Controller, "--controller-id", $ControllerId, "--parameters-base64", $parametersBase64, "--output-dir", $planDir)
if (-not $RefreshBaselines -and (Test-Path $EffectiveBaselineCatalog)) { $arguments += @("--baseline-catalog", ([IO.Path]::GetFullPath($EffectiveBaselineCatalog))) }
& python @arguments
if ($LASTEXITCODE -ne 0) { throw "Benchmark planning failed." }
$planPath = Join-Path $planDir "run-matrix.json"
$plan = Get-Content $planPath -Raw | ConvertFrom-Json
if ($RequireAllStages -and $plan.blocked_stages.Count) { throw "One or more benchmark stages are blocked." }
$plan.runs | Select-Object sequence, role, benchmark_stage, benchmark_case_id, strategy | Format-Table -AutoSize
Write-Host "Planned jobs: $($plan.runs.Count) (candidate=$($plan.planned_candidate_runs), Full100=$($plan.planned_baseline_runs))"
if ($PlanOnly) {
    Write-Host "Plan only: nothing was uploaded or queued." -ForegroundColor Green
    return
}
$runId = $plan.benchmark_run_id
$runStage = Join-Path $stageRoot $runId
$workspace = Join-Path $runStage "workspace"
New-Item -ItemType Directory -Force $workspace | Out-Null

New-Item -ItemType Directory -Force (Join-Path $workspace "rotationally-invariant-cnns"), (Join-Path $workspace "slurm"), (Join-Path $workspace "containers"), (Join-Path $workspace "controller_benchmark") | Out-Null
Copy-Item -Recurse -Force "rotationally-invariant-cnns/src" (Join-Path $workspace "rotationally-invariant-cnns")
foreach ($file in @("rotationally-invariant-cnns/pyproject.toml", "rotationally-invariant-cnns/uv.lock")) { if (Test-Path $file) { Copy-Item -Force $file (Join-Path $workspace "rotationally-invariant-cnns") } }
Copy-Item -Recurse -Force "slurm/*" (Join-Path $workspace "slurm")
Copy-Item -Force "containers/runtime-requirements.txt" (Join-Path $workspace "containers")
$benchmarkWorkspace = Join-Path $workspace "controller_benchmark"
Get-ChildItem "controller_benchmark" -File -Filter "*.py" | Copy-Item -Destination $benchmarkWorkspace -Force
foreach ($directory in @("config", "controllers", "dashboard", "runners", "server", "slurm")) {
    Copy-Item -Recurse -Force (Join-Path "controller_benchmark" $directory) $benchmarkWorkspace
}
Copy-Item -Force $Manifest (Join-Path $runStage "manifest.json")
& python -m controller_benchmark.dashboard.build_dashboard --output (Join-Path $runStage "controller-benchmark.json")
if ($LASTEXITCODE -ne 0) { throw "Dashboard generation failed." }

$clusterRunDir = "/workspace-cache/controller-benchmarks/$runId"
if ($plan.baseline_cache.catalog_path) {
    $catalogRoot = Split-Path -Parent $EffectiveBaselineCatalog
    $runBaselineRoot = Join-Path $runStage "baseline-cache"
    New-Item -ItemType Directory -Force $runBaselineRoot | Out-Null
    Copy-Item -Recurse -Force (Join-Path $catalogRoot "*") $runBaselineRoot
    $plan.baseline_cache.catalog_path = "$clusterRunDir/baseline-cache/catalog.json"
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $runStage "run-matrix.json"), ($plan | ConvertTo-Json -Depth 30), $utf8)
$request = @{schema_version=1; benchmark_run_id=$runId; run_dir=$clusterRunDir; submitted_at=(Get-Date).ToUniversalTime().ToString("o"); submitted_from=$env:COMPUTERNAME}
[IO.File]::WriteAllText((Join-Path $runStage "request.json"), ($request | ConvertTo-Json), $utf8)

& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $orchestratorPod -- mkdir -p $clusterRunDir
if ($LASTEXITCODE -ne 0) { throw "Could not create server run directory." }
$parent = Split-Path -Parent $runStage
$leaf = Split-Path -Leaf $runStage
Push-Location $parent
try { & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "$leaf/." "${orchestratorPod}:$clusterRunDir" } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "Could not upload benchmark bundle." }
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
if ($LASTEXITCODE -ne 0) { throw "Could not enqueue benchmark request." }
Write-Host "Benchmark queued on Rancher: $runId" -ForegroundColor Green
Write-Host "Your computer may now be turned off. Rancher continues independently." -ForegroundColor Green
Write-Host "Status: powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\get-controller-benchmark-status.ps1 -RunId '$runId'"
