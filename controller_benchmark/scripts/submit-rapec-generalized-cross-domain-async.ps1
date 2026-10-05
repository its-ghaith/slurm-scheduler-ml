[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [ValidateSet("Standard", "Hard", "Expensive", "Hardest")]
    [string]$Suite = "Standard",
    [switch]$QueueAfterPretraining,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
$generatedRoot = Join-Path $RepoRoot "results\controller-benchmark\generated-manifests"
$isHard = $Suite -eq "Hard"
$isExpensive = $Suite -eq "Expensive"
$isHardest = $Suite -eq "Hardest"
$benchmarkVersion = if ($isHardest) { "rapec-generalized-nonvision-hardest-v1" } elseif ($isExpensive) { "rapec-generalized-nonvision-expensive-v1" } elseif ($isHard) { "rapec-generalized-nonvision-hard-v1" } else { "rapec-generalized-nonvision-v2" }
$checkpointRoot = if ($isHardest) { "/workspace-cache/controller-pretrained-cross-domain-hardest-v1" } elseif ($isExpensive) { "/workspace-cache/controller-pretrained-cross-domain-expensive-v1" } elseif ($isHard) { "/workspace-cache/controller-pretrained-cross-domain-hard-v1" } else { "/workspace-cache/controller-pretrained-cross-domain" }
$manifest = Join-Path $generatedRoot "$benchmarkVersion.json"
New-Item -ItemType Directory -Force $generatedRoot | Out-Null

$manifestArguments = @(
    "-m", "controller_benchmark.generalized_manifest",
    "--vision-source", "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    "--classification-source", "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json",
    "--pretrained-source", "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json",
    "--output", $manifest
)
if ($isHard) { $manifestArguments += @("--suite", "hard") }
if ($isExpensive) { $manifestArguments += @("--suite", "expensive") }
if ($isHardest) { $manifestArguments += @("--suite", "hardest") }
& python @manifestArguments
if ($LASTEXITCODE -ne 0) { throw "Cross-domain manifest generation failed." }

$arguments = @{
    Manifest = $manifest
    Controllers = "controller_benchmark/config/generalized-live-shadow-controllers.json"
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    ExpectedCases = if ($isExpensive -or $isHardest) { 30 } else { 48 }
    BenchmarkVersion = $benchmarkVersion
    DashboardFile = "$benchmarkVersion.json"
    ExperimentName = $benchmarkVersion
    DashboardTitle = if ($isHardest) { "RAPEC-G Hardest Practical Non-Vision Generalisation" } elseif ($isExpensive) { "RAPEC-G Expensive Non-Vision Generalisation" } elseif ($isHard) { "RAPEC-G Hard Non-Vision Generalisation" } else { "RAPEC-G Non-Vision Generalisation" }
    DashboardUid = $benchmarkVersion
    SubmissionLabel = $benchmarkVersion
}
$pretrainingJobId = ""
if (-not [string]::IsNullOrWhiteSpace($RunId)) { $arguments.RunId = $RunId }
if ($PlanOnly) { $arguments.PlanOnly = $true }

if (-not $PlanOnly) {
    if (-not [IO.Path]::IsPathRooted($Kubeconfig)) {
        $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
    }
    $pod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
    if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "No running slurmd pod found." }
    $checkpointCount = (& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- bash -lc "find '$checkpointRoot' -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l").Trim()
    $expectedCheckpoints = if ($isExpensive -or $isHardest) { 15 } else { 24 }
    if ($LASTEXITCODE -ne 0 -or [int]$checkpointCount -lt $expectedCheckpoints) {
        if (-not $QueueAfterPretraining) {
            throw "Only $checkpointCount/$expectedCheckpoints $Suite source-domain checkpoints are available. Run prepare-cross-domain-pretrained-models-rancher.ps1 -Suite $Suite first, or use -QueueAfterPretraining while that job is active."
        }
        $slurmctldPod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmctld -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
        if ($LASTEXITCODE -ne 0 -or -not $slurmctldPod) { throw "No running slurmctld pod found." }
        $activePretrainingOutput = @(& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $slurmctldPod -c slurmctld -- squeue -h -n rapec-pretrain -t PENDING,RUNNING -o "%A")
        $activePretraining = @($activePretrainingOutput | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+$' })
        if ($LASTEXITCODE -ne 0 -or $activePretraining.Count -eq 0) {
            throw "Only $checkpointCount/$expectedCheckpoints checkpoints are available and no active rapec-pretrain job was found."
        }
        if ($activePretraining.Count -ne 1) {
            throw "Expected one active $Suite pretraining job, found: $($activePretraining -join ', ')."
        }
        $pretrainingJobId = $activePretraining[0]
        $arguments.PretrainingJobId = $pretrainingJobId
        Write-Host "Only $checkpointCount/$expectedCheckpoints checkpoints exist; Rancher orchestrator will wait for successful pretraining job $pretrainingJobId." -ForegroundColor Yellow
    }
}

& (Join-Path $PSScriptRoot "submit-live-shadow-nine-dataset-async.ps1") @arguments
if ($LASTEXITCODE -ne 0) { throw "Cross-domain live-shadow submission failed." }
