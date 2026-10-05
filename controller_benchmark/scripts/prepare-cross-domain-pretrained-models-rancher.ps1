[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int]$PretrainingCheckpoints = 20,
    [ValidateSet("Standard", "Hard", "Expensive", "Hardest")]
    [string]$Suite = "Standard"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}
if ($PretrainingCheckpoints -lt 1) { throw "PretrainingCheckpoints must be positive." }

$generatedRoot = Join-Path $RepoRoot "results\controller-benchmark\generated-manifests"
$isHard = $Suite -eq "Hard"
$isExpensive = $Suite -eq "Expensive"
$isHardest = $Suite -eq "Hardest"
$benchmarkVersion = if ($isHardest) { "rapec-generalized-nonvision-hardest-v1" } elseif ($isExpensive) { "rapec-generalized-nonvision-expensive-v1" } elseif ($isHard) { "rapec-generalized-nonvision-hard-v1" } else { "rapec-generalized-nonvision-v2" }
$checkpointRoot = if ($isHardest) { "/workspace-cache/controller-pretrained-cross-domain-hardest-v1" } elseif ($isExpensive) { "/workspace-cache/controller-pretrained-cross-domain-expensive-v1" } elseif ($isHard) { "/workspace-cache/controller-pretrained-cross-domain-hard-v1" } else { "/workspace-cache/controller-pretrained-cross-domain" }
$statusPath = if ($isHardest) { "/workspace-cache/controller-pretraining-status/hardest-current.out" } elseif ($isExpensive) { "/workspace-cache/controller-pretraining-status/expensive-current.out" } elseif ($isHard) { "/workspace-cache/controller-pretraining-status/hard-current.out" } else { "/workspace-cache/controller-pretraining-status/current.out" }
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
if ($LASTEXITCODE -ne 0) { throw "Could not generate the non-vision manifest." }

$pod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "No running slurmd pod found." }
$slurmctldPod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmctld -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $slurmctldPod) { throw "No running slurmctld pod found." }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$remote = "/workspace-cache/controller-tools/$benchmarkVersion-pretraining-$stamp"
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- mkdir -p $remote/workspace $remote/logs
if ($LASTEXITCODE -ne 0) { throw "Could not create the shared pretraining workspace." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace cp controller_benchmark "${pod}:$remote/workspace/controller_benchmark" -c slurmd
if ($LASTEXITCODE -ne 0) { throw "Could not upload controller_benchmark." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace cp slurm "${pod}:$remote/workspace/slurm" -c slurmd
if ($LASTEXITCODE -ne 0) { throw "Could not upload SLURM energy helpers." }
Push-Location $generatedRoot
try {
    $manifestLeaf = Split-Path -Leaf $manifest
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp $manifestLeaf "${pod}:$remote/manifest.json" -c slurmd
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw "Could not upload the generated manifest." }

$submitScript = "/workspace/rapec-generalized-pretraining-$stamp.slurm"
Push-Location (Join-Path $RepoRoot "controller_benchmark\slurm")
try {
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "prepare-cross-domain-pretrained.slurm" "${slurmctldPod}:$submitScript" -c slurmctld
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw "Could not upload the pretraining SLURM script to slurmctld." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $slurmctldPod -c slurmctld -- sed -i 's/\r$//' $submitScript
if ($LASTEXITCODE -ne 0) { throw "Could not normalize the pretraining SLURM script." }

$submit = "BENCHMARK_WORKSPACE='$remote/workspace' BENCHMARK_MANIFEST='$remote/manifest.json' PRETRAINING_OUTPUT_ROOT='$checkpointRoot' PRETRAINING_CHECKPOINTS='$PretrainingCheckpoints' PRETRAINING_STATUS_PATH='$statusPath' sbatch --parsable '$submitScript'"
$jobRaw = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $slurmctldPod -c slurmctld -- bash -lc $submit
$jobText = (@($jobRaw) -join "`n").Trim()
if ($LASTEXITCODE -ne 0 -or $jobText -notmatch '^\d+') { throw "Could not submit the pretraining job: $jobText" }
$jobId = ($jobText -split '\s+')[0]

Write-Host "$Suite cross-domain source pretraining queued as SLURM job $jobId." -ForegroundColor Green
Write-Host "The job continues on Rancher if this computer is turned off." -ForegroundColor Green
Write-Host "Status: kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- squeue -j $jobId" -ForegroundColor Cyan
Write-Host "Logs: /workspace/logs/pretrain-cross-domain-$jobId.out" -ForegroundColor Cyan
