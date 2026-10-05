[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunSpecJson,
    [Parameter(Mandatory = $true)][string]$Runner,
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int]$TimeoutSeconds = 7200
)

$ErrorActionPreference = "Stop"
function Invoke-Kubectl([string[]]$Arguments) {
    $output = & kubectl @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "kubectl failed: kubectl $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

$slurmctldPod = (Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmctld", "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}")) -split '\s+' | Select-Object -First 1
$slurmdPod = (Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd", "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}")) -split '\s+' | Select-Object -First 1
$runtimeImageId = Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", $slurmdPod, "-o", "jsonpath={.status.containerStatuses[?(@.name=='slurmd')].imageID}")
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($RunSpecJson))
$submitCommand = "RUN_SPEC_BASE64='$encoded' BENCHMARK_RUNNER='$Runner' RUNTIME_IMAGE_ID='$runtimeImageId' sbatch /workspace/controller_benchmark/slurm/run-generic-benchmark.slurm"
$submitted = Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc", $submitCommand)
if ($submitted -notmatch 'Submitted batch job\s+(\d+)') { throw "Could not parse SLURM job id: $submitted" }
$jobId = $Matches[1]
Write-Host "Submitted generic benchmark job $jobId ($Runner)." -ForegroundColor Green

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    Start-Sleep 5
    $jobInfo = Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmctldPod, "-c", "slurmctld", "--", "bash", "-lc", "scontrol show job $jobId 2>/dev/null || true")
    $state = if ($jobInfo -match 'JobState=([A-Z_]+)') { $Matches[1] } else { "UNKNOWN" }
    Write-Host "Job ${jobId}: $state"
} while ($state -notmatch '^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)' -and (Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) { throw "Timeout while waiting for generic job $jobId." }
if ($state -ne "COMPLETED") {
    $logs = Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc", "cat /workspace/logs/controller-benchmark-$jobId.out /workspace/logs/controller-benchmark-$jobId.err 2>/dev/null || true")
    throw "Generic job $jobId ended with $state.`n$logs"
}
Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc", "test -s /workspace/energy_metrics/epoch_summary_job_$jobId.json && test -s /workspace/energy_metrics/gpu_summary_job_$jobId.json && test -s /workspace/energy_metrics/node_exporter/job_${jobId}_epochs.prom") | Out-Null
Write-Host "Generic benchmark job $jobId validated." -ForegroundColor Green
