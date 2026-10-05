[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$OutputDir = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int]$HorizonCheckpoints = 5,
    [int]$MinimumCheckpoints = 10,
    [double]$UsefulGain = 0.002,
    [int]$Patience = 3
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}
$pod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "No running slurmd pod found." }

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "results\controller-benchmark\lcpfn-comparisons\$RunId\$stamp"
} elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}
New-Item -ItemType Directory -Force $OutputDir | Out-Null

$remoteRun = "/workspace-cache/controller-benchmarks/$RunId"
$remoteSnapshot = "$remoteRun/snapshot"
$remoteTool = "/tmp/lcpfn-comparison-$RunId-$stamp"
$remoteOutput = "$remoteRun/lcpfn-comparisons/$stamp"
$remoteEnvironment = "/workspace-cache/controller-tools/lcpfn-0.1.3-python310"

& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- test -d $remoteSnapshot
if ($LASTEXITCODE -ne 0) { throw "Completed Rancher snapshot not found: $remoteSnapshot" }
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- mkdir -p $remoteTool $remoteOutput
if ($LASTEXITCODE -ne 0) { throw "Could not prepare the isolated LC-PFN workspace." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace cp "controller_benchmark" "${pod}:$remoteTool/controller_benchmark" -c slurmd
if ($LASTEXITCODE -ne 0) { throw "Could not upload the LC-PFN comparison code." }

$command = @"
set -e
if [ ! -x '$remoteEnvironment/bin/python' ]; then
  python -m venv '$remoteEnvironment'
  '$remoteEnvironment/bin/python' -m pip install --upgrade pip
  '$remoteEnvironment/bin/python' -m pip install lcpfn==0.1.3
fi
cd '$remoteTool'
PYTHONPATH='$remoteTool' '$remoteEnvironment/bin/python' -m controller_benchmark.compare_lcpfn \
  --snapshot '$remoteSnapshot' \
  --output-dir '$remoteOutput' \
  --horizon-checkpoints '$HorizonCheckpoints' \
  --minimum-checkpoints '$MinimumCheckpoints' \
  --useful-gain '$UsefulGain' \
  --patience '$Patience'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($command))
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- bash -lc "echo '$encoded' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "Isolated Rancher LC-PFN comparison failed." }

& kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:$remoteOutput/." $OutputDir -c slurmd
if ($LASTEXITCODE -ne 0) { throw "Could not download the LC-PFN comparison." }
Write-Host "LC-PFN comparison downloaded to: $OutputDir" -ForegroundColor Green
Write-Host "The isolated LC-PFN environment did not modify the training runtime." -ForegroundColor Green
