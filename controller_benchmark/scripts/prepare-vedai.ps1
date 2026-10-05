[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
Set-Location $RepoRoot
function Invoke-Kubectl([string[]]$Arguments) {
    & kubectl @Arguments
    if ($LASTEXITCODE -ne 0) { throw "kubectl failed: kubectl $($Arguments -join ' ')" }
}

$pod = (& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+' | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "No running slurmd pod found." }
$converter = "controller_benchmark/runners/prepare_vedai.py"
Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "cp", $converter, "${pod}:/tmp/prepare_vedai.py", "-c", "slurmd")
$forceValue = if ($Force) { "1" } else { "0" }
$command = @'
set -euo pipefail
SOURCE=/workspace-cache/vedai512-source
OUTPUT=/workspace-cache/vedai512-yolo
mkdir -p "$SOURCE"
if [ "__FORCE__" = "1" ]; then rm -rf "$OUTPUT"; fi
if [ -s "$OUTPUT/dataset-manifest.json" ]; then
  echo "VEDAI cache already prepared: $OUTPUT"
  cat "$OUTPUT/dataset-manifest.json"
  exit 0
fi
cd "$SOURCE"
python /tmp/prepare_vedai.py --source "$SOURCE" --output "$OUTPUT" --seed 0 --download
echo "VEDAI preparation complete. Research/non-commercial terms and required citation apply."
'@.Replace("__FORCE__", $forceValue)
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($command))
Invoke-Kubectl @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $pod, "-c", "slurmd", "--", "bash", "-lc", "echo '$encodedCommand' | base64 -d | bash")
