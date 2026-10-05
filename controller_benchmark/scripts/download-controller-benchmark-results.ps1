[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$Destination = "D:\Masterarbeit_Vergleichssicherung\ControllerBenchmarks",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy"
)
$ErrorActionPreference = "Stop"
$target = Join-Path $Destination $RunId
if (Test-Path $target) { throw "Destination already exists: $target" }
New-Item -ItemType Directory -Force $target | Out-Null
$pod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=benchmark-orchestrator -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if (-not $pod) { throw "benchmark-orchestrator is not running." }
Push-Location $target
try { & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:/workspace-cache/controller-benchmarks/$RunId/." . } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "Result download failed." }
Write-Host "Benchmark downloaded: $target" -ForegroundColor Green
