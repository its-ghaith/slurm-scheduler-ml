[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$Dashboard
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
function Invoke-Native([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}
if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$Kubeconfig = [System.IO.Path]::GetFullPath($Kubeconfig)
$Snapshot = [System.IO.Path]::GetFullPath($Snapshot)
$Dashboard = [System.IO.Path]::GetFullPath($Dashboard)
$prom = Join-Path $Snapshot "data\energy_metrics\node_exporter\controller_benchmark.prom"
if (-not (Test-Path $prom)) { throw "Benchmark Prometheus file missing: $prom" }
$slurmdPod = (& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}").Trim().Split()[0]
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "cp", "-c", "slurmd", $prom, "${slurmdPod}:/workspace/energy_metrics/node_exporter/controller-benchmark-$([IO.Path]::GetFileName($Snapshot)).prom")

$dashboardFiles = Get-ChildItem (Join-Path $RepoRoot "grafana\dashboards") -Filter "*.json" -File
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "configmap", "grafana-dashboard-slurm-energy", "--ignore-not-found=true")
$create = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "configmap", "grafana-dashboard-slurm-energy")
foreach ($file in $dashboardFiles) { $create += "--from-file=$($file.Name)=$($file.FullName)" }
$create += "--from-file=controller-benchmark.json=$Dashboard"
Invoke-Native "kubectl" $create
Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/grafana", "--", "curl", "-fsS", "-u", "admin:admin", "-X", "POST", "http://localhost:3000/api/admin/provisioning/dashboards/reload")
Write-Host "Controller Benchmark dashboard deployed: /d/controller-benchmark/controller-benchmark-stability-and-generalisation" -ForegroundColor Green
