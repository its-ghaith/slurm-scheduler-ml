[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$DestinationRoot = "D:\\Masterarbeit_Vergleichssicherung",
    [switch]$SkipPrometheusData
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
$destination = Join-Path $DestinationRoot "Thesis_48_ES_V4_V5_PreRun_$stamp"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

function Invoke-KubectlText {
    param([string[]]$Arguments)
    $output = & kubectl --kubeconfig $Kubeconfig -n $Namespace @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: kubectl $($Arguments -join ' ')`n$($output -join "`n")"
    }
    return ($output -join "`n") + "`n"
}

function Write-Utf8File {
    param([string]$Path, [string]$Content)
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}

function Get-RunningPod {
    param([string]$Selector)
    $json = Invoke-KubectlText @("get", "pod", "-l", $Selector, "-o", "json") | ConvertFrom-Json
    $pod = @($json.items | Where-Object { $_.status.phase -eq "Running" }) | Select-Object -First 1
    if (-not $pod) { throw "No running pod found for $Selector" }
    return $pod.metadata.name
}

$metadataDir = Join-Path $destination "cluster-metadata"
New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null
Write-Utf8File (Join-Path $metadataDir "all-resources.yaml") (Invoke-KubectlText @("get", "deploy,sts,daemonset,service,pvc,pod,configmap", "-o", "yaml"))
Write-Utf8File (Join-Path $metadataDir "grafana-dashboards.yaml") (Invoke-KubectlText @("get", "configmap", "grafana-dashboard-slurm-energy", "grafana-dashboard-thesis-results", "-o", "yaml"))
Write-Utf8File (Join-Path $metadataDir "prometheus-config.yaml") (Invoke-KubectlText @("get", "configmap", "prometheus-config", "-o", "yaml"))
Write-Utf8File (Join-Path $metadataDir "slurm-resources.yaml") (Invoke-KubectlText @("get", "deployment", "slurmctld", "slurmd", "benchmark-orchestrator", "controller-benchmark-status-web", "-o", "yaml"))

$pushgatewayPod = Get-RunningPod "app=pushgateway"
$mlflowPod = Get-RunningPod "app=mlflow"
$prometheusPod = Get-RunningPod "app=prometheus"

$pushgatewayDir = Join-Path $destination "pushgateway"
$mlflowDir = Join-Path $destination "mlflow"
New-Item -ItemType Directory -Force -Path $pushgatewayDir, $mlflowDir | Out-Null

# kubectl interprets the colon in an absolute Windows destination as another
# remote specification. Run from the backup root and use relative destinations.
Push-Location $destination
try {
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pushgatewayPod}:/data/." ".\pushgateway"
    if ($LASTEXITCODE -ne 0) { throw "Could not back up Pushgateway data." }
    New-Item -ItemType Directory -Force -Path ".\mlflow\backend", ".\mlflow\artifacts" | Out-Null
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${mlflowPod}:/mlflow/." ".\mlflow\backend"
    if ($LASTEXITCODE -ne 0) { throw "Could not back up MLflow backend data." }
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${mlflowPod}:/mlruns/." ".\mlflow\artifacts"
    if ($LASTEXITCODE -ne 0) { throw "Could not back up MLflow artifacts." }

    if (-not $SkipPrometheusData) {
        New-Item -ItemType Directory -Force -Path ".\prometheus" | Out-Null
        & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${prometheusPod}:/prometheus/." ".\prometheus"
        if ($LASTEXITCODE -ne 0) { throw "Could not back up Prometheus data." }
    }
} finally {
    Pop-Location
}

$checksums = Get-ChildItem -LiteralPath $destination -File -Recurse |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        $relative = [IO.Path]::GetRelativePath($destination, $_.FullName).Replace('\\', '/')
        "$($hash.Hash.ToLowerInvariant())  $relative"
    }
Write-Utf8File (Join-Path $destination "SHA256SUMS.txt") (($checksums -join "`n") + "`n")

$summary = [ordered]@{
    schema_version = 1
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    namespace = $Namespace
    kubeconfig = $Kubeconfig
    destination = $destination
    prometheus_data_included = -not $SkipPrometheusData
    pushgateway_pod = $pushgatewayPod
    prometheus_pod = $prometheusPod
    mlflow_pod = $mlflowPod
}
Write-Utf8File (Join-Path $destination "backup-summary.json") (($summary | ConvertTo-Json -Depth 10) + "`n")
Write-Host "Backup complete: $destination" -ForegroundColor Green

