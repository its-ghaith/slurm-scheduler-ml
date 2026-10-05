[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [string]$Dashboard = "grafana/dashboards/energy-guard-study.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}
function Invoke-Native([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}
function Get-NativeText([string]$Command, [string[]]$Arguments) {
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

$Kubeconfig = Resolve-RepoPath $Kubeconfig
$Snapshot = Resolve-RepoPath $Snapshot
$Dashboard = Resolve-RepoPath $Dashboard
$manifest = Get-Content -LiteralPath (Join-Path $Snapshot "snapshot-manifest.json") -Raw | ConvertFrom-Json
if ([int]$manifest.jobs -ne 24) { throw "Energy Guard snapshot must contain 24 jobs; found $($manifest.jobs)." }

$tempRoot = Join-Path $env:TEMP "energy-guard-dashboard-$PID"
$metricDirectory = Join-Path $tempRoot "metrics"
$archiveName = "energy-guard-metrics-$PID.tar.gz"
$archivePath = Join-Path $tempRoot $archiveName
$remoteArchive = "/tmp/$archiveName"
$repoDashboard = Join-Path $RepoRoot "grafana\dashboards\energy-guard-study.json"
try {
    New-Item -ItemType Directory -Force -Path $metricDirectory | Out-Null
    foreach ($source in Get-ChildItem -LiteralPath (Join-Path $Snapshot "data\energy_metrics\node_exporter") -Filter "*.prom" -File) {
        $lines = foreach ($line in [System.IO.File]::ReadAllLines($source.FullName)) {
            if ($line.StartsWith("#") -or [string]::IsNullOrWhiteSpace($line)) { $line; continue }
            $match = [regex]::Match($line, '^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?(\s+.+)$')
            if (-not $match.Success) { $line; continue }
            $labels = $match.Groups[2].Value
            $comparison = 'comparison_set="energy_guard_study"'
            if ($labels) { $labels = "$comparison,$labels" } else { $labels = $comparison }
            "$($match.Groups[1].Value){$labels}$($match.Groups[3].Value)"
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $metricDirectory "energy-guard-$($source.Name)"),
            (([string[]]$lines -join "`n") + "`n"),
            $Utf8NoBom
        )
    }
    Invoke-Native "tar" @("-czf", $archivePath, "-C", $metricDirectory, ".")
    $slurmdPod = Get-NativeText "kubectl" @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd",
        "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}"
    )
    $slurmdPod = @($slurmdPod -split '\s+' | Where-Object { $_ })[0]
    Push-Location $tempRoot
    try { Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "cp", "-c", "slurmd", $archiveName, "${slurmdPod}:$remoteArchive") }
    finally { Pop-Location }
    Invoke-Native "kubectl" @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $slurmdPod, "-c", "slurmd", "--",
        "bash", "-lc", "set -e; tar -xzf $remoteArchive -C /workspace/energy_metrics/node_exporter; rm -f $remoteArchive"
    )

    if ([System.IO.Path]::GetFullPath($Dashboard) -ne [System.IO.Path]::GetFullPath($repoDashboard)) {
        Copy-Item -LiteralPath $Dashboard -Destination $repoDashboard -Force
    }
    $dashboardFiles = @("7-run.json", "60-run.json", "8-run.json", "stage-a.json", "stage-b.json", "energy-guard-study.json")
    foreach ($file in $dashboardFiles) {
        if (-not (Test-Path (Join-Path $RepoRoot "grafana\dashboards\$file"))) { throw "Dashboard file missing: $file" }
    }
    Invoke-Native "kubectl" @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "configmap", "grafana-dashboard-slurm-energy", "--ignore-not-found=true")
    $create = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "configmap", "grafana-dashboard-slurm-energy")
    foreach ($file in $dashboardFiles) { $create += "--from-file=$file=$(Join-Path $RepoRoot "grafana\dashboards\$file")" }
    Invoke-Native "kubectl" $create
    Invoke-Native "kubectl" @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/grafana", "--",
        "curl", "-fsS", "-u", "admin:admin", "-X", "POST", "http://localhost:3000/api/admin/provisioning/dashboards/reload"
    )
    Write-Host "Dashboard deployed: /d/slurm-energy-guard-study/generalized-energy-guard-cross-stage-20-validation" -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
