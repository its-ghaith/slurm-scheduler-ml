[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [string]$SnapshotPath = "",
    [string]$Kubeconfig = "",
    [string]$Namespace = "",
    [switch]$SkipDashboard,
    [switch]$SkipModels,
    [switch]$ValidateOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $SnapshotPath) { $SnapshotPath = $scriptDirectory }
$SnapshotPath = [System.IO.Path]::GetFullPath($SnapshotPath)
$manifestPath = Join-Path $SnapshotPath "snapshot-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Kein gueltiger Snapshot: $manifestPath fehlt."
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

if (-not $Namespace) { $Namespace = [string]$manifest.namespace }
if (-not $Kubeconfig) {
    $candidates = @(
        (Join-Path (Get-Location).Path "rancherConfigs\main.yaml"),
        (Join-Path ([string]$manifest.source_repo_path) "rancherConfigs\main.yaml"),
        $env:KUBECONFIG
    ) | Where-Object { $_ }
    $Kubeconfig = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Kubeconfig) {
        throw "Kubeconfig nicht automatisch gefunden. Mit -Kubeconfig D:\repo\rancherConfigs\main.yaml angeben."
    }
}
$Kubeconfig = [System.IO.Path]::GetFullPath($Kubeconfig)
if (-not (Test-Path -LiteralPath $Kubeconfig)) { throw "Kubeconfig nicht gefunden: $Kubeconfig" }

function Invoke-Native {
    param([Parameter(Mandatory = $true)][string]$Command, [Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command fehlgeschlagen: $Command $($Arguments -join ' ')" }
}

function Get-NativeText {
    param([Parameter(Mandatory = $true)][string]$Command, [Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command fehlgeschlagen: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

function Get-PodName {
    param([Parameter(Mandatory = $true)][string]$App)
    $pod = Get-NativeText -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace,
        "get", "pod", "-l", "app=$App", "-o", "jsonpath={.items[0].metadata.name}"
    )
    if (-not $pod) { throw "Kein Pod mit Label app=$App gefunden." }
    return $pod
}

function Copy-ToPod {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$Pod,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [string]$Container = ""
    )
    if (-not (Test-Path -LiteralPath $LocalPath)) { throw "Snapshot-Daten fehlen: $LocalPath" }
    # kubectl cp interprets the colon in an absolute Windows drive path as a pod separator.
    Push-Location -LiteralPath $LocalPath
    try {
        $args = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "cp")
        if ($Container) { $args += @("-c", $Container) }
        $args += @(".", "${Pod}:${RemotePath}")
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            & kubectl @args
            if ($LASTEXITCODE -eq 0) { break }
            if ($attempt -eq 3) { throw "kubectl cp fehlgeschlagen: $LocalPath" }
            Write-Host "Kopierversuch $attempt fehlgeschlagen; wiederhole ..." -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Pruefe Snapshot-Dateien und SHA256-Pruefsummen ..." -ForegroundColor Cyan
$checksumPath = Join-Path $SnapshotPath "checksums\sha256.txt"
if (-not (Test-Path -LiteralPath $checksumPath)) { throw "Pruefsummendatei fehlt: $checksumPath" }
foreach ($line in Get-Content -LiteralPath $checksumPath) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split "  ", 2
    if ($parts.Count -ne 2) { throw "Ungueltige Pruefsummenzeile: $line" }
    $file = Join-Path $SnapshotPath ($parts[1].Replace("/", "\"))
    if (-not (Test-Path -LiteralPath $file)) { throw "Snapshot-Datei fehlt: $file" }
    $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $parts[0]) { throw "Pruefsummenfehler: $file" }
}

Invoke-Native -Command "kubectl" -Arguments @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pods")
$summary = "Restore von $($manifest.jobs) Vergleichsjobs aus '$SnapshotPath' nach Namespace '$Namespace'. Aktuelle Metriken, Logs und MLflow-Daten werden ersetzt."
Write-Host $summary -ForegroundColor Yellow
if ($ValidateOnly) {
    Write-Host "Snapshot, Pruefsummen und Clusterzugriff sind gueltig. Es wurden keine Daten veraendert." -ForegroundColor Green
    return
}
if (-not $Force -and -not $PSCmdlet.ShouldProcess($Namespace, $summary)) { return }

$slurmdPod = Get-PodName -App "slurmd"
$mlflowPod = Get-PodName -App "mlflow"

Write-Host "Breche aktive SLURM-Jobs ab ..." -ForegroundColor Cyan
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "exec", "deploy/slurmctld", "--", "bash", "-lc", "scancel --state=PENDING,RUNNING 2>/dev/null || true"
)

Write-Host "Ersetze Jobmetriken und Logs ..." -ForegroundColor Cyan
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc",
    "mkdir -p /workspace/energy_metrics/node_exporter /workspace/logs; rm -rf /workspace/energy_metrics/* /workspace/logs/*"
)
Copy-ToPod -LocalPath (Join-Path $SnapshotPath "data\energy_metrics") -Pod $slurmdPod -Container "slurmd" -RemotePath "/workspace/energy_metrics"
Copy-ToPod -LocalPath (Join-Path $SnapshotPath "data\logs") -Pod $slurmdPod -Container "slurmd" -RemotePath "/workspace/logs"

if (-not $SkipModels) {
    Write-Host "Ersetze gespeicherte Jobmodelle ..." -ForegroundColor Cyan
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace,
        "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc",
        "mkdir -p /workspace-cache/model_registry; rm -rf /workspace-cache/model_registry/*"
    )
    $modelRoot = Join-Path $SnapshotPath "data\model_registry"
    foreach ($modelDirectory in Get-ChildItem -LiteralPath $modelRoot -Directory | Sort-Object Name) {
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace,
            "exec", $slurmdPod, "-c", "slurmd", "--", "mkdir", "-p",
            "/workspace-cache/model_registry/$($modelDirectory.Name)"
        )
        Copy-ToPod -LocalPath $modelDirectory.FullName -Pod $slurmdPod -Container "slurmd" `
            -RemotePath "/workspace-cache/model_registry/$($modelDirectory.Name)"
    }
}

Write-Host "Ersetze MLflow-Backend und Artefakte ..." -ForegroundColor Cyan
Invoke-Native -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "exec", $mlflowPod, "--", "bash", "-lc", "mkdir -p /mlflow /mlruns; rm -rf /mlflow/* /mlruns/*"
)
Copy-ToPod -LocalPath (Join-Path $SnapshotPath "data\mlflow\backend") -Pod $mlflowPod -RemotePath "/mlflow"
Copy-ToPod -LocalPath (Join-Path $SnapshotPath "data\mlflow\artifacts") -Pod $mlflowPod -RemotePath "/mlruns"

if (-not $SkipDashboard) {
    Write-Host "Stelle das gesicherte Grafana-Dashboard bereit ..." -ForegroundColor Cyan
    $dashboardPath = Join-Path $SnapshotPath "config\slurm-energy-overview.json"
    if (-not (Test-Path -LiteralPath $dashboardPath)) { throw "Dashboard fehlt: $dashboardPath" }
    $tempYaml = Join-Path $env:TEMP "grafana-dashboard-restore-$PID.yaml"
    $tempDashboard = Join-Path $env:TEMP "slurm-energy-overview-restore-$PID.json"
    try {
        $dashboardContent = Get-Content -LiteralPath $dashboardPath -Raw
        [System.IO.File]::WriteAllText(
            $tempDashboard,
            $dashboardContent,
            (New-Object System.Text.UTF8Encoding($false))
        )
        Get-NativeText -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace,
            "create", "configmap", "grafana-dashboard-slurm-energy",
            "--from-file=slurm-energy-overview.json=$tempDashboard",
            "--dry-run=client", "-o", "yaml"
        ) | Set-Content -LiteralPath $tempYaml -Encoding utf8
        Invoke-Native -Command "kubectl" -Arguments @("--kubeconfig", $Kubeconfig, "apply", "-f", $tempYaml)
        Invoke-Native -Command "kubectl" -Arguments @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deployment/grafana")
        Invoke-Native -Command "kubectl" -Arguments @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/grafana", "--timeout=300s")
    }
    finally {
        Remove-Item -LiteralPath $tempYaml -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $tempDashboard -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Warte kurz auf den naechsten Prometheus-Scrape ..." -ForegroundColor Cyan
Start-Sleep -Seconds 8
$query = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "exec", "deploy/prometheus", "--", "sh", "-lc",
    "wget -qO- 'http://localhost:9090/api/v1/query?query=count(slurm_job_info)'"
)
Write-Host "Prometheus-Pruefung: $query"

Write-Host "" 
Write-Host "Vergleich erfolgreich wiederhergestellt." -ForegroundColor Green
Write-Host "Erwartete Jobs: $($manifest.jobs)" -ForegroundColor Green
Write-Host "Git-Referenz: $($manifest.git.commit)" -ForegroundColor Green
