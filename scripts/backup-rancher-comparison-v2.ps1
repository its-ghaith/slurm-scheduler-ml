[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$OutputPath = "",
    [string]$SnapshotName = "carpk-stop-policy-comparison"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not [System.IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}
if (-not (Test-Path -LiteralPath $Kubeconfig)) {
    throw "Kubeconfig nicht gefunden: $Kubeconfig"
}
if (-not $OutputPath) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputPath = "D:\Masterarbeit_Vergleichssicherung\$SnapshotName-$stamp"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputPath) {
    throw "Das Snapshot-Ziel existiert bereits: $OutputPath"
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command fehlgeschlagen: $Command $($Arguments -join ' ')"
    }
}

function Get-NativeText {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$Command fehlgeschlagen: $Command $($Arguments -join ' ')`n$($output | Out-String)"
    }
    return ($output | Out-String).Trim()
}

function Get-PodName {
    param([Parameter(Mandatory = $true)][string]$App)
    $pod = Get-NativeText -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace,
        "get", "pod", "-l", "app=$App",
        "-o", "jsonpath={.items[0].metadata.name}"
    )
    if (-not $pod) { throw "Kein Pod mit Label app=$App gefunden." }
    return $pod
}

function Copy-FromPod {
    param(
        [Parameter(Mandatory = $true)][string]$Pod,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [string]$Container = ""
    )
    New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
    $archiveName = "rancher-snapshot-$PID-$([Guid]::NewGuid().ToString('N')).tar.gz"
    $remoteArchive = "/tmp/$archiveName"
    $localArchive = Join-Path $LocalPath $archiveName
    $remoteParent = [System.IO.Path]::GetDirectoryName($RemotePath).Replace("\", "/")
    $remoteLeaf = [System.IO.Path]::GetFileName($RemotePath)

    try {
        $execArgs = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $Pod)
        if ($Container) { $execArgs += @("-c", $Container) }
        $execArgs += @("--", "tar", "-czf", $remoteArchive, "-C", $remoteParent, $remoteLeaf)
        Invoke-Native -Command "kubectl" -Arguments $execArgs

        # Smaller compressed streams avoid long-lived kubectl cp connections through Rancher.
        Push-Location -LiteralPath $LocalPath
        try {
            $copyArgs = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "cp")
            if ($Container) { $copyArgs += @("-c", $Container) }
            $copyArgs += @("${Pod}:${remoteArchive}", $archiveName)
            for ($attempt = 1; $attempt -le 5; $attempt++) {
                & kubectl @copyArgs
                if ($LASTEXITCODE -eq 0) { break }
                Remove-Item -LiteralPath $localArchive -Force -ErrorAction SilentlyContinue
                if ($attempt -eq 5) { throw "kubectl cp fehlgeschlagen: $RemotePath" }
                Write-Host "Kopierversuch $attempt fehlgeschlagen; wiederhole komprimierte Uebertragung ..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
            }
        }
        finally {
            Pop-Location
        }

        & tar -xzf $localArchive -C $LocalPath --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "Lokales Entpacken fehlgeschlagen: $localArchive" }
    }
    finally {
        Remove-Item -LiteralPath $localArchive -Force -ErrorAction SilentlyContinue
        $cleanupArgs = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $Pod)
        if ($Container) { $cleanupArgs += @("-c", $Container) }
        $cleanupArgs += @("--", "rm", "-f", $remoteArchive)
        & kubectl @cleanupArgs 2>&1 | Out-Null
    }
}

function Copy-SourceItem {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $source = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source)) { return }
    $destination = Join-Path $OutputPath "source\$RelativePath"
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

Write-Host "Erstelle Vergleichssnapshot: $OutputPath" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
foreach ($folder in @("data", "config", "source", "reports", "checksums")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $OutputPath $folder) | Out-Null
}

Write-Host "Pruefe Cluster und ermittle Pods ..." -ForegroundColor Cyan
Invoke-Native -Command "kubectl" -Arguments @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pods")
$slurmdPod = Get-PodName -App "slurmd"
$mlflowPod = Get-PodName -App "mlflow"

Write-Host "Sichere kanonische Job-, Epochen- und Phasenmetriken ..." -ForegroundColor Cyan
Copy-FromPod -Pod $slurmdPod -Container "slurmd" -RemotePath "/workspace/energy_metrics" -LocalPath (Join-Path $OutputPath "data\energy_metrics")
Copy-FromPod -Pod $slurmdPod -Container "slurmd" -RemotePath "/workspace/logs" -LocalPath (Join-Path $OutputPath "data\logs")

Write-Host "Sichere trainierte Modelle und Inferenz-Metadaten ..." -ForegroundColor Cyan
$modelRoot = Join-Path $OutputPath "data\model_registry"
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
$modelDirectories = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "exec", $slurmdPod, "-c", "slurmd", "--", "bash", "-lc",
    "for f in /workspace/energy_metrics/epoch_summary_job_*.json; do [ -f `"`$f`" ] || continue; basename `"`$f`" | sed -E 's/^epoch_summary_(job_[0-9]+)\.json$/\1/'; done | sort -V"
)
foreach ($modelDirectory in @($modelDirectories -split "[`r`n]+" | Where-Object { $_ })) {
    Copy-FromPod -Pod $slurmdPod -Container "slurmd" `
        -RemotePath "/workspace-cache/model_registry/$modelDirectory" `
        -LocalPath (Join-Path $modelRoot $modelDirectory)
}

Write-Host "Sichere MLflow-Experimente und Artefakte ..." -ForegroundColor Cyan
Copy-FromPod -Pod $mlflowPod -RemotePath "/mlflow" -LocalPath (Join-Path $OutputPath "data\mlflow\backend")
Copy-FromPod -Pod $mlflowPod -RemotePath "/mlruns" -LocalPath (Join-Path $OutputPath "data\mlflow\artifacts")

Write-Host "Sichere Dashboard und Kubernetes-Definitionen ..." -ForegroundColor Cyan
$dashboardJson = Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "get", "configmap", "grafana-dashboard-slurm-energy",
    "-o", "jsonpath={.data.slurm-energy-overview\.json}"
)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $OutputPath "config\slurm-energy-overview.json"),
    $dashboardJson,
    $utf8NoBom
)
Get-NativeText -Command "kubectl" -Arguments @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace,
    "get", "deploy,svc,configmap,pvc", "-o", "yaml"
) | Set-Content -LiteralPath (Join-Path $OutputPath "config\cluster-resources.yaml") -Encoding utf8

foreach ($item in @(
    "scripts",
    "slurm",
    "grafana",
    "prometheus",
    "rancherConfigs\slurm-stack.yaml",
    "carpk_yolo_pipeline",
    "rotationally-invariant-cnns\src",
    "rotationally-invariant-cnns\slurm",
    "README.md",
    "requirements.txt",
    ".env.example"
)) {
    Copy-SourceItem -RelativePath $item
}

$gitCommit = Get-NativeText -Command "git" -Arguments @("-C", $RepoRoot, "rev-parse", "HEAD")
$gitBranch = Get-NativeText -Command "git" -Arguments @("-C", $RepoRoot, "branch", "--show-current")
$gitRemote = Get-NativeText -Command "git" -Arguments @("-C", $RepoRoot, "remote", "get-url", "origin")
$gitStatus = Get-NativeText -Command "git" -Arguments @("-C", $RepoRoot, "status", "--porcelain=v1")
$gitStatus | Set-Content -LiteralPath (Join-Path $OutputPath "source\git-status.txt") -Encoding utf8

Write-Host "Erzeuge lesbare Ergebnisuebersicht ..." -ForegroundColor Cyan
$rows = foreach ($epochFile in Get-ChildItem -LiteralPath (Join-Path $OutputPath "data\energy_metrics") -Filter "epoch_summary_job_*.json" | Sort-Object Name) {
    $epoch = Get-Content -LiteralPath $epochFile.FullName -Raw | ConvertFrom-Json
    $jobId = [string]$epoch.job_id
    $gpuPath = Join-Path $OutputPath "data\energy_metrics\gpu_summary_job_${jobId}.json"
    $phasePath = Join-Path $OutputPath "data\energy_metrics\gpu_summary_job_${jobId}_phases.json"
    $gpu = if (Test-Path -LiteralPath $gpuPath) { Get-Content -LiteralPath $gpuPath -Raw | ConvertFrom-Json } else { $null }
    $phase = if (Test-Path -LiteralPath $phasePath) { Get-Content -LiteralPath $phasePath -Raw | ConvertFrom-Json } else { $null }
    [pscustomobject]@{
        job_id = $jobId
        strategy = $epoch.comparison_strategy
        controller = $epoch.controller_mode
        epochs = @($epoch.epochs).Count
        adaptive_stopped = [bool]$epoch.adaptive_stopped
        stop_reason = $epoch.stop_reason
        map50 = $epoch.final_map50
        map50_95 = $epoch.final_map50_95
        precision = $epoch.final_precision
        recall = $epoch.final_recall
        slurm_gpu_energy_kwh = if ($gpu) { $gpu.gpu_energy_kwh } else { $null }
        slurm_duration_seconds = if ($gpu) { $gpu.duration_seconds } else { $null }
        codecarbon_total_energy_kwh = if ($phase) { $phase.codecarbon_job_total_energy_kwh } else { $null }
        codecarbon_gpu_energy_kwh = if ($phase) { $phase.codecarbon_job_total_gpu_energy_kwh } else { $null }
        cost_eur = if ($gpu) { $gpu.estimated_electricity_cost_eur } else { $null }
        co2_kg = if ($gpu) { $gpu.estimated_co2_kg } else { $null }
    }
}
$rows | Export-Csv -LiteralPath (Join-Path $OutputPath "reports\comparison-summary.csv") -NoTypeInformation -Encoding utf8
$rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputPath "reports\comparison-summary.json") -Encoding utf8

$snapshot = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString("o")
    snapshot_name = $SnapshotName
    namespace = $Namespace
    source_repo_path = $RepoRoot
    git = [ordered]@{
        branch = $gitBranch
        commit = $gitCommit
        remote = $gitRemote
        working_tree_clean = [string]::IsNullOrWhiteSpace($gitStatus)
    }
    jobs = @($rows).Count
    canonical_restore_sources = @(
        "data/energy_metrics",
        "data/mlflow/backend",
        "data/mlflow/artifacts",
        "data/model_registry",
        "data/logs",
        "config/slurm-energy-overview.json"
    )
    note = "Prometheus TSDB is intentionally not copied. Restored node-exporter textfiles recreate all dashboard series without stale WAL data."
}
$snapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputPath "snapshot-manifest.json") -Encoding utf8

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "restore-rancher-comparison.ps1") -Destination (Join-Path $OutputPath "restore.ps1") -Force

$hashes = Get-ChildItem -LiteralPath $OutputPath -Recurse -File |
    Where-Object { $_.FullName -notlike "*\checksums\sha256.txt" } |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        $relative = $_.FullName.Substring($OutputPath.Length + 1).Replace("\", "/")
        "$($hash.Hash.ToLowerInvariant())  $relative"
    }
$hashes | Set-Content -LiteralPath (Join-Path $OutputPath "checksums\sha256.txt") -Encoding ascii

$readme = @"
# CARPK stop-policy comparison snapshot

Created: $($snapshot.created_at)
Jobs: $($snapshot.jobs)
Git commit: $gitCommit

Restore into the Rancher namespace:

``````powershell
powershell -ExecutionPolicy Bypass -File .\restore.ps1 -Kubeconfig "$Kubeconfig" -Force
``````

The restore replaces current MLflow data, job metrics, logs, and matching job models in the target namespace. It does not reset or replay training.
"@
$readme | Set-Content -LiteralPath (Join-Path $OutputPath "README.md") -Encoding utf8

Write-Host "" 
Write-Host "Snapshot erfolgreich erstellt." -ForegroundColor Green
Write-Host "Pfad: $OutputPath" -ForegroundColor Green
Write-Host "Jobs: $($snapshot.jobs), Git: $gitCommit" -ForegroundColor Green
