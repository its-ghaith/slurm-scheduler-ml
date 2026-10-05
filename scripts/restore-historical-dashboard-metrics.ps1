[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$OutputDirectory = "results/historical-dashboard-restore"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Invoke-Native([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed: $Command $($Arguments -join ' ')"
    }
}

function Get-NativeText([string]$Command, [string[]]$Arguments) {
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)"
    }
    return ($output | Out-String).Trim()
}

function Test-SnapshotMetrics([string]$Snapshot) {
    $checksumPath = Join-Path $Snapshot "checksums\sha256.txt"
    if (-not (Test-Path -LiteralPath $checksumPath)) {
        throw "Checksum file missing: $checksumPath"
    }

    $verified = 0
    foreach ($line in [System.IO.File]::ReadLines($checksumPath)) {
        $match = [regex]::Match($line, '^([0-9a-fA-F]{64})\s\s(.+)$')
        if (-not $match.Success) { continue }
        $relativePath = $match.Groups[2].Value.Replace('/', '\')
        if ($relativePath -notlike 'data\energy_metrics\node_exporter\*.prom') { continue }
        $path = Join-Path $Snapshot $relativePath
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Snapshot metric missing: $path"
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($actual -ine $match.Groups[1].Value) {
            throw "Checksum mismatch: $path"
        }
        $verified++
    }
    if ($verified -eq 0) {
        throw "No metric checksums found in $checksumPath"
    }
    return $verified
}

function Merge-PrometheusFiles([string]$SourceDirectory, [string]$Destination) {
    $help = [ordered]@{}
    $types = [ordered]@{}
    $samples = [ordered]@{}

    foreach ($source in Get-ChildItem -LiteralPath $SourceDirectory -Filter '*.prom' -File | Sort-Object Name) {
        foreach ($line in [System.IO.File]::ReadLines($source.FullName)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $helpMatch = [regex]::Match($line, '^# HELP\s+([A-Za-z_:][A-Za-z0-9_:]*)\s+(.+)$')
            if ($helpMatch.Success) {
                $name = $helpMatch.Groups[1].Value
                if (-not $help.Contains($name)) { $help[$name] = $line }
                continue
            }
            $typeMatch = [regex]::Match($line, '^# TYPE\s+([A-Za-z_:][A-Za-z0-9_:]*)\s+(\S+)$')
            if ($typeMatch.Success) {
                $name = $typeMatch.Groups[1].Value
                $type = $typeMatch.Groups[2].Value
                if ($types.Contains($name) -and $types[$name] -ne $type) {
                    throw "Conflicting TYPE for $name in $($source.FullName)"
                }
                $types[$name] = $type
                continue
            }
            if ($line.StartsWith('#')) { continue }

            $separator = $line.LastIndexOf(' ')
            if ($separator -lt 1) {
                throw "Invalid Prometheus sample in $($source.FullName): $line"
            }
            $identity = $line.Substring(0, $separator)
            if ($samples.Contains($identity) -and $samples[$identity] -ne $line) {
                throw "Conflicting duplicate sample '$identity' in $($source.FullName)"
            }
            $samples[$identity] = $line
        }
    }

    # The presentation export already registers some of these names as UNTYPED
    # in Pushgateway. Keep all historical values and labels, but omit exposition
    # metadata so the shared metric families remain type-compatible.
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in $samples.Values) { $lines.Add($line) }
    [System.IO.File]::WriteAllText($Destination, (($lines -join "`n") + "`n"), $Utf8NoBom)

    return [pscustomobject]@{
        files = @(Get-ChildItem -LiteralPath $SourceDirectory -Filter '*.prom' -File).Count
        samples = $samples.Count
        metric_families = @($help.Keys + $types.Keys | Sort-Object -Unique).Count
        bytes = (Get-Item -LiteralPath $Destination).Length
    }
}

$Kubeconfig = Resolve-RepoPath $Kubeconfig
$OutputDirectory = Resolve-RepoPath $OutputDirectory
$snapshots = @(
    [pscustomobject]@{
        comparison_set = '7_run'
        push_job = 'historical_dashboard_7_run'
        expected_jobs = 7
        path = 'D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_7Jobs_2026-07-13'
    },
    [pscustomobject]@{
        comparison_set = '60_run'
        push_job = 'historical_dashboard_60_run'
        expected_jobs = 60
        path = 'D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125'
    },
    [pscustomobject]@{
        comparison_set = '8_run'
        push_job = 'historical_dashboard_9_run_complete_rerun'
        expected_jobs = 9
        path = 'D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_9Jobs_CompleteRerun_2026-07-19_155800'
    },
    [pscustomobject]@{
        comparison_set = 'energy_guard_study'
        push_job = 'historical_dashboard_energy_guard_study'
        expected_jobs = 24
        path = 'D:\Masterarbeit_Vergleichssicherung\CARPK_GeneralizedEnergyGuard_20pct_24Jobs_2026-07-22_034745'
    }
)

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$report = [System.Collections.Generic.List[object]]::new()
$pushgatewayPod = Get-NativeText 'kubectl' @(
    '--kubeconfig', $Kubeconfig, '-n', $Namespace,
    'get', 'pod', '-l', 'app=pushgateway',
    '-o', "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}"
)
$pushgatewayPod = @($pushgatewayPod -split '\s+' | Where-Object { $_ })[0]
if (-not $pushgatewayPod) { throw 'No running Pushgateway pod found.' }

foreach ($snapshot in $snapshots) {
    $manifestPath = Join-Path $snapshot.path 'snapshot-manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Snapshot manifest missing: $manifestPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ([int]$manifest.jobs -ne $snapshot.expected_jobs) {
        throw "Snapshot $($snapshot.path) contains $($manifest.jobs) jobs; expected $($snapshot.expected_jobs)."
    }
    $verified = Test-SnapshotMetrics -Snapshot $snapshot.path
    $source = Join-Path $snapshot.path 'data\energy_metrics\node_exporter'
    $destination = Join-Path $OutputDirectory "$($snapshot.comparison_set).prom"
    $merged = Merge-PrometheusFiles -SourceDirectory $source -Destination $destination

    $remotePath = "/tmp/historical-$($snapshot.comparison_set).prom"
    Push-Location -LiteralPath $OutputDirectory
    try {
        Invoke-Native 'kubectl' @(
            '--kubeconfig', $Kubeconfig, '-n', $Namespace,
            'cp', (Split-Path -Leaf $destination), "${pushgatewayPod}:$remotePath"
        )
    }
    finally {
        Pop-Location
    }
    $endpoint = "http://127.0.0.1:9091/metrics/job/$($snapshot.push_job)/comparison_set/$($snapshot.comparison_set)"
    Invoke-Native 'kubectl' @(
        '--kubeconfig', $Kubeconfig, '-n', $Namespace,
        'exec', 'deploy/pushgateway', '--', 'wget', '-qO-', '--post-file', $remotePath, $endpoint
    )
    Invoke-Native 'kubectl' @(
        '--kubeconfig', $Kubeconfig, '-n', $Namespace,
        'exec', 'deploy/pushgateway', '--', 'rm', '-f', $remotePath
    )

    $report.Add([pscustomobject]@{
        comparison_set = $snapshot.comparison_set
        jobs = $snapshot.expected_jobs
        verified_files = $verified
        source_files = $merged.files
        samples = $merged.samples
        metric_families = $merged.metric_families
        bytes = $merged.bytes
        source_snapshot = $snapshot.path
    })
}

$reportPath = Join-Path $OutputDirectory 'restore-report.json'
[System.IO.File]::WriteAllText(
    $reportPath,
    (($report | ConvertTo-Json -Depth 5) + "`n"),
    $Utf8NoBom
)
$report | Format-Table -AutoSize
Write-Host "Restore report: $reportPath"
