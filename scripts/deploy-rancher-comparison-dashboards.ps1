[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$SevenRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_7Jobs_2026-07-13",
    [string]$SixtyRunSnapshot = "D:\Masterarbeit_Vergleichssicherung\CARPK_StopPolicy_60Jobs_Multiseed_2026-07-14_2026-07-14_102125",
    [string]$EightRunSnapshot = "",
    [string]$EightRunDashboardTemplate = "grafana/dashboards/slurm-energy-overview.json",
    [string]$EightRunTitle = "8 Run - Conformal Controller",
    [int]$EightRunExpectedJobs = 8,
    [string]$Manifest = "rancherConfigs/slurm-stack.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Invoke-Native {
    param([Parameter(Mandatory = $true)][string]$Command, [Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')" }
}

function Get-NativeText {
    param([Parameter(Mandatory = $true)][string]$Command, [Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "$Command failed: $Command $($Arguments -join ' ')`n$($output | Out-String)" }
    return ($output | Out-String).Trim()
}

function Assert-Snapshot {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][int]$ExpectedJobs)
    foreach ($required in @(
        "snapshot-manifest.json",
        "config\slurm-energy-overview.json",
        "data\energy_metrics\node_exporter"
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $required))) {
            throw "Snapshot component missing: $(Join-Path $Path $required)"
        }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $Path "snapshot-manifest.json") -Raw | ConvertFrom-Json
    if ([int]$manifest.jobs -ne $ExpectedJobs) {
        throw "Snapshot '$Path' contains $($manifest.jobs) jobs; expected $ExpectedJobs."
    }
}

function Add-ComparisonLabelToMetrics {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDirectory,
        [Parameter(Mandatory = $true)][string]$DestinationDirectory,
        [Parameter(Mandatory = $true)][string]$ComparisonSet,
        [Parameter(Mandatory = $true)][string]$FilePrefix
    )

    foreach ($source in Get-ChildItem -LiteralPath $SourceDirectory -Filter "*.prom" -File | Sort-Object Name) {
        $destination = Join-Path $DestinationDirectory "$FilePrefix-$($source.Name)"
        $lines = foreach ($line in [System.IO.File]::ReadAllLines($source.FullName)) {
            if ($line.StartsWith("#") -or [string]::IsNullOrWhiteSpace($line)) { $line; continue }
            $match = [regex]::Match($line, '^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?(\s+.+)$')
            if (-not $match.Success) { $line; continue }
            $labels = $match.Groups[2].Value
            $comparisonLabel = "comparison_set=`"$ComparisonSet`""
            if ($labels) { $labels = "$comparisonLabel,$labels" } else { $labels = $comparisonLabel }
            "$($match.Groups[1].Value){$labels}$($match.Groups[3].Value)"
        }
        # Prometheus textfiles are consumed in Linux; force LF even when this
        # deployment script runs in Windows PowerShell.
        $content = ([string[]]$lines -join "`n") + "`n"
        [System.IO.File]::WriteAllText($destination, $content, $Utf8NoBom)
    }
}

function Add-ComparisonFilterToPromQl {
    param([Parameter(Mandatory = $true)][string]$Query, [Parameter(Mandatory = $true)][string]$ComparisonSet)
    $pattern = '(?<![\$A-Za-z0-9_:])(slurm_[A-Za-z0-9_:]+)(?:\{([^}]*)\})?'
    return [regex]::Replace($Query, $pattern, {
        param($match)
        $metric = $match.Groups[1].Value
        $labels = $match.Groups[2].Value
        if ($labels -match '(^|,)\s*comparison_set\s*=') { return $match.Value }
        $filter = "comparison_set=`"$ComparisonSet`""
        if ($labels) { return "$metric{$filter,$labels}" }
        return "$metric{$filter}"
    })
}

function Add-ComparisonFilterRecursively {
    param([object]$Node, [Parameter(Mandatory = $true)][string]$ComparisonSet)
    if ($null -eq $Node -or $Node -is [string] -or $Node -is [ValueType]) { return }
    if ($Node -is [System.Collections.IEnumerable] -and $Node -isnot [pscustomobject]) {
        foreach ($item in $Node) { Add-ComparisonFilterRecursively -Node $item -ComparisonSet $ComparisonSet }
        return
    }
    foreach ($property in @($Node.PSObject.Properties)) {
        if ($property.Value -is [string] -and (
            $property.Name -in @("expr", "definition") -or
            ($property.Name -eq "query" -and $property.Value -match 'label_values\s*\(')
        )) {
            $property.Value = Add-ComparisonFilterToPromQl -Query $property.Value -ComparisonSet $ComparisonSet
        }
        else {
            Add-ComparisonFilterRecursively -Node $property.Value -ComparisonSet $ComparisonSet
        }
    }
}

function New-ComparisonDashboard {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Uid,
        [Parameter(Mandatory = $true)][string]$ComparisonSet
    )
    $dashboard = Get-Content -LiteralPath $Source -Raw | ConvertFrom-Json
    $dashboard.title = $Title
    $dashboard.uid = $Uid
    $dashboard.id = $null
    $dashboard.version = 1
    Add-ComparisonFilterRecursively -Node $dashboard -ComparisonSet $ComparisonSet
    [System.IO.File]::WriteAllText($Destination, ($dashboard | ConvertTo-Json -Depth 100 -Compress), $Utf8NoBom)
}

function Update-ManifestDashboardConfigMap {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$SevenRunDashboard,
        [Parameter(Mandatory = $true)][string]$SixtyRunDashboard,
        [string]$EightRunDashboard = ""
    )
    $content = [System.IO.File]::ReadAllText($ManifestPath)
    $blockLines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in @(
        "apiVersion: v1", "kind: ConfigMap", "metadata:",
        "  name: grafana-dashboard-slurm-energy", "  namespace: $Namespace", "data:"
    )) { $blockLines.Add($line) }
    $dashboardItems = @(
        @{ Key = "7-run.json"; Path = $SevenRunDashboard },
        @{ Key = "60-run.json"; Path = $SixtyRunDashboard }
    )
    if ($EightRunDashboard) { $dashboardItems += @{ Key = "8-run.json"; Path = $EightRunDashboard } }
    foreach ($item in $dashboardItems) {
        $blockLines.Add("  $($item.Key): |")
        foreach ($line in [System.IO.File]::ReadAllLines($item.Path)) { $blockLines.Add("    $line") }
    }
    $newBlock = ($blockLines -join "`n") + "`n"
    $pattern = '(?ms)^apiVersion: v1\r?\nkind: ConfigMap\r?\nmetadata:\r?\n  name: grafana-dashboard-slurm-energy\r?\n.*?(?=^---\s*$)'
    if (-not [regex]::IsMatch($content, $pattern)) { throw "Dashboard ConfigMap block not found in $ManifestPath" }
    $blockRegex = [regex]::new($pattern)
    $content = $blockRegex.Replace($content, [System.Text.RegularExpressions.MatchEvaluator]{ param($match) $newBlock }, 1)
    $oldMount = 'mountPath: /var/lib/grafana/dashboards/slurm-energy-overview.json\r?\n\s+subPath: slurm-energy-overview.json'
    if ([regex]::IsMatch($content, $oldMount)) {
        $content = [regex]::Replace($content, $oldMount, "mountPath: /var/lib/grafana/dashboards", 1)
    }
    [System.IO.File]::WriteAllText($ManifestPath, $content, $Utf8NoBom)
}

$Kubeconfig = Resolve-RepoPath $Kubeconfig
$Manifest = Resolve-RepoPath $Manifest
$SevenRunSnapshot = [System.IO.Path]::GetFullPath($SevenRunSnapshot)
$SixtyRunSnapshot = [System.IO.Path]::GetFullPath($SixtyRunSnapshot)
$EightRunDashboardTemplate = Resolve-RepoPath $EightRunDashboardTemplate
Assert-Snapshot -Path $SevenRunSnapshot -ExpectedJobs 7
Assert-Snapshot -Path $SixtyRunSnapshot -ExpectedJobs 60
if ($EightRunSnapshot) {
    $EightRunSnapshot = [System.IO.Path]::GetFullPath($EightRunSnapshot)
    Assert-Snapshot -Path $EightRunSnapshot -ExpectedJobs $EightRunExpectedJobs
}

$tempRoot = Join-Path $env:TEMP "rancher-comparison-dashboards-$PID"
$metricDirectory = Join-Path $tempRoot "metrics"
$dashboardDirectory = Join-Path $RepoRoot "grafana\dashboards"
$sevenDashboard = Join-Path $dashboardDirectory "7-run.json"
$sixtyDashboard = Join-Path $dashboardDirectory "60-run.json"
$eightDashboard = Join-Path $dashboardDirectory "8-run.json"
$archiveName = "comparison-dashboard-metrics-$PID.tar.gz"
$archivePath = Join-Path $tempRoot $archiveName
$remoteArchive = "/tmp/$archiveName"

try {
    New-Item -ItemType Directory -Force -Path $metricDirectory, $dashboardDirectory | Out-Null
    Write-Host "Erzeuge getrennte Prometheus-Metriken ..." -ForegroundColor Cyan
    Add-ComparisonLabelToMetrics `
        -SourceDirectory (Join-Path $SevenRunSnapshot "data\energy_metrics\node_exporter") `
        -DestinationDirectory $metricDirectory -ComparisonSet "7_run" -FilePrefix "7-run"
    Add-ComparisonLabelToMetrics `
        -SourceDirectory (Join-Path $SixtyRunSnapshot "data\energy_metrics\node_exporter") `
        -DestinationDirectory $metricDirectory -ComparisonSet "60_run" -FilePrefix "60-run"
    if ($EightRunSnapshot) {
        Add-ComparisonLabelToMetrics `
            -SourceDirectory (Join-Path $EightRunSnapshot "data\energy_metrics\node_exporter") `
            -DestinationDirectory $metricDirectory -ComparisonSet "8_run" -FilePrefix "8-run"
    }

    Write-Host "Erzeuge historische Dashboard-Kopien ..." -ForegroundColor Cyan
    New-ComparisonDashboard `
        -Source (Join-Path $SevenRunSnapshot "config\slurm-energy-overview.json") `
        -Destination $sevenDashboard -Title "7 Run" -Uid "slurm-energy-7-run" -ComparisonSet "7_run"
    New-ComparisonDashboard `
        -Source (Join-Path $SixtyRunSnapshot "config\slurm-energy-overview.json") `
        -Destination $sixtyDashboard -Title "60 Run" -Uid "slurm-energy-60-run" -ComparisonSet "60_run"
    if ($EightRunSnapshot) {
        New-ComparisonDashboard `
            -Source $EightRunDashboardTemplate `
            -Destination $eightDashboard -Title $EightRunTitle `
            -Uid "slurm-energy-8-run" -ComparisonSet "8_run"
    }
    Update-ManifestDashboardConfigMap -ManifestPath $Manifest -SevenRunDashboard $sevenDashboard `
        -SixtyRunDashboard $sixtyDashboard -EightRunDashboard $(if ($EightRunSnapshot) { $eightDashboard } else { "" })

    Write-Host "Lade die getrennten Metriksätze in den Node Exporter ..." -ForegroundColor Cyan
    Invoke-Native -Command "tar" -Arguments @("-czf", $archivePath, "-C", $metricDirectory, ".")
    $slurmdPod = Get-NativeText -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pod", "-l", "app=slurmd",
        "-o", "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}"
    )
    $slurmdPod = @($slurmdPod -split '\s+' | Where-Object { $_ })[0]
    if (-not $slurmdPod) { throw "No running slurmd pod found." }
    Push-Location -LiteralPath $tempRoot
    try {
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "cp", "-c", "slurmd",
            $archiveName, "${slurmdPod}:$remoteArchive"
        )
    }
    finally { Pop-Location }
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmd", "-c", "slurmd", "--",
        "bash", "-lc", "set -e; test -d /workspace/energy_metrics/node_exporter; rm -f /workspace/energy_metrics/node_exporter/*.prom; tar -xzf $remoteArchive -C /workspace/energy_metrics/node_exporter; rm -f $remoteArchive"
    )

    Write-Host "Passe den Prometheus-Scrape an die historischen Zeitreihen an ..." -ForegroundColor Cyan
    $prometheusConfig = Get-NativeText -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "configmap", "prometheus-config", "-o", "json"
    ) | ConvertFrom-Json
    $prometheusYaml = [string]$prometheusConfig.data.'prometheus.yml'
    $scrapeSettings = "`n    scrape_interval: 60s`n    scrape_timeout: 55s"
    if ($prometheusYaml -notmatch '(?ms)- job_name:\s*slurmd-node-exporter.*?scrape_timeout:\s*55s') {
        $prometheusYaml = [regex]::Replace(
            $prometheusYaml,
            '(?m)^(\s*- job_name:\s*slurmd-node-exporter\s*)$',
            { param($match) $match.Groups[1].Value + $scrapeSettings },
            1
        )
        $prometheusPatchPath = Join-Path $tempRoot "prometheus-config-patch.json"
        $prometheusPatch = @{ data = @{ 'prometheus.yml' = $prometheusYaml } }
        [System.IO.File]::WriteAllText(
            $prometheusPatchPath,
            (ConvertTo-Json -InputObject $prometheusPatch -Compress),
            $Utf8NoBom
        )
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "patch", "configmap", "prometheus-config",
            "--type=merge", "--patch-file", $prometheusPatchPath
        )
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deployment/prometheus"
        )
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/prometheus", "--timeout=300s"
        )
    }

    Write-Host "Provisioniere getrennte Vergleichsdashboards ..." -ForegroundColor Cyan
    # Recreate instead of apply: kubectl's last-applied annotation duplicates the
    # large dashboard payload and can exceed the Kubernetes API request limit.
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "configmap",
        "grafana-dashboard-slurm-energy", "--ignore-not-found=true"
    )
    $configMapArguments = @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "configmap", "grafana-dashboard-slurm-energy",
        "--from-file=7-run.json=$sevenDashboard", "--from-file=60-run.json=$sixtyDashboard"
    )
    if ($EightRunSnapshot) { $configMapArguments += "--from-file=8-run.json=$eightDashboard" }
    Invoke-Native -Command "kubectl" -Arguments $configMapArguments

    $deployment = Get-NativeText -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "deployment", "grafana", "-o", "json"
    ) | ConvertFrom-Json
    $containerIndex = -1
    for ($index = 0; $index -lt @($deployment.spec.template.spec.containers).Count; $index++) {
        if ($deployment.spec.template.spec.containers[$index].name -eq "grafana") { $containerIndex = $index; break }
    }
    if ($containerIndex -lt 0) { throw "Grafana container not found." }
    $mounts = [object[]]$deployment.spec.template.spec.containers[$containerIndex].volumeMounts
    $mountIndex = -1
    for ($index = 0; $index -lt $mounts.Count; $index++) {
        if ($mounts[$index].name -eq "dashboard") { $mountIndex = $index; break }
    }
    if ($mountIndex -lt 0) { throw "Grafana dashboard volume mount not found." }
    $patch = [System.Collections.Generic.List[object]]::new()
    if ($mounts[$mountIndex].mountPath -ne "/var/lib/grafana/dashboards") {
        $patch.Add(@{ op = "replace"; path = "/spec/template/spec/containers/$containerIndex/volumeMounts/$mountIndex/mountPath"; value = "/var/lib/grafana/dashboards" })
    }
    if ($mounts[$mountIndex].PSObject.Properties.Name -contains "subPath") {
        $patch.Add(@{ op = "remove"; path = "/spec/template/spec/containers/$containerIndex/volumeMounts/$mountIndex/subPath" })
    }
    if ($patch.Count -gt 0) {
        $patchPath = Join-Path $tempRoot "grafana-dashboard-mount-patch.json"
        $patchJson = ConvertTo-Json -InputObject ([object[]]$patch) -Compress
        [System.IO.File]::WriteAllText($patchPath, $patchJson, $Utf8NoBom)
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "patch", "deployment", "grafana",
            "--type=json", "--patch-file", $patchPath
        )
        Invoke-Native -Command "kubectl" -Arguments @(
            "--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deployment/grafana", "--timeout=300s"
        )
    }
    else {
        Write-Host "Grafana mount is already correct; ConfigMap projection updates dashboards without a pod restart." -ForegroundColor DarkCyan
    }

    # Provisioned files can change without a pod rollout, but Grafana may keep
    # the previous dashboard until its next filesystem poll. Reload explicitly.
    Invoke-Native -Command "kubectl" -Arguments @(
        "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/grafana", "--",
        "curl", "-fsS", "-u", "admin:admin", "-X", "POST",
        "http://localhost:3000/api/admin/provisioning/dashboards/reload"
    )

    Write-Host "Warte auf Prometheus-Scrape und Grafana-Provisionierung ..." -ForegroundColor Cyan
    Start-Sleep -Seconds 12
    Write-Host "Vergleichsdashboards wurden bereitgestellt." -ForegroundColor Green
    Write-Host "7 Run UID: slurm-energy-7-run" -ForegroundColor Green
    Write-Host "60 Run UID: slurm-energy-60-run" -ForegroundColor Green
    if ($EightRunSnapshot) { Write-Host "8 Run UID: slurm-energy-8-run" -ForegroundColor Green }
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
