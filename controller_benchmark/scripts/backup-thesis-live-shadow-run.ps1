[CmdletBinding()]
param(
    [string]$RunId = "20261005T205600Z-thesis-48-es-v4-v5-real",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$DestinationRoot = "D:\\Masterarbeit_Vergleichssicherung",
    [int]$ChunkMiB = 64,
    [int]$CopyAttempts = 6,
    [switch]$AllowIncomplete
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }

$podJson = & kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=benchmark-orchestrator -o json
if ($LASTEXITCODE -ne 0) { throw "Could not list benchmark orchestrator pods." }
$pod = @(
    ($podJson | ConvertFrom-Json).items |
        Where-Object { $_.status.phase -eq "Running" } |
        Select-Object -ExpandProperty metadata |
        Select-Object -ExpandProperty name
) | Select-Object -First 1
if (-not $pod) { throw "No running benchmark orchestrator pod found." }

$statusText = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -- cat "/workspace-cache/controller-benchmarks/$RunId/status.json"
if ($LASTEXITCODE -ne 0) { throw "Run status not found for $RunId." }
$status = ($statusText -join "`n") | ConvertFrom-Json
if ($status.state -ne "COMPLETED" -and -not $AllowIncomplete) {
    Write-Host "Run $RunId is $($status.state) ($($status.completed_runs)/$($status.total_runs)); backup deferred." -ForegroundColor Yellow
    exit 0
}

$archivePath = [string]$status.result.long_term_archive.path
if ([string]::IsNullOrWhiteSpace($archivePath)) {
    if (-not $AllowIncomplete) { throw "Completed run has no long-term archive path." }
    $archivePath = "/workspace-cache/controller-benchmarks/$RunId"
}
$safeRunId = $RunId -replace '[^A-Za-z0-9_.-]', '_'
$remotePackage = "/tmp/$safeRunId-final-backup.tar.gz"
$remotePrefix = "/tmp/$safeRunId-final-backup.part-"
$prepare = @"
set -e
rm -f '$remotePackage' '$remotePackage.sha256' '$remotePrefix'*
tar -czf '$remotePackage' -C '`$(dirname '$archivePath')' '`$(basename '$archivePath')'
sha256sum '$remotePackage' > '$remotePackage.sha256'
split -b ${ChunkMiB}m -a 4 '$remotePackage' '$remotePrefix'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($prepare))
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -- sh -c "echo '$encoded' | base64 -d | sh"
if ($LASTEXITCODE -ne 0) { throw "Could not package the cluster archive." }

$remoteParts = @(& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -- sh -c "ls '$remotePrefix'*")
if ($LASTEXITCODE -ne 0 -or $remoteParts.Count -eq 0) { throw "No remote archive chunks found." }
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
$destination = Join-Path $DestinationRoot "Thesis_48_ES_V4_V5_Final_${safeRunId}_$stamp"
$partsDir = Join-Path $destination "parts"
New-Item -ItemType Directory -Force -Path $partsDir | Out-Null

Push-Location $partsDir
try {
    foreach ($remotePart in $remoteParts) {
        $name = Split-Path -Leaf $remotePart
        $copied = $false
        for ($attempt = 1; $attempt -le $CopyAttempts -and -not $copied; $attempt++) {
            Write-Host "Downloading $name, attempt $attempt/$CopyAttempts..."
            & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:$remotePart" ".\$name"
            $copied = $LASTEXITCODE -eq 0
        }
        if (-not $copied) { throw "Could not download $name." }
    }
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp "${pod}:$remotePackage.sha256" ".\$safeRunId-final-backup.tar.gz.sha256"
    if ($LASTEXITCODE -ne 0) { throw "Could not download the cluster checksum." }
} finally { Pop-Location }

$localPackage = Join-Path $destination "$safeRunId-final-backup.tar.gz"
$output = [IO.File]::Open($localPackage, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    foreach ($part in Get-ChildItem -LiteralPath $partsDir -Filter "$safeRunId-final-backup.part-*" -File | Sort-Object Name) {
        $input = [IO.File]::OpenRead($part.FullName)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally { $output.Dispose() }

$expectedHash = ((Get-Content -Raw -LiteralPath (Join-Path $partsDir "$safeRunId-final-backup.tar.gz.sha256")).Trim() -split '\s+')[0].ToLowerInvariant()
$actualHash = (Get-FileHash -LiteralPath $localPackage -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw "Final backup checksum mismatch: expected $expectedHash, got $actualHash." }

[IO.File]::WriteAllText(
    (Join-Path $destination "status.json"),
    (($status | ConvertTo-Json -Depth 100) + "`n"),
    [Text.UTF8Encoding]::new($false)
)
$summary = [ordered]@{
    schema_version = 1
    benchmark_run_id = $RunId
    backed_up_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_archive = $archivePath
    local_package = $localPackage
    sha256 = $actualHash
    run_state = $status.state
    completed_runs = $status.completed_runs
    total_runs = $status.total_runs
}
[IO.File]::WriteAllText(
    (Join-Path $destination "backup-summary.json"),
    (($summary | ConvertTo-Json -Depth 10) + "`n"),
    [Text.UTF8Encoding]::new($false)
)
Write-Host "Final thesis run backup complete and verified: $destination" -ForegroundColor Green

