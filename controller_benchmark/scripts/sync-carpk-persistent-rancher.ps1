[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$DatasetPath = "rotationally-invariant-cnns/data/carpk",
    [string]$BackupRoot = "D:\\Masterarbeit_Vergleichssicherung\\Thesis_48_ES_V4_V5_Datasets",
    [int]$ChunkMiB = 32,
    [int]$CopyAttempts = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
if (-not [IO.Path]::IsPathRooted($DatasetPath)) { $DatasetPath = Join-Path $RepoRoot $DatasetPath }
if (-not (Test-Path -LiteralPath $DatasetPath -PathType Container)) { throw "CARPK dataset not found: $DatasetPath" }

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
$archive = Join-Path $BackupRoot "carpk-source.tar.gz"
if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host "Creating the persistent CARPK source archive..." -ForegroundColor Cyan
    & tar.exe -czf $archive -C $DatasetPath .
    if ($LASTEXITCODE -ne 0) { throw "Could not create CARPK archive." }
}
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$shortHash = $hash.Substring(0, 16)
$partsDir = Join-Path $BackupRoot "carpk-source-$shortHash-parts"
New-Item -ItemType Directory -Force -Path $partsDir | Out-Null

$chunkBytes = [int64]$ChunkMiB * 1MB
$archiveLength = (Get-Item -LiteralPath $archive).Length
$expectedParts = [int][Math]::Ceiling($archiveLength / [double]$chunkBytes)
$existingParts = @(Get-ChildItem -LiteralPath $partsDir -Filter "part-*" -File | Sort-Object Name)
$validParts = $existingParts.Count -eq $expectedParts
if ($validParts) {
    for ($index = 0; $index -lt $existingParts.Count; $index++) {
        $expectedLength = [Math]::Min($chunkBytes, $archiveLength - ([int64]$index * $chunkBytes))
        if ($existingParts[$index].Length -ne $expectedLength) { $validParts = $false; break }
    }
}
if (-not $validParts) {
    if ($existingParts.Count -gt 0) {
        throw "Existing CARPK chunks are incomplete. Preserve them and use a new BackupRoot for a fresh split."
    }
    Write-Host "Splitting CARPK archive into $expectedParts chunks..." -ForegroundColor Cyan
    $input = [IO.File]::OpenRead($archive)
    try {
        $buffer = New-Object byte[] (1MB)
        for ($index = 0; $index -lt $expectedParts; $index++) {
            $partPath = Join-Path $partsDir ("part-{0:D4}" -f $index)
            $output = [IO.File]::Open($partPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $remaining = [Math]::Min($chunkBytes, $archiveLength - $input.Position)
                while ($remaining -gt 0) {
                    $read = $input.Read($buffer, 0, [int][Math]::Min($buffer.Length, $remaining))
                    if ($read -le 0) { throw "Unexpected end of CARPK archive." }
                    $output.Write($buffer, 0, $read)
                    $remaining -= $read
                }
            } finally { $output.Dispose() }
        }
    } finally { $input.Dispose() }
}

$podJson = & kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o json
if ($LASTEXITCODE -ne 0) { throw "Could not list slurmd pods." }
$pod = @(
    ($podJson | ConvertFrom-Json).items |
        Where-Object { $_.status.phase -eq "Running" } |
        Select-Object -ExpandProperty metadata |
        Select-Object -ExpandProperty name
) | Select-Object -First 1
if (-not $pod) { throw "No running slurmd pod found." }

$incoming = "/workspace-cache/.incoming/carpk-$shortHash"
$remoteArchive = "/workspace-cache/dataset-archives/carpk-$hash.tar.gz"
$remoteDataset = "/workspace-cache/datasets/carpk-$shortHash"
& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- mkdir -p $incoming /workspace-cache/dataset-archives /workspace-cache/datasets
if ($LASTEXITCODE -ne 0) { throw "Could not prepare the persistent CARPK upload directory." }

$parts = @(Get-ChildItem -LiteralPath $partsDir -Filter "part-*" -File | Sort-Object Name)
Push-Location $partsDir
try {
    foreach ($part in $parts) {
        $remoteLength = (& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- sh -c "wc -c < '$incoming/$($part.Name)' 2>/dev/null || echo 0").Trim()
        if ($LASTEXITCODE -eq 0 -and $remoteLength -eq [string]$part.Length) {
            Write-Host "Already uploaded: $($part.Name)" -ForegroundColor DarkGray
            continue
        }
        $copied = $false
        for ($attempt = 1; $attempt -le $CopyAttempts -and -not $copied; $attempt++) {
            Write-Host "Uploading $($part.Name) ($($part.Length) bytes), attempt $attempt/$CopyAttempts..."
            & kubectl --kubeconfig $Kubeconfig -n $Namespace cp ".\$($part.Name)" "${pod}:$incoming/$($part.Name)" -c slurmd
            if ($LASTEXITCODE -eq 0) {
                $remoteLength = (& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- sh -c "wc -c < '$incoming/$($part.Name)' 2>/dev/null || echo 0").Trim()
                $copied = $LASTEXITCODE -eq 0 -and $remoteLength -eq [string]$part.Length
            }
        }
        if (-not $copied) { throw "Could not upload $($part.Name)." }
    }
} finally { Pop-Location }

$finish = @"
set -e
if [ ! -f '$remoteArchive' ] || [ "`$(sha256sum '$remoteArchive' | awk '{print `$1}')" != '$hash' ]; then
  cat '$incoming'/part-* > '$remoteArchive'
fi
test "`$(sha256sum '$remoteArchive' | awk '{print `$1}')" = '$hash'
mkdir -p '$remoteDataset'
if [ ! -f '$remoteDataset/data.yaml' ]; then
  tar -xzf '$remoteArchive' -C '$remoteDataset'
fi
if [ ! -e /workspace-cache/carpk ]; then
  ln -s '$remoteDataset' /workspace-cache/carpk
fi
test -f /workspace-cache/carpk/data.yaml
test -d /workspace-cache/carpk/raw
find /workspace-cache/carpk/raw -type f | wc -l
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($finish))
$remoteCount = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec $pod -c slurmd -- bash -lc "echo '$encoded' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "CARPK archive verification or extraction failed." }

[IO.File]::WriteAllText(
    (Join-Path $BackupRoot "carpk-source.tar.gz.sha256"),
    "$hash  carpk-source.tar.gz`n",
    [Text.UTF8Encoding]::new($false)
)
Write-Host "CARPK synchronized and verified: $remoteCount raw files, SHA256 $hash" -ForegroundColor Green

