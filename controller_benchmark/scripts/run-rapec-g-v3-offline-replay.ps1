[CmdletBinding()]
param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Output)) {
    $stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
    $Output = Join-Path $repoRoot "results\controller-benchmark\offline-replays\rapec-g-v3-vs-v2-$stamp.json"
}
elseif (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $repoRoot $Output
}

Push-Location $repoRoot
try {
    & python -m controller_benchmark.rapec_g_v3_offline_replay `
        --vision-source "results/controller-optimization/sources/20260729T170915Z-rapec-v3-v8-nine-dataset" `
        --nonvision-run-dir "results/controller-benchmark/async-submissions/2026-08-10_220548-rapec-generalized-nonvision-hardest-v1/20260810T200548Z-live-shadow-dev" `
        --nonvision-metrics-dir "results/controller-benchmark/offline-replays/20260810T200548Z-live-shadow-dev-metrics" `
        --controllers "controller_benchmark/config/generalized-live-shadow-v3-small-controllers.json" `
        --output $Output
    if ($LASTEXITCODE -ne 0) { throw "RAPEC-G v3 offline replay failed." }
}
finally {
    Pop-Location
}
