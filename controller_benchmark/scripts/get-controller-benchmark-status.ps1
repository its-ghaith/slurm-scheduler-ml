[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$Follow
)
$ErrorActionPreference = "Stop"

function Format-PercentValue {
    param([object]$Value)
    if ($null -eq $Value) { return "n/a" }
    return "{0:N2} %" -f ([double]$Value * 100.0)
}

function Format-SecondsAsElapsed {
    param([object]$Value)
    if ($null -eq $Value) { return "n/a" }
    try {
        $total = [int][double]$Value
    }
    catch {
        return "n/a"
    }
    if ($total -lt 0) { $total = 0 }
    $span = [TimeSpan]::FromSeconds($total)
    return "{0}:{1:00}:{2:00}" -f [int]$span.TotalHours, $span.Minutes, $span.Seconds
}

function Format-ElapsedValue {
    param(
        [object]$Value,
        [object]$FallbackSeconds = $null
    )
    $text = (@($Value) -join "`n").Trim()
    if (-not [string]::IsNullOrWhiteSpace($text)) { return $text }
    return Format-SecondsAsElapsed $FallbackSeconds
}

function Get-CurrentJobDetails {
    param(
        [string]$RunId,
        [object]$Status
    )

    $jobId = $Status.current_job_id
    if (-not $jobId) { return }

    $elapsedRaw = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmctld -- `
        bash -lc "squeue -h -j $jobId -o '%M' 2>/dev/null || true" 2>$null

    $metricPath = "/workspace-cache/controller-benchmarks/$RunId/metrics/epoch_timeline_job_$jobId.jsonl"
    $fallbackMetricPath = "/workspace/energy_metrics/epoch_timeline_job_$jobId.jsonl"
    $lastEndCommand = @"
import json
from pathlib import Path

paths = [Path('$metricPath'), Path('$fallbackMetricPath')]
latest = None
for path in paths:
    if not path.is_file() or path.stat().st_size == 0:
        continue
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_kind = event.get('event')
        if event_kind == 'end' or (event_kind is None and event.get('epoch') is not None):
            latest = event
print(json.dumps(latest) if latest else '')
"@
    $lastEpochRaw = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmd -c slurmd -- `
        python -c $lastEndCommand 2>$null

    $details = [ordered]@{
        Laufzeit = Format-ElapsedValue $elapsedRaw $Status.current_job_running_seconds
    }

    if ($lastEpochRaw) {
        $epoch = $lastEpochRaw | ConvertFrom-Json
        $quality = $epoch.quality_score
        if ($null -eq $quality) { $quality = $epoch.map50_95 }

        $details["Aktuelle letzte Epoche"] = $epoch.epoch
        $details["Quality / mAP50-95"] = Format-PercentValue $quality
        $details["mAP50"] = Format-PercentValue $epoch.map50
    }
    else {
        $details["Aktuelle letzte Epoche"] = "n/a"
        $details["Quality / mAP50-95"] = "n/a"
        $details["mAP50"] = "n/a"
    }

    [pscustomobject]$details | Format-List
}

if (-not $RunId) {
    $RunId = (& kubectl --kubeconfig $Kubeconfig -n $Namespace exec deployment/benchmark-orchestrator -- bash -lc "find /workspace-cache/controller-benchmarks -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1").Trim()
}
if (-not $RunId) { throw "No server-side benchmark was found." }
do {
    $raw = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deployment/benchmark-orchestrator -- cat "/workspace-cache/controller-benchmarks/$RunId/status.json" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "Benchmark is queued; status file is not initialized yet." -ForegroundColor Yellow }
    else {
        $status = $raw | ConvertFrom-Json
        [pscustomobject]@{RunId=$status.benchmark_run_id;State=$status.state;Progress="$($status.completed_runs)/$($status.total_runs)";CurrentJob=$status.current_job_id;CurrentCase=$status.current_case_id;Updated=$status.updated_at;Error=$status.error} | Format-List
        Get-CurrentJobDetails -RunId $RunId -Status $status
        $status.runs | Select-Object sequence,case_id,strategy,job_id,state | Format-Table -AutoSize
        if (-not $Follow -or $status.state -in @("COMPLETED", "FAILED")) { break }
    }
    if (-not $Follow) { break }
    Start-Sleep 10
} while ($true)
