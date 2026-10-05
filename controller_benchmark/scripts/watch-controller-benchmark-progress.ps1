[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [int]$IntervalSeconds = 15,
    [switch]$ShowLogs,
    [int]$LogLines = 80
)

$ErrorActionPreference = "Stop"

function Format-PercentValue {
    param([object]$Value)
    if ($null -eq $Value) { return "n/a" }
    return "{0:N2} %" -f ([double]$Value * 100.0)
}

function Format-WhValue {
    param([object]$Value)
    if ($null -eq $Value) { return "n/a" }
    return "{0:N3} Wh" -f ([double]$Value * 1000.0)
}

function Format-SecondsValue {
    param([object]$Value)
    if ($null -eq $Value) { return "n/a" }
    return "{0:N1} s" -f [double]$Value
}

while ($true) {
    Clear-Host

    $statusRaw = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/benchmark-orchestrator -- `
        cat "/workspace-cache/controller-benchmarks/$RunId/status.json" 2>$null

    if ($LASTEXITCODE -ne 0 -or -not $statusRaw) {
        Write-Host "Benchmark ist noch in der Queue oder status.json ist noch nicht vorhanden." -ForegroundColor Yellow
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    $status = $statusRaw | ConvertFrom-Json

    [pscustomobject]@{
        RunId       = $status.benchmark_run_id
        State       = $status.state
        Progress    = "$($status.completed_runs)/$($status.total_runs)"
        CurrentJob  = $status.current_job_id
        CurrentCase = $status.current_case_id
        Updated     = $status.updated_at
        Error       = $status.error
    } | Format-List

    $status.runs |
        Select-Object sequence, case_id, strategy, job_id, state |
        Format-Table -AutoSize

    $jobId = $status.current_job_id
    if ($jobId) {
        Write-Host "`nAktuelle Epoch-Metriken fuer SLURM Job ${jobId}:" -ForegroundColor Cyan

        $lastEpochRaw = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmd -c slurmd -- `
            bash -lc "test -s /workspace/energy_metrics/epoch_timeline_job_$jobId.jsonl && tail -n 1 /workspace/energy_metrics/epoch_timeline_job_$jobId.jsonl || true"

        if ($lastEpochRaw) {
            $epoch = $lastEpochRaw | ConvertFrom-Json

            $quality = $epoch.quality_score
            if ($null -eq $quality) { $quality = $epoch.map50_95 }

            $bestQuality = $epoch.best_quality_score
            if ($null -eq $bestQuality) { $bestQuality = $epoch.best_map50_95 }

            [pscustomobject]@{
                Epoch         = $epoch.epoch
                Quality       = Format-PercentValue $quality
                BestQuality   = Format-PercentValue $bestQuality
                mAP50         = Format-PercentValue $epoch.map50
                mAP50_95      = Format-PercentValue $epoch.map50_95
                EpochWh       = Format-WhValue $epoch.gpu_energy_kwh
                DurationSec   = Format-SecondsValue $epoch.duration_seconds
                StopCandidate = $epoch.stop_candidate
                StopReason    = $epoch.stop_reason
            } | Format-List
        }
        else {
            Write-Host "Noch keine abgeschlossene Epoche fuer diesen Job gefunden." -ForegroundColor Yellow
        }

        if ($ShowLogs) {
            Write-Host "`nLive-Training-Log fuer SLURM Job ${jobId}:" -ForegroundColor Cyan

            & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/slurmd -c slurmd -- `
                bash -lc "echo '--- OUT ---'; test -f /workspace/logs/controller-benchmark-$jobId.out && tail -n $LogLines /workspace/logs/controller-benchmark-$jobId.out || echo 'stdout log not found yet'; echo '--- ERR ---'; test -f /workspace/logs/controller-benchmark-$jobId.err && tail -n $LogLines /workspace/logs/controller-benchmark-$jobId.err || echo 'stderr log not found yet'"
        }
    }

    if ($status.state -in @("COMPLETED", "FAILED")) {
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
