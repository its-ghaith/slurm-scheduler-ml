[CmdletBinding()]
param(
    [string]$AnalysisId = "",
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunsRoot = Join-Path $RepoRoot "results/controller-v8-sensitivity/runs"

if (-not $AnalysisId) {
    $latest = Get-ChildItem -LiteralPath $RunsRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No RAPEC-v8 sensitivity run was found below $RunsRoot"
    }
    $AnalysisId = $latest.Name
}

$RunDir = Join-Path $RunsRoot $AnalysisId
$ProgressPath = Join-Path $RunDir "progress.jsonl"

function Get-SensitivityProcess {
    $candidate = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.Contains("controller_benchmark.v8_sensitivity.analysis") -and
            $_.CommandLine.Contains($AnalysisId)
        } |
        ForEach-Object {
            Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        } |
        Sort-Object CPU, WorkingSet64 -Descending |
        Select-Object -First 1
    if (-not $candidate) { return $null }
    return $candidate
}

function Get-ProgressRecords {
    if (-not (Test-Path -LiteralPath $ProgressPath)) { return @() }
    $records = @()
    foreach ($line in Get-Content -LiteralPath $ProgressPath -Tail 2000 -ErrorAction SilentlyContinue) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $records += ($line | ConvertFrom-Json) } catch { }
    }
    return @($records)
}

function Format-Duration([double]$Seconds) {
    $span = [TimeSpan]::FromSeconds([math]::Max(0, $Seconds))
    if ($span.Days -gt 0) {
        return "{0}d {1:00}:{2:00}:{3:00}" -f $span.Days, $span.Hours, $span.Minutes, $span.Seconds
    }
    return "{0:00}:{1:00}:{2:00}" -f [math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds
}

function Show-Status {
    $process = Get-SensitivityProcess
    $records = @(Get-ProgressRecords)
    $latest = $records | Select-Object -Last 1
    $latestDesign = $records |
        Where-Object { $_.event -eq "design_evaluation_completed" } |
        Select-Object -Last 1

    if ($latest -and $latest.event -eq "sensitivity_completed") {
        $state = "COMPLETED"
    } elseif ($latest -and $latest.event -eq "sensitivity_failed") {
        $state = "FAILED"
    } elseif ($process) {
        $state = "RUNNING"
    } elseif ($latest) {
        $state = "INTERRUPTED_OR_FINISHED"
    } else {
        $state = "UNKNOWN"
    }

    $elapsed = if ($latest) { [double]$latest.elapsed_seconds } elseif ($process) {
        ((Get-Date) - $process.StartTime).TotalSeconds
    } else { 0 }
    $completed = if ($latestDesign -and $latestDesign.completed -ne $null) {
        [string]$latestDesign.completed
    } else { "0" }
    $total = if ($latestDesign -and $latestDesign.total -ne $null) {
        [string]$latestDesign.total
    } else { "n/a" }
    $topParameter = if ($latest -and $latest.top_parameter) {
        [string]$latest.top_parameter
    } else { "n/a" }
    $reliable = if ($latest -and $latest.surrogate_reliable -ne $null) {
        [string]$latest.surrogate_reliable
    } else { "n/a" }
    $cpu = if ($process) { "{0:N1}" -f [double]$process.CPU } else { "n/a" }
    $memory = if ($process) { "{0:N1}" -f ($process.WorkingSet64 / 1MB) } else { "n/a" }
    $workerProcessId = if ($process) { [string]$process.Id } else { "n/a" }

    [pscustomobject]@{
        AnalysisId = $AnalysisId
        State = $state
        Event = if ($latest) { [string]$latest.event } else { "n/a" }
        ReplayProgress = "$completed/$total"
        TopParameter = $topParameter
        SurrogateReliable = $reliable
        Runtime = Format-Duration $elapsed
        WorkerPID = $workerProcessId
        WorkerCPUSeconds = $cpu
        WorkerMemoryMB = $memory
        Updated = if ($latest) { [string]$latest.timestamp } else { "n/a" }
        Output = $RunDir
    } | Format-List | Out-Host

    if ($state -eq "FAILED") {
        $latest | Format-List error_type, error | Out-Host
    }
    return $state
}

do {
    if (-not $Once) { Clear-Host }
    $state = Show-Status
    if ($Once -or $state -in @("COMPLETED", "FAILED", "INTERRUPTED_OR_FINISHED", "UNKNOWN")) {
        break
    }
    Write-Host "Refresh every $IntervalSeconds seconds. Press Ctrl+C to stop watching." -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
