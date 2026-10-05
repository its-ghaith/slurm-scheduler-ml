[CmdletBinding()]
param(
    [string]$OptimizationId = "",
    [string]$RunsRoot = "",
    [string]$ProcessModule = "controller_benchmark.scientific_optimization.optimize",
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $RunsRoot) {
    $RunsRoot = Join-Path $RepoRoot "results/controller-scientific-optimization/runs"
} elseif (-not [IO.Path]::IsPathRooted($RunsRoot)) {
    $RunsRoot = Join-Path $RepoRoot $RunsRoot
}

if (-not $OptimizationId) {
    $latest = Get-ChildItem -LiteralPath $RunsRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No scientific optimization run was found below $RunsRoot"
    }
    $OptimizationId = $latest.Name
}

$RunDir = Join-Path $RunsRoot $OptimizationId
$ProgressPath = Join-Path $RunDir "progress.jsonl"

function Get-OptimizerProcesses {
    $candidates = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.Contains($ProcessModule) -and
            $_.CommandLine.Contains($OptimizationId)
        }
    $processes = @()
    foreach ($candidate in $candidates) {
        $process = Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
        if ($process) { $processes += $process }
    }
    return @($processes | Sort-Object CPU -Descending)
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
    if ($Seconds -lt 0) { $Seconds = 0 }
    $span = [TimeSpan]::FromSeconds($Seconds)
    if ($span.Days -gt 0) {
        return "{0}d {1:00}:{2:00}:{3:00}" -f $span.Days, $span.Hours, $span.Minutes, $span.Seconds
    }
    return "{0:00}:{1:00}:{2:00}" -f [math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds
}

function Show-Status {
    $processes = @(Get-OptimizerProcesses)
    $worker = $processes | Select-Object -First 1
    $records = @(Get-ProgressRecords)
    $latest = $records | Select-Object -Last 1
    $latestTrial = $records |
        Where-Object { $_.event -eq "optimization_trial_completed" } |
        Select-Object -Last 1

    $state = "UNKNOWN"
    if ($latest -and $latest.event -in @(
        "optimization_completed",
        "optimizer_benchmark_completed",
        "optimizer_benchmark_planned"
    )) {
        $state = "COMPLETED"
    } elseif ($latest -and $latest.event -eq "optimization_failed") {
        $state = "FAILED"
    } elseif ($worker) {
        $state = "RUNNING"
    } elseif ($latest) {
        $state = "INTERRUPTED_OR_FINISHED"
    } elseif (Test-Path -LiteralPath $RunDir) {
        $state = "INTERRUPTED"
    }

    if ($latest) {
        $elapsedSeconds = [double]$latest.elapsed_seconds
        $mode = "detailed JSONL progress"
        $eventName = [string]$latest.event
        $search = if ($latest.search) { [string]$latest.search } else { "n/a" }
        $completed = if ($latest.completed -ne $null) { [string]$latest.completed } else { "n/a" }
        $total = if ($latest.total -ne $null) { [string]$latest.total } else { "n/a" }
        $trial = if ($latestTrial) { [string]$latestTrial.trial } else { "n/a" }
        $backend = if ($latestTrial) { [string]$latestTrial.backend } else { "n/a" }
        $quality = if ($latestTrial -and $latestTrial.quality_cvar90 -ne $null) {
            "{0:N6}" -f [double]$latestTrial.quality_cvar90
        } else { "n/a" }
        $energy = if ($latestTrial -and $latestTrial.energy_saving_q25 -ne $null) {
            "{0:N2} %" -f (100.0 * [double]$latestTrial.energy_saving_q25)
        } else { "n/a" }
        $feasible = if ($latestTrial -and $latestTrial.quality_feasible -ne $null) {
            [string]$latestTrial.quality_feasible
        } else { "n/a" }
        $updated = [string]$latest.timestamp
    } else {
        $mode = "legacy run without detailed events"
        $completedSearches = @(
            Get-ChildItem -LiteralPath $RunDir -Filter "selected-controller.json" -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.DirectoryName -ne $RunDir }
        )
        $unfinishedSearches = @(
            Get-ChildItem -LiteralPath $RunDir -Directory -Recurse -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Parent -and
                    ($_.Parent.Name -eq "outer-folds" -or $_.Name -eq "final-fit") -and
                    -not (Test-Path -LiteralPath (Join-Path $_.FullName "selected-controller.json"))
                }
        )
        $activeSearch = $unfinishedSearches | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $eventName = "Optimizer process is active; exact trial is unavailable"
        if ($activeSearch) {
            $search = $activeSearch.FullName.Substring($RunDir.Length + 1)
            $eventName = "Running search; $($completedSearches.Count) search section(s) completed"
        } else {
            $search = "initialization or transition between searches"
        }
        $completed = "n/a"
        $total = "n/a"
        $trial = "n/a"
        $backend = "qnehvi (from command line)"
        $quality = "n/a"
        $energy = "n/a"
        $feasible = "n/a"
        $updated = (Get-Date).ToString("o")
        if ($worker) {
            $elapsedSeconds = ((Get-Date) - $worker.StartTime).TotalSeconds
        } else {
            $elapsedSeconds = 0
        }
    }

    $cpuSeconds = if ($worker) { "{0:N1}" -f [double]$worker.CPU } else { "n/a" }
    $memoryMb = if ($worker) { "{0:N1}" -f ($worker.WorkingSet64 / 1MB) } else { "n/a" }
    $workerPid = if ($worker) { [string]$worker.Id } else { "n/a" }
    $files = if (Test-Path -LiteralPath $RunDir) {
        @(Get-ChildItem -LiteralPath $RunDir -File -Recurse -ErrorAction SilentlyContinue).Count
    } else { 0 }

    $statusView = [pscustomobject]@{
        OptimizationId = $OptimizationId
        State = $state
        Mode = $mode
        Event = $eventName
        Search = $search
        StageProgress = "$completed/$total"
        LastCompletedTrial = $trial
        Backend = $backend
        QualityCVaR90 = $quality
        EnergySavingQ25 = $energy
        QualityFeasible = $feasible
        Runtime = Format-Duration $elapsedSeconds
        WorkerPID = $workerPid
        WorkerCPUSeconds = $cpuSeconds
        WorkerMemoryMB = $memoryMb
        OutputFiles = $files
        Updated = $updated
        Output = $RunDir
    }
    Write-Host ($statusView | Format-List | Out-String)

    if ($state -eq "RUNNING" -and -not $latest) {
        Write-Host "This optimization was started before detailed logging was added." -ForegroundColor Yellow
        Write-Host "It is active and was not restarted; detailed events are available for future runs." -ForegroundColor Yellow
    }
    return $state
}

do {
    if (-not $Once) { Clear-Host }
    $state = Show-Status
    if ($Once -or $state -in @("COMPLETED", "FAILED", "INTERRUPTED", "INTERRUPTED_OR_FINISHED", "UNKNOWN")) { break }
    Write-Host "Refresh every $IntervalSeconds seconds. Press Ctrl+C to stop watching." -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
