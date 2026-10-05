[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$PretrainingJobId = "",
    [switch]$QueueAfterPretraining,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'") + "-thesis-48-live-shadow"
}
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}

function Get-RunningPodName([string]$LabelSelector) {
    $podJson = & kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l $LabelSelector -o json
    if ($LASTEXITCODE -ne 0) { throw "Could not list pods matching $LabelSelector." }
    $podDocument = $podJson | ConvertFrom-Json
    return @(
        $podDocument.items |
            Where-Object { $_.status.phase -eq "Running" } |
            ForEach-Object { $_.metadata.name }
    ) | Select-Object -First 1
}

$benchmarkVersion = "thesis-results-live-shadow-48-v1"
$generatedRoot = Join-Path $RepoRoot "results\controller-benchmark\generated-manifests"
$manifest = Join-Path $generatedRoot "$benchmarkVersion.json"
New-Item -ItemType Directory -Force $generatedRoot | Out-Null

$manifestArguments = @(
    "-m", "controller_benchmark.generalized_manifest",
    "--vision-source", "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json",
    "--classification-source", "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json",
    "--pretrained-source", "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json",
    "--include-vision",
    "--suite", "hardest",
    "--output", $manifest
)
& python @manifestArguments
if ($LASTEXITCODE -ne 0) { throw "Could not generate the 48-case thesis manifest." }

$manifestDocument = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
$manifestDocument.benchmark_version = $benchmarkVersion
$manifestDocument.execution.experiment_name = $benchmarkVersion
$manifestDocument.execution.timeout_seconds = 172800
$manifestDocument.execution.cooldown_seconds = 10
$manifestDocument.execution | Add-Member -NotePropertyName require_complete_lifecycle_metrics -NotePropertyValue $true -Force
$manifestDocument.execution | Add-Member -NotePropertyName measurement_max_attempts -NotePropertyValue 3 -Force
$manifestDocument.execution | Add-Member -NotePropertyName dashboard_deployment_mode -NotePropertyValue "provisioned" -Force
$manifestDocument.execution.dashboard_files = @("thesis-results-energy-adaptive-mlops.json")
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText(
    $manifest,
    ($manifestDocument | ConvertTo-Json -Depth 100),
    $utf8
)

$cases = @($manifestDocument.stages | ForEach-Object { $_.cases }).Count
if ($cases -ne 48) { throw "Expected exactly 48 Full100 cases, generated $cases." }

$dashboardPath = Join-Path $RepoRoot "grafana\dashboards\rapec-g-v4-live-shadow-cv-nv.json"
& python -m controller_benchmark.dashboard.build_thesis_results_dashboard `
    --output $dashboardPath --default-run-id $RunId `
    --uid "rapec-g-v4-live-shadow-cv-nv"
if ($LASTEXITCODE -ne 0) { throw "Could not generate the thesis dashboard." }

if (-not $PlanOnly) {
    $slurmdPod = Get-RunningPodName "app=slurmd"
    if (-not $slurmdPod) { throw "No running slurmd pod found." }
    $checkpointCount = (& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $slurmdPod -c slurmd -- bash -lc "find '/workspace-cache/controller-pretrained-cross-domain-hardest-v1' -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l").Trim()
    if ($LASTEXITCODE -ne 0 -or [int]$checkpointCount -lt 15) {
        if (-not $QueueAfterPretraining) {
            throw "Only $checkpointCount/15 required pretrained source checkpoints exist. Complete pretraining first or use -QueueAfterPretraining."
        }
        $slurmctldPod = Get-RunningPodName "app=slurmctld"
        if (-not $slurmctldPod) { throw "No running slurmctld pod found." }
        $active = @(& kubectl --kubeconfig $Kubeconfig -n $Namespace exec $slurmctldPod -c slurmctld -- squeue -h -n rapec-pretrain -t PENDING,RUNNING -o "%A" | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+$' })
        if ($active.Count -ne 1) { throw "Expected exactly one active rapec-pretrain job, found $($active.Count)." }
        $PretrainingJobId = $active[0]
    }

    & kubectl --kubeconfig $Kubeconfig -n $Namespace create configmap controller-benchmark-status-web `
        "--from-file=server.py=$(Join-Path $RepoRoot 'controller_benchmark\status_web\server.py')" `
        --dry-run=client -o yaml | & kubectl --kubeconfig $Kubeconfig -n $Namespace apply -f -
    if ($LASTEXITCODE -ne 0) { throw "Could not update the persistent status website." }
    & kubectl --kubeconfig $Kubeconfig -n $Namespace rollout restart deployment/controller-benchmark-status-web
    if ($LASTEXITCODE -ne 0) { throw "Could not restart the status website." }

    $thesisDashboardConfigMap = "grafana-dashboard-thesis-results"
    & kubectl --kubeconfig $Kubeconfig -n $Namespace create configmap $thesisDashboardConfigMap `
        "--from-file=$([IO.Path]::GetFileName($dashboardPath))=$dashboardPath" `
        --dry-run=client -o yaml | & kubectl --kubeconfig $Kubeconfig -n $Namespace apply -f -
    if ($LASTEXITCODE -ne 0) { throw "Could not apply the dedicated thesis dashboard ConfigMap." }

    # Older deployments may contain a previous copy under the same filename.
    # Remove only that key after the dedicated replacement exists, preventing
    # duplicate Grafana UIDs while preserving every other dashboard.
    $legacyDashboardKey = [IO.Path]::GetFileName($dashboardPath)
    $legacyConfigJson = & kubectl --kubeconfig $Kubeconfig -n $Namespace get configmap grafana-dashboard-slurm-energy -o json
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the existing Grafana dashboard ConfigMap." }
    $legacyConfig = $legacyConfigJson | ConvertFrom-Json
    if ($legacyConfig.data.PSObject.Properties.Name -contains $legacyDashboardKey) {
        $removePatch = "[" + (@{ op = "remove"; path = "/data/$legacyDashboardKey" } | ConvertTo-Json -Compress) + "]"
        $removePatchFile = Join-Path ([IO.Path]::GetTempPath()) "grafana-thesis-key-patch-$PID.json"
        [IO.File]::WriteAllText($removePatchFile, $removePatch, $utf8)
        try {
            & kubectl --kubeconfig $Kubeconfig -n $Namespace patch configmap grafana-dashboard-slurm-energy --type=json "--patch-file=$removePatchFile"
            if ($LASTEXITCODE -ne 0) { throw "Could not remove the superseded thesis dashboard key." }
        } finally {
            Remove-Item -LiteralPath $removePatchFile -Force -ErrorAction SilentlyContinue
        }
    }

    $grafanaJson = & kubectl --kubeconfig $Kubeconfig -n $Namespace get deployment grafana -o json
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the Grafana deployment." }
    $grafana = $grafanaJson | ConvertFrom-Json
    $dashboardVolumeIndex = -1
    for ($index = 0; $index -lt $grafana.spec.template.spec.volumes.Count; $index++) {
        if ($grafana.spec.template.spec.volumes[$index].name -eq "dashboard") {
            $dashboardVolumeIndex = $index
            break
        }
    }
    if ($dashboardVolumeIndex -lt 0) { throw "Grafana dashboard volume was not found." }
    $dashboardVolume = @{
        name = "dashboard"
        projected = @{
            sources = @(
                @{ configMap = @{ name = "grafana-dashboard-slurm-energy" } }
                @{ configMap = @{ name = $thesisDashboardConfigMap } }
            )
        }
    }
    $volumePatchOperation = @{
        op = "replace"
        path = "/spec/template/spec/volumes/$dashboardVolumeIndex"
        value = $dashboardVolume
    }
    $volumePatch = "[" + ($volumePatchOperation | ConvertTo-Json -Compress -Depth 10) + "]"
    $volumePatchFile = Join-Path ([IO.Path]::GetTempPath()) "grafana-thesis-volume-patch-$PID.json"
    [IO.File]::WriteAllText($volumePatchFile, $volumePatch, $utf8)
    try {
        & kubectl --kubeconfig $Kubeconfig -n $Namespace patch deployment grafana --type=json "--patch-file=$volumePatchFile"
        if ($LASTEXITCODE -ne 0) { throw "Could not project the thesis dashboard into Grafana." }
    } finally {
        Remove-Item -LiteralPath $volumePatchFile -Force -ErrorAction SilentlyContinue
    }
    & kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status deployment/grafana --timeout=180s
    if ($LASTEXITCODE -ne 0) { throw "Grafana did not become ready after dashboard provisioning." }
    $grafanaReloaded = $false
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        $reloadOutput = & kubectl --kubeconfig $Kubeconfig -n $Namespace exec deploy/grafana -- curl -fsS -u admin:admin -X POST http://localhost:3000/api/admin/provisioning/dashboards/reload 2>&1
        if ($LASTEXITCODE -eq 0) {
            $grafanaReloaded = $true
            Write-Host $reloadOutput
            break
        }
        Start-Sleep -Seconds 3
    }
    if (-not $grafanaReloaded) { throw "Could not reload Grafana provisioning after 60 seconds." }
}

$arguments = @{
    Manifest = $manifest
    Controllers = "controller_benchmark/config/rapec-g-v4-live-shadow-controllers.json"
    Kubeconfig = $Kubeconfig
    Namespace = $Namespace
    ExpectedCases = 48
    BenchmarkVersion = $benchmarkVersion
    DashboardFile = "thesis-results-energy-adaptive-mlops.json"
    DashboardBuilder = "controller_benchmark.dashboard.build_thesis_results_dashboard"
    ExperimentName = $benchmarkVersion
    DashboardTitle = ""
    DashboardUid = "rapec-g-v4-live-shadow-cv-nv"
    SubmissionLabel = "thesis-results-48-live-shadow"
}
$arguments.RunId = $RunId
if (-not [string]::IsNullOrWhiteSpace($PretrainingJobId)) { $arguments.PretrainingJobId = $PretrainingJobId }
if ($PlanOnly) { $arguments.PlanOnly = $true }

& (Join-Path $PSScriptRoot "submit-live-shadow-nine-dataset-async.ps1") @arguments
if ($LASTEXITCODE -ne 0) { throw "Thesis 48-case Live-Shadow submission failed." }

if ($PlanOnly) {
    Write-Host "Plan validated: 48 real Full100 jobs, 48 virtual Standard-ES jobs, 48 virtual RAPEC-G-v4 jobs." -ForegroundColor Green
} else {
    Write-Host "Rancher now owns the complete campaign; this computer may be switched off." -ForegroundColor Green
    $tunnelPod = & kubectl --kubeconfig $Kubeconfig -n $Namespace get pods `
        -l app=controller-benchmark-status-tunnel `
        --sort-by=.metadata.creationTimestamp `
        -o jsonpath='{.items[-1:].metadata.name}'
    $tunnelLog = if ($tunnelPod) {
        (& kubectl --kubeconfig $Kubeconfig -n $Namespace logs $tunnelPod --tail=100 2>&1) -join "`n"
    } else { "" }
    $urlMatches = [regex]::Matches($tunnelLog, 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($urlMatches.Count -gt 0) {
        Write-Host "Live results: $($urlMatches[$urlMatches.Count - 1].Value)/" -ForegroundColor Green
    } else {
        Write-Warning "The status service is running, but the current Quick Tunnel URL was not available in its log yet."
    }
}
