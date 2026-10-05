[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) { $Kubeconfig = Join-Path $RepoRoot $Kubeconfig }
$installDir = Join-Path $RepoRoot "results\controller-benchmark\orchestrator-install"
New-Item -ItemType Directory -Force $installDir | Out-Null

# Remove the legacy sidecar without waiting for the GPU-backed slurmd pod.
$removeSidecar = '{"spec":{"template":{"spec":{"containers":[{"name":"benchmark-orchestrator","$patch":"delete"}]}}}}'
$removePatchPath = Join-Path $installDir "remove-legacy-sidecar.json"
[IO.File]::WriteAllText($removePatchPath, $removeSidecar, [Text.UTF8Encoding]::new($false))
& kubectl --kubeconfig $Kubeconfig -n $Namespace patch deployment slurmd --type strategic --patch-file $removePatchPath
if ($LASTEXITCODE -ne 0) { throw "Could not remove the legacy orchestrator sidecar." }

$document = Get-Content (Join-Path $RepoRoot "rancherConfigs/slurm-stack.yaml") -Raw
$parts = $document -split "(?m)^---\s*$"
$orchestrator = $parts | Where-Object { $_ -match "(?m)^kind:\s*Deployment\s*$" -and $_ -match "(?m)^\s*name:\s*benchmark-orchestrator\s*$" } | Select-Object -First 1
if (-not $orchestrator) { throw "benchmark-orchestrator Deployment not found in rancherConfigs/slurm-stack.yaml." }

$manifestPath = Join-Path $installDir "benchmark-orchestrator.yaml"
[IO.File]::WriteAllText($manifestPath, $orchestrator.Trim() + "`n", [Text.UTF8Encoding]::new($false))
& kubectl --kubeconfig $Kubeconfig apply -f $manifestPath
if ($LASTEXITCODE -ne 0) { throw "Could not install benchmark-orchestrator Deployment." }

$prometheusConfig = $parts | Where-Object { $_ -match "(?m)^kind:\s*ConfigMap\s*$" -and $_ -match "(?m)^\s*name:\s*prometheus-config\s*$" } | Select-Object -First 1
if (-not $prometheusConfig) { throw "prometheus-config ConfigMap not found in rancherConfigs/slurm-stack.yaml." }
$prometheusPath = Join-Path $installDir "prometheus-config.yaml"
[IO.File]::WriteAllText($prometheusPath, $prometheusConfig.Trim() + "`n", [Text.UTF8Encoding]::new($false))
& kubectl --kubeconfig $Kubeconfig apply -f $prometheusPath
if ($LASTEXITCODE -ne 0) { throw "Could not update Prometheus configuration." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace rollout restart deployment/prometheus
if ($LASTEXITCODE -ne 0) { throw "Could not restart Prometheus." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status deployment/prometheus --timeout=180s
if ($LASTEXITCODE -ne 0) { throw "Prometheus rollout failed." }

& kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status deployment/benchmark-orchestrator --timeout=300s
if ($LASTEXITCODE -ne 0) { throw "benchmark-orchestrator rollout failed." }
& kubectl --kubeconfig $Kubeconfig -n $Namespace get pods -l app=benchmark-orchestrator
Write-Host "Server-side benchmark orchestrator installed without a GPU request." -ForegroundColor Green
