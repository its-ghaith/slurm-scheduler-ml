[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$LocalMetricsDir = "results/stop-policy-study/raw-metrics",
    [string]$OutputDir = "results/stop-policy-study/analysis",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
if (-not $SkipDownload) {
    New-Item -ItemType Directory -Force -Path $LocalMetricsDir | Out-Null
    $pod = (& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o jsonpath='{.items[0].metadata.name}').Trim()
    if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "No slurmd pod found." }
    & kubectl --kubeconfig $Kubeconfig -n $Namespace cp -c slurmd "${pod}:/workspace/energy_metrics/." $LocalMetricsDir
    if ($LASTEXITCODE -ne 0) { throw "Could not download study metrics from slurmd." }
}

python .\slurm\analyze_stop_policy_study.py --input-dir $LocalMetricsDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Study analysis failed." }
Write-Host "Analysis written to $OutputDir" -ForegroundColor Green
