[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^ghcr\.io/.+@sha256:[a-f0-9]{64}$')]
    [string]$Image,
    [string]$Manifest = "rancherConfigs/slurm-stack.yaml",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$path = (Resolve-Path $Manifest).Path
$content = [IO.File]::ReadAllText($path)
$pattern = 'ghcr\.io/its-ghaith/slurm-scheduler-ml/mlops-slurm-runtime(?::[^\s]+|@sha256:[a-f0-9]{64})'
$updated = [regex]::Replace($content, $pattern, $Image)
$count = ([regex]::Matches($content, $pattern)).Count
if ($count -eq 0) { throw "No runtime image reference found in $Manifest." }
[IO.File]::WriteAllText($path, $updated, [Text.UTF8Encoding]::new($false))
Write-Host "Pinned $count runtime image references to $Image" -ForegroundColor Green

if ($Apply) {
    & kubectl --kubeconfig $Kubeconfig apply -f $Manifest
    if ($LASTEXITCODE -ne 0) { throw "kubectl apply failed." }
    foreach ($deployment in @("mlflow", "slurmctld", "slurmd")) {
        & kubectl --kubeconfig $Kubeconfig -n $Namespace rollout status "deployment/$deployment" --timeout=600s
        if ($LASTEXITCODE -ne 0) { throw "Rollout failed for $deployment." }
    }
}
