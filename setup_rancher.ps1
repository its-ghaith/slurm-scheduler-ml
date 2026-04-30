param(
    [ValidateSet("bootstrap", "rollout", "submit", "test", "portforward")]
    [string]$Action = "bootstrap",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$Manifest = "rancherConfigs/slurm-stack.yaml",
    [string]$TrainCmd = "python train_with_energy_tracking_mlflow.py",
    [string]$BatchSize = "",
    [string]$MlflowTrackingUri = "",
    [string]$GpuSampleIntervalSec = "",
    [string]$ElectricityPriceEurPerKwh = "",
    [string]$GridCo2KgPerKwh = "",
    [string]$PueFactor = "",
    [string]$MlflowLogJobEnergy = "",
    [string]$MlflowJobEnergyExperiment = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Kubectl {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )
    & kubectl @Args
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed: kubectl $($Args -join ' ')"
    }
}

function Import-DotEnv {
    param([string]$Path = ".env")
    if (-not (Test-Path $Path)) { return }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { return }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim().Trim("'`"")
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Ensure-Namespace {
    $ns = & kubectl --kubeconfig $Kubeconfig get ns $Namespace --ignore-not-found -o name
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed while checking namespace '$Namespace' using kubeconfig '$Kubeconfig'"
    }
    if (-not $ns) {
        Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "create", "namespace", $Namespace)
    }
}

function Ensure-GhcrSecret {
    if (-not $env:CR_PAT) {
        throw "CR_PAT missing. Put CR_PAT in .env or env var."
    }
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "secret", "ghcr-pull-secret", "--ignore-not-found")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "secret", "docker-registry", "ghcr-pull-secret", "--docker-server=ghcr.io", "--docker-username=its-ghaith", "--docker-password=$env:CR_PAT")
}

function Apply-Stack {
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "apply", "-f", $Manifest)
}

function Restart-And-Wait {
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deploy/mlflow", "deploy/slurmctld", "deploy/slurmd")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/mlflow", "--timeout=300s")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/slurmctld", "--timeout=300s")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/slurmd", "--timeout=300s")
}

function Submit-Job {
    $pairs = @("TRAIN_CMD='$TrainCmd'")
    if ($BatchSize) { $pairs += "BATCH_SIZE='$BatchSize'" }
    if ($MlflowTrackingUri) { $pairs += "MLFLOW_TRACKING_URI='$MlflowTrackingUri'" }
    if ($GpuSampleIntervalSec) { $pairs += "GPU_SAMPLE_INTERVAL_SEC='$GpuSampleIntervalSec'" }
    if ($ElectricityPriceEurPerKwh) { $pairs += "ELECTRICITY_PRICE_EUR_PER_KWH='$ElectricityPriceEurPerKwh'" }
    if ($GridCo2KgPerKwh) { $pairs += "GRID_CO2_KG_PER_KWH='$GridCo2KgPerKwh'" }
    if ($PueFactor) { $pairs += "PUE_FACTOR='$PueFactor'" }
    if ($MlflowLogJobEnergy) { $pairs += "MLFLOW_LOG_JOB_ENERGY='$MlflowLogJobEnergy'" }
    if ($MlflowJobEnergyExperiment) { $pairs += "MLFLOW_JOB_ENERGY_EXPERIMENT='$MlflowJobEnergyExperiment'" }
    $cmd = ($pairs -join " ") + " sbatch /workspace/slurm/train_mlflow_local.slurm"
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--", "bash", "-lc", $cmd)
}

function Run-Test {
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pods", "-o", "wide")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurmctld", "--", "bash", "-lc", "scontrol ping && sinfo -N -l")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/prometheus", "--", "sh", "-lc", "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'")
}

function Show-PortForward {
    Write-Host "Run in separate terminals:" -ForegroundColor Yellow
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/grafana 3000:3000"
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/mlflow 5000:5000"
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/prometheus 9090:9090"
}

Import-DotEnv

if ($PSBoundParameters.ContainsKey("Action") -eq $false -and $env:ACTION) { $Action = $env:ACTION }
if ($PSBoundParameters.ContainsKey("Kubeconfig") -eq $false -and $env:KUBECONFIG_PATH) { $Kubeconfig = $env:KUBECONFIG_PATH }
if ($PSBoundParameters.ContainsKey("Namespace") -eq $false -and $env:K8S_NAMESPACE) { $Namespace = $env:K8S_NAMESPACE }
if ($PSBoundParameters.ContainsKey("Manifest") -eq $false -and $env:K8S_MANIFEST) { $Manifest = $env:K8S_MANIFEST }
if ($PSBoundParameters.ContainsKey("TrainCmd") -eq $false -and $env:TRAIN_CMD) { $TrainCmd = $env:TRAIN_CMD }
if ($PSBoundParameters.ContainsKey("BatchSize") -eq $false -and $env:BATCH_SIZE) { $BatchSize = $env:BATCH_SIZE }
if ($PSBoundParameters.ContainsKey("MlflowTrackingUri") -eq $false -and $env:MLFLOW_TRACKING_URI) { $MlflowTrackingUri = $env:MLFLOW_TRACKING_URI }
if ($PSBoundParameters.ContainsKey("GpuSampleIntervalSec") -eq $false -and $env:GPU_SAMPLE_INTERVAL_SEC) { $GpuSampleIntervalSec = $env:GPU_SAMPLE_INTERVAL_SEC }
if ($PSBoundParameters.ContainsKey("ElectricityPriceEurPerKwh") -eq $false -and $env:ELECTRICITY_PRICE_EUR_PER_KWH) { $ElectricityPriceEurPerKwh = $env:ELECTRICITY_PRICE_EUR_PER_KWH }
if ($PSBoundParameters.ContainsKey("GridCo2KgPerKwh") -eq $false -and $env:GRID_CO2_KG_PER_KWH) { $GridCo2KgPerKwh = $env:GRID_CO2_KG_PER_KWH }
if ($PSBoundParameters.ContainsKey("PueFactor") -eq $false -and $env:PUE_FACTOR) { $PueFactor = $env:PUE_FACTOR }
if ($PSBoundParameters.ContainsKey("MlflowLogJobEnergy") -eq $false -and $env:MLFLOW_LOG_JOB_ENERGY) { $MlflowLogJobEnergy = $env:MLFLOW_LOG_JOB_ENERGY }
if ($PSBoundParameters.ContainsKey("MlflowJobEnergyExperiment") -eq $false -and $env:MLFLOW_JOB_ENERGY_EXPERIMENT) { $MlflowJobEnergyExperiment = $env:MLFLOW_JOB_ENERGY_EXPERIMENT }

switch ($Action) {
    "bootstrap" {
        Ensure-Namespace
        Ensure-GhcrSecret
        Apply-Stack
        Restart-And-Wait
        Run-Test
    }
    "rollout" { Restart-And-Wait }
    "submit" { Submit-Job }
    "test" { Run-Test }
    "portforward" { Show-PortForward }
}
