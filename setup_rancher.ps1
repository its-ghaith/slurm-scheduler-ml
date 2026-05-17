param(
    [ValidateSet("bootstrap", "bootstrap-dev", "rollout", "submit", "test", "portforward")]
    [string]$Action = "bootstrap",
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [string]$Manifest = "rancherConfigs/hybrid-stack.yaml",
    [string]$DevManifest = "rancherConfigs/fake-slurm-dev.yaml",
    [string]$BatchSize = "",
    [string]$MlflowTrackingUri = "",
    [string]$GpuSampleIntervalSec = "",
    [string]$ElectricityPriceEurPerKwh = "",
    [string]$GridCo2KgPerKwh = "",
    [string]$PueFactor = "",
    [string]$MlflowLogJobEnergy = "",
    [string]$MlflowJobEnergyExperiment = "",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"

function Invoke-Kubectl {
    param([Parameter(Mandatory = $true)][string[]]$Args)
    & kubectl @Args
    if ($LASTEXITCODE -ne 0) { throw "kubectl failed: kubectl $($Args -join ' ')" }
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
    if ($LASTEXITCODE -ne 0) { throw "kubectl failed while checking namespace '$Namespace' using kubeconfig '$Kubeconfig'" }
    if (-not $ns) { Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "create", "namespace", $Namespace) }
}

function Ensure-GhcrSecret {
    if (-not $env:CR_PAT) { throw "CR_PAT missing. Put CR_PAT in .env or env var." }
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "secret", "ghcr-pull-secret", "--ignore-not-found")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "secret", "docker-registry", "ghcr-pull-secret", "--docker-server=ghcr.io", "--docker-username=its-ghaith", "--docker-password=$env:CR_PAT")
}

function Ensure-DevKeyPair {
    $keyDir = Join-Path (Get-Location) ".tmp/ssh"
    $keyPath = Join-Path $keyDir "fake_slurm_dev_ed25519"
    $pubPath = "$keyPath.pub"
    New-Item -ItemType Directory -Force -Path $keyDir | Out-Null
    if (-not (Test-Path $keyPath) -or -not (Test-Path $pubPath)) {
        & ssh-keygen --% -t ed25519 -f .tmp\ssh\fake_slurm_dev_ed25519 -N ""
        if ($LASTEXITCODE -ne 0) { throw "Failed to generate dev SSH keypair." }
    }
    return @{ KeyPath = $keyPath; PubPath = $pubPath }
}

function Ensure-FakeSlurmAuthSecret {
    param([string]$PublicKey)
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "secret", "fake-slurm-auth", "--ignore-not-found")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "secret", "generic", "fake-slurm-auth", "--from-literal=user=slurm", "--from-literal=password=devslurm123", "--from-literal=public_key=$PublicKey")
}

function Ensure-SlurmConnectorSecrets {
    if (-not $env:SLURM_SSH_HOST) { throw "SLURM_SSH_HOST missing in .env or env var." }
    if (-not $env:SLURM_SSH_USER) { throw "SLURM_SSH_USER missing in .env or env var." }

    $strictMode = if ($env:SSH_STRICT_HOST_KEY_CHECKING) { $env:SSH_STRICT_HOST_KEY_CHECKING } else { "yes" }
    if (($strictMode -ne "no") -and (-not $env:SLURM_SSH_KNOWN_HOSTS)) {
        throw "SLURM_SSH_KNOWN_HOSTS missing while SSH_STRICT_HOST_KEY_CHECKING is not 'no'."
    }

    $sshPort = if ($env:SLURM_SSH_PORT) { $env:SLURM_SSH_PORT } else { "22" }

    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "secret", "slurm-connector-target", "--ignore-not-found")
    $targetArgs = @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "secret", "generic", "slurm-connector-target", "--from-literal=host=$($env:SLURM_SSH_HOST)", "--from-literal=port=$sshPort", "--from-literal=user=$($env:SLURM_SSH_USER)")
    if ($env:SLURM_SSH_PASSWORD) { $targetArgs += "--from-literal=password=$($env:SLURM_SSH_PASSWORD)" }
    Invoke-Kubectl -Args $targetArgs

    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "delete", "secret", "slurm-connector-ssh", "--ignore-not-found")
    if ($env:SLURM_SSH_PRIVATE_KEY_FILE -and (Test-Path $env:SLURM_SSH_PRIVATE_KEY_FILE)) {
        $known = if ($env:SLURM_SSH_KNOWN_HOSTS) { $env:SLURM_SSH_KNOWN_HOSTS } else { "" }
        Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "secret", "generic", "slurm-connector-ssh", "--from-file=id_ed25519=$($env:SLURM_SSH_PRIVATE_KEY_FILE)", "--from-literal=known_hosts=$known")
    }
    else {
        if (-not $env:SLURM_SSH_PRIVATE_KEY) { throw "SLURM_SSH_PRIVATE_KEY or SLURM_SSH_PRIVATE_KEY_FILE missing." }
        $known = if ($env:SLURM_SSH_KNOWN_HOSTS) { $env:SLURM_SSH_KNOWN_HOSTS } else { "" }
        Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "create", "secret", "generic", "slurm-connector-ssh", "--from-literal=id_ed25519=$($env:SLURM_SSH_PRIVATE_KEY)", "--from-literal=known_hosts=$known")
    }
}

function Apply-Stack { Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "apply", "-f", $Manifest) }
function Apply-DevFakeSlurm { Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "apply", "-f", $DevManifest) }

function Update-ConnectorConfig {
    $remoteProjectDir = if ($env:REMOTE_PROJECT_DIR) { $env:REMOTE_PROJECT_DIR } else { "/home/slurm/mlops-project" }
    $remoteSlurmScript = if ($env:REMOTE_SLURM_SCRIPT) { $env:REMOTE_SLURM_SCRIPT } else { "slurm/train_distributed_mlflow_job.slurm" }
    $remoteMlflowUri = if ($env:REMOTE_MLFLOW_TRACKING_URI) { $env:REMOTE_MLFLOW_TRACKING_URI } else { "http://mlflow.mlops-energy.svc.cluster.local:5000" }
    $enableReverseTunnel = if ($env:ENABLE_REVERSE_TUNNEL) { $env:ENABLE_REVERSE_TUNNEL } else { "0" }
    $reverseTunnelBind = if ($env:REVERSE_TUNNEL_BIND) { $env:REVERSE_TUNNEL_BIND } else { "5000:mlflow.mlops-energy.svc.cluster.local:5000" }
    $strictMode = if ($env:SSH_STRICT_HOST_KEY_CHECKING) { $env:SSH_STRICT_HOST_KEY_CHECKING } else { "yes" }

    $patch = @{
        data = @{
            REMOTE_PROJECT_DIR = $remoteProjectDir
            REMOTE_SLURM_SCRIPT = $remoteSlurmScript
            REMOTE_MLFLOW_TRACKING_URI = $remoteMlflowUri
            ENABLE_REVERSE_TUNNEL = $enableReverseTunnel
            REVERSE_TUNNEL_BIND = $reverseTunnelBind
            SSH_STRICT_HOST_KEY_CHECKING = $strictMode
        }
    } | ConvertTo-Json -Compress
    $patchFile = Join-Path $env:TEMP "slurm-connector-config-patch.json"
    Set-Content -Path $patchFile -Value $patch -Encoding ASCII
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "patch", "configmap", "slurm-connector-config", "--type=merge", "--patch-file", $patchFile)
}

function Restart-And-Wait {
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "restart", "deploy/mlflow", "deploy/prometheus", "deploy/grafana", "deploy/slurm-connector", "deploy/pushgateway")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/mlflow", "--timeout=300s")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/prometheus", "--timeout=300s")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/grafana", "--timeout=300s")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/slurm-connector", "--timeout=300s")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/pushgateway", "--timeout=300s")
}

function Wait-FakeSlurm {
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "rollout", "status", "deploy/fake-slurm-dev", "--timeout=300s")
}

function Seed-FakeSlurmAuthorizedKey {
    param([string]$PublicKey)
    $cmd = "mkdir -p /config/.ssh /home/slurm/.ssh && printf '%s\n' '$PublicKey' > /config/.ssh/authorized_keys && printf '%s\n' '$PublicKey' > /home/slurm/.ssh/authorized_keys && chmod 600 /config/.ssh/authorized_keys /home/slurm/.ssh/authorized_keys && chown -R slurm:users /config/.ssh /home/slurm/.ssh"
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/fake-slurm-dev", "--", "sh", "-lc", $cmd)
}

function Submit-Job {
    $pairs = @()
    if ($BatchSize) { $pairs += "BATCH_SIZE='$BatchSize'" }
    if ($MlflowTrackingUri) { $pairs += "REMOTE_MLFLOW_TRACKING_URI='$MlflowTrackingUri'" }
    if ($GpuSampleIntervalSec) { $pairs += "GPU_SAMPLE_INTERVAL_SEC='$GpuSampleIntervalSec'" }
    if ($ElectricityPriceEurPerKwh) { $pairs += "ELECTRICITY_PRICE_EUR_PER_KWH='$ElectricityPriceEurPerKwh'" }
    if ($GridCo2KgPerKwh) { $pairs += "GRID_CO2_KG_PER_KWH='$GridCo2KgPerKwh'" }
    if ($PueFactor) { $pairs += "PUE_FACTOR='$PueFactor'" }
    if ($MlflowLogJobEnergy) { $pairs += "MLFLOW_LOG_JOB_ENERGY='$MlflowLogJobEnergy'" }
    if ($MlflowJobEnergyExperiment) { $pairs += "MLFLOW_JOB_ENERGY_EXPERIMENT='$MlflowJobEnergyExperiment'" }
    $cmd = ($pairs -join " ") + " /opt/slurm-connector/submit_remote_slurm.sh"
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurm-connector", "--", "bash", "-lc", $cmd)
}

function Run-Test {
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "get", "pods", "-o", "wide")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/slurm-connector", "--", "bash", "/opt/slurm-connector/test_remote_slurm.sh")
    Invoke-Kubectl -Args @("--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", "deploy/prometheus", "--", "sh", "-lc", "wget -qO- 'http://localhost:9090/api/v1/query?query=up'")
}

function Configure-DevEnvForFakeSlurm {
    $kp = Ensure-DevKeyPair
    $pub = (Get-Content $kp.PubPath -Raw).Trim()
    Ensure-FakeSlurmAuthSecret -PublicKey $pub

    Set-Item -Path Env:SLURM_SSH_HOST -Value "fake-slurm-dev.mlops-energy.svc.cluster.local"
    Set-Item -Path Env:SLURM_SSH_PORT -Value "2222"
    Set-Item -Path Env:SLURM_SSH_USER -Value "slurm"
    Set-Item -Path Env:SLURM_SSH_PRIVATE_KEY_FILE -Value $kp.KeyPath
    Set-Item -Path Env:SLURM_SSH_PASSWORD -Value "devslurm123"
    Set-Item -Path Env:SLURM_SSH_KNOWN_HOSTS -Value ""
    Set-Item -Path Env:SSH_STRICT_HOST_KEY_CHECKING -Value "no"
    Set-Item -Path Env:REMOTE_PROJECT_DIR -Value "/home/slurm"
    Set-Item -Path Env:REMOTE_SLURM_SCRIPT -Value "slurm/train_distributed_mlflow_job.slurm"
    Set-Item -Path Env:REMOTE_MLFLOW_TRACKING_URI -Value "http://mlflow.mlops-energy.svc.cluster.local:5000"
    return $pub
}

function Show-PortForward {
    Write-Host "Run in separate terminals:" -ForegroundColor Yellow
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/grafana 3000:3000"
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/mlflow 5000:5000"
    Write-Host "kubectl --kubeconfig $Kubeconfig -n $Namespace port-forward svc/prometheus 9090:9090"
}

Import-DotEnv -Path $EnvFile
if ($PSBoundParameters.ContainsKey("Action") -eq $false -and $env:ACTION) { $Action = $env:ACTION }
if ($PSBoundParameters.ContainsKey("Kubeconfig") -eq $false -and $env:KUBECONFIG_PATH) { $Kubeconfig = $env:KUBECONFIG_PATH }
if ($PSBoundParameters.ContainsKey("Namespace") -eq $false -and $env:K8S_NAMESPACE) { $Namespace = $env:K8S_NAMESPACE }
if ($PSBoundParameters.ContainsKey("Manifest") -eq $false -and $env:K8S_MANIFEST) { $Manifest = $env:K8S_MANIFEST }
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
        Ensure-SlurmConnectorSecrets
        Apply-Stack
        Update-ConnectorConfig
        Restart-And-Wait
        Run-Test
    }
    "bootstrap-dev" {
        Ensure-Namespace
        Ensure-GhcrSecret
        $devPubKey = Configure-DevEnvForFakeSlurm
        Ensure-SlurmConnectorSecrets
        Apply-DevFakeSlurm
        Wait-FakeSlurm
        Seed-FakeSlurmAuthorizedKey -PublicKey $devPubKey
        Apply-Stack
        Update-ConnectorConfig
        Restart-And-Wait
        Run-Test
    }
    "rollout" {
        Update-ConnectorConfig
        Restart-And-Wait
    }
    "submit" { Submit-Job }
    "test" { Run-Test }
    "portforward" { Show-PortForward }
}
