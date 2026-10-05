[CmdletBinding()]
param(
    [string]$Kubeconfig = "rancherConfigs/main.yaml",
    [string]$Namespace = "mlops-energy",
    [ValidateSet("Standard", "Hard", "Expensive", "All")]
    [string]$Suite = "Standard"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if (-not [IO.Path]::IsPathRooted($Kubeconfig)) {
    $Kubeconfig = Join-Path $RepoRoot $Kubeconfig
}
$pod = ((& kubectl --kubeconfig $Kubeconfig -n $Namespace get pod -l app=slurmd -o "jsonpath={.items[?(@.status.phase=='Running')].metadata.name}") -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $pod) { throw "No running slurmd pod found." }

$remote = "/tmp/rapec-cross-domain-assets.py"
& kubectl --kubeconfig $Kubeconfig -n $Namespace cp `
    "controller_benchmark/runners/cross_domain_assets.py" `
    "${pod}:$remote" `
    -c slurmd
if ($LASTEXITCODE -ne 0) { throw "Could not upload the asset preparation script." }

$standardAssets = @(
    "adult", "covertype", "bank_marketing", "california_housing", "diabetes",
    "bike_sharing", "kddcup99", "breast_cancer", "wine", "imdb", "ag_news",
    "dbpedia", "wikitext2", "penn_treebank", "tiny_shakespeare", "etth1",
    "etth2", "ettm1", "movielens100k", "movielens1m", "movielens10m",
    "cartpole", "cartpole_heavy", "mountaincar"
)
$hardAssets = @(
    "covertype", "sensorless_drive", "letter_recognition", "year_prediction_msd",
    "online_news_popularity", "superconductivity", "kddcup99", "covertype_anomaly",
    "sensorless_drive_anomaly", "imdb", "ag_news", "yelp_review_full", "wikitext103",
    "wikitext2", "penn_treebank", "etth1", "ettm1", "ettm2", "movielens1m",
    "movielens10m", "movielens20m", "cartpole_heavy", "cartpole_noisy", "mountaincar"
)
$expensiveAssets = @(
    "ag_news", "imdb", "yelp_review_full", "wikitext103", "wikitext2",
    "penn_treebank", "etth1", "ettm1", "ettm2", "movielens1m",
    "movielens10m", "movielens20m", "cartpole_heavy", "cartpole_noisy",
    "mountaincar"
)
$selectedAssets = if ($Suite -eq "Expensive") {
    $expensiveAssets
} elseif ($Suite -eq "Hard") {
    $hardAssets
} elseif ($Suite -eq "All") {
    @($standardAssets + $hardAssets + $expensiveAssets | Sort-Object -Unique)
} else {
    $standardAssets
}

Write-Host "Preparing $Suite cross-domain datasets in the persistent Rancher cache..." -ForegroundColor Cyan
$assetArguments = @(
    "--kubeconfig", $Kubeconfig, "-n", $Namespace, "exec", $pod, "-c", "slurmd", "--",
    "env", "CONTROLLER_DATASET_ROOT=/workspace-cache/controller-datasets",
    "python", $remote
) + $selectedAssets
& kubectl @assetArguments
if ($LASTEXITCODE -ne 0) { throw "Cross-domain dataset preparation failed." }
Write-Host "Cross-domain datasets are cached outside the measured training intervals." -ForegroundColor Green
