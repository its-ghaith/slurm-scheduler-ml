# RAPEC-v8 Sobol-ALE Sensitivity Analysis

This workflow quantifies how all 23 RAPEC-v8 parameters affect energy saving,
quality regret, stop epoch, non-inferiority, Bayesian vetoes, and controller
runtime. It is separate from the existing Morris and qLogNEHVI workflows.

## Method

1. Verify immutable Full100 source bundles by SHA-256.
2. Replay RAPEC-v8 using only information available up to each simulated epoch.
3. Evaluate a scrambled Sobol design across all nine benchmark datasets.
4. Fit an Extra-Trees multi-output surrogate.
5. Validate every surrogate output on an untouched holdout subset using R2 and
   MAE. Outputs with R2 below 0.75 are explicitly marked as low fidelity.
6. Calculate Sobol first-order, total-order, and second-order indices with 95%
   bootstrap confidence intervals.
7. Calculate first-order and centered second-order accumulated local effects.
8. Repeat all metrics overall and separately for object detection, image
   classification, and semantic segmentation.

The direct mode skips the surrogate for Sobol indices. A surrogate is still fit
for ALE visualizations and validated separately.

## Default run

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v8-sobol-sensitivity.ps1
```

The default surrogate mode evaluates 1,024 real replay configurations. It then
uses 49,152 inexpensive surrogate predictions for the full 23-parameter Sobol
design.

Run in the background:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v8-sobol-sensitivity.ps1 `
  -StartInBackground
```

The computer must remain powered on. This is a local offline analysis and does
not submit Rancher jobs.

Monitor it from another PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v8-sobol-sensitivity.ps1 `
  -AnalysisId <analysis-id>
```

Resume an interrupted analysis without overwriting completed replay evaluations:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v8-sobol-sensitivity.ps1 `
  -AnalysisId <analysis-id> `
  -Resume
```

## Fast pipeline check

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v8-sobol-sensitivity.ps1 `
  -AnalysisId rapec-v8-sobol-smoke `
  -DesignSamples 32 `
  -SobolBaseSamples 16 `
  -Trees 64 `
  -BootstrapResamples 32 `
  -TopParameters 2 `
  -TopInteractions 1 `
  -AleBins 4
```

## Outputs

Each new run is stored below:

```text
results/controller-v8-sensitivity/runs/<analysis-id>/
```

Important files:

- `SUMMARY.md`: readable ranking and surrogate validation.
- `parameter-ranking.csv`: combined numerical importance ranking.
- `sobol-output-rankings.csv`: separate ranking for every overall and task-specific output.
- `sobol-parameters.csv`: S1, ST, interaction gap, and confidence intervals.
- `sobol-interactions.csv`: pairwise S2 indices and confidence intervals.
- `ale-main-effects.csv`: direction and nonlinear shape of parameter effects.
- `ale-interactions.csv`: centered pairwise ALE surfaces.
- `surrogate-validation.csv`: holdout R2 and MAE for every output.
- `plots/`: Sobol rankings, interaction heatmaps, and ALE plots.
- `sensitivity-manifest.json`: method, source locks, limitations, and settings.

## Interpretation

Sobol indices describe variance within the declared parameter ranges. They do
not prove causality outside those ranges. A large difference between ST and S1
indicates that a parameter acts mainly through interactions. ALE values show the
direction of the effect around the average prediction.

Offline replay is appropriate for sensitivity analysis but cannot prove behavior
on a new random seed. Final controller claims still require fresh paired Rancher
runs against Full100.
