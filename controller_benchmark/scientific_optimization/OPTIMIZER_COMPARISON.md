# RAPEC-v9 Optimizer Comparison

This workflow compares three parameter-search methods without changing any
existing controller, Full100 source, optimization result, or Rancher run:

- Sobol search as a model-free baseline,
- the existing constrained qLogNEHVI implementation,
- constraint-first random-forest SMBO with ParEGO scalarization.

The comparison is offline. It replays RAPEC-v9 on checksum-verified Full100
epoch curves and submits no Rancher jobs.

## Scientific objectives

For dataset group `d`, controller parameters `theta`, Full100 quality `Q*`,
stopped quality `Q(theta)`, dynamic tolerance `delta_d`, and training energy
`E`, the workflow uses:

```text
normalized_regret_d = max(0, Q*_d - Q_d(theta)) / delta_d
energy_saving_d = 1 - E_d(theta) / E_d(Full100)
```

The two optimization objectives are:

```text
minimize CVaR90(normalized_regret_d)
maximize Q25(energy_saving_d)
```

Quality is a hard constraint. A configuration is feasible only if every
dataset group reaches the configured non-inferiority probability. Quality is
never traded for energy, and no fixed percentage-point tolerance is added.

With only one Full100 seed per dataset, the bootstrap probability is binary.
The generated `oracle-upper-bound.json` records this limitation explicitly.
Fresh repeated training seeds are required for a probabilistic scientific
claim.

## RF-ParEGO backend

The RF backend uses the exact decoded mixed parameter configuration. Integer
levels are canonicalized before model fitting, and floats are not rounded.
This avoids evaluating multiple continuous vectors that decode to the same
integer configuration.

At each sequential search step it:

1. draws random ParEGO weights,
2. normalizes quality risk and negative energy saving,
3. computes an augmented Tchebycheff scalarization,
4. fits a random forest to the scalarized objective,
5. fits a separate random forest to continuous quality-constraint violation,
6. minimizes expected violation until the first feasible point exists,
7. then maximizes constrained expected improvement,
8. retains a small random-design probability for global exploration.

The selected controller is the maximum robust-energy configuration after hard
quality filtering. There is no Nash trade-off and no knee-point selection.
`scientific_selection_ready` becomes true only when the representative
final-fit configuration is feasible and the winning optimizer was feasible in
every outer task-type holdout. A lower rate remains visible and must not be
reported as evidence of generalisation.

## Removed unnecessary work

All seven RAPEC-v9 search parameters are mandatory. Morris screening therefore
cannot reduce the search dimension. The optimizer comparison skips Morris in
every fold. Morris remains useful as a separate sensitivity analysis, but it is
not repeated as part of parameter search.

## Persistent replay cache

Every replay is cached by:

- complete Full100 trace checksum,
- controller plugin source checksum,
- fixed evaluation controller id,
- exact decoded parameters.

The cache is shared across backends, folds, and optimizer seeds. It uses atomic
writes and never modifies the immutable source bundle. A persistent worker pool
is reused across trials so Windows does not start a new Python process for every
parameter evaluation.

## Oracle diagnosis

The Oracle report finds the earliest Full100 epoch whose best-so-far quality is
within the dynamic tolerance of the Full100 best quality. Its energy saving is
a non-causal upper bound.

- A high Oracle saving and low optimizer saving indicate a search problem.
- A high Oracle saving but consistently late controller decisions indicate a
  controller-architecture problem.
- A low Oracle saving means the requested quality constraint leaves little
  avoidable training energy.

## Plan the comparison

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v9-optimizer-comparison.ps1 `
  -PlanOnly
```

The plan must show three task-type folds, six training datasets and three
holdout datasets per fold, and zero Rancher jobs.

## Pilot comparison

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v9-optimizer-comparison.ps1 `
  -OptimizerSeeds "0,1,2" `
  -Trials 40 `
  -TrialsPerFold 40 `
  -StartInBackground
```

## Full comparison

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v9-optimizer-comparison.ps1 `
  -OptimizerSeeds "0,1,2,3,4" `
  -Trials 100 `
  -TrialsPerFold 100 `
  -StartInBackground
```

Five paired optimizer seeds permit a paired Wilcoxon signed-rank comparison.
The report reminds the researcher to apply a Holm correction when several
method pairs are tested.

Watch progress:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v9-optimizer-comparison.ps1 `
  -BenchmarkId <benchmark-id>
```

## Outputs

Each run is stored below:

```text
results/controller-optimizer-comparison/runs/<benchmark-id>/
```

Important files are:

- `plan.json`: immutable experimental plan,
- `oracle-upper-bound.json`: causal-limit diagnosis,
- `run-results.json`: every method, seed, and fold,
- `method-summary.json`: robust method-level comparison,
- `paired-statistics.json`: paired optimizer-seed tests,
- `selected-controller.json`: representative feasible controller,
- `optimizer-benchmark-manifest.json`: sources, assumptions, and cache audit.

## Fresh Rancher validation

Optimizer seeds measure search stability; they do not replace training seeds.
After choosing the method, plan fresh paired Full100/controller runs:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v9-scientific-validation-async.ps1 `
  -OptimizationDir .\results\controller-optimizer-comparison\runs\<benchmark-id> `
  -FreshSeeds "1,2,3,4,5" `
  -PlanOnly
```

Remove `-PlanOnly` only after checking the complete paired job matrix. Rancher
continues independently after queueing, so the local computer can then be
turned off.

## Interpretation boundary

Outer task-type holdouts compare optimizer methods. They are not used as proof
that the selected controller generalizes. The final claim must use fresh
Rancher runs that were not part of parameter selection.
