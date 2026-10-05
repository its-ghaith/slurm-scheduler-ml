# Scientific RAPEC-v3 Parameter Selection

This directory contains the versioned scientific parameter-selection workflow for
the task-independent RAPEC-v3 variant. It does not replace or modify the original
RAPEC-v3 controller or the previous NSGA-II optimization results.

## Method

1. Verify one or more immutable Full100 source bundles by SHA-256.
2. Derive a non-inferiority tolerance for each dataset from late-curve validation
   noise and, when available, variation across Full100 seeds.
3. Screen all controller parameters with Morris elementary effects.
4. Optimize the most influential parameters with constrained qLogNEHVI, the
   numerically stable logarithmic qNEHVI implementation.
5. Minimize the CVaR90 of normalized quality regret and maximize the lower-quartile
   relative training-energy saving.
6. Reject configurations whose bootstrap probability of non-inferiority is below
   0.95 for any dataset group.
7. Select one point from the feasible Pareto front using the Nash product.
8. Evaluate transfer with outer leave-one-task-type-out folds.
9. Confirm the selected parameters on fresh Rancher seeds before reporting them.

No ideal-stop epoch or stop-error metric is used.

## Metrics

For dataset `c`, seed `s`, and controller parameters `theta`:

```text
quality_regret = max(0, Full100_best_quality - stopped_best_quality)
energy_saving = 1 - stopped_training_energy / Full100_training_energy
```

Datasets receive equal weight even when they have different numbers of seed
replicates. Energy means training/epoch GPU energy because this is the energy scope
available to the online stop decision.

## Run optimization

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v3-scientific-optimization.ps1 `
  -SourceRunIds 20260729T170915Z-rapec-v3-v8-nine-dataset
```

The default output is created below:

```text
results/controller-scientific-optimization/runs/<optimization-id>/
```

An existing output directory is never overwritten.

The optimizer writes every Morris evaluation, Bayesian trial, outer fold, and
final selection immediately to both the console and:

```text
results/controller-scientific-optimization/runs/<optimization-id>/progress.jsonl
```

Watch a run from another PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v3-scientific-optimization.ps1 `
  -OptimizationId <optimization-id>
```

Use `-Once` for one status snapshot. Runs that were already active before this
logging was introduced can still be monitored by process runtime, CPU use, and
memory use, but cannot expose their exact internal trial retroactively.

Resume an interrupted run without overwriting completed search sections:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v3-scientific-optimization.ps1 `
  -OptimizationId <optimization-id> `
  -Resume
```

Resume accepts only completed search sections or empty interrupted sections. It
refuses to overwrite a nonempty incomplete search directory.

For a quick pipeline check that does not fit Gaussian processes:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v3-scientific-optimization.ps1 `
  -Backend sobol `
  -Trials 12 `
  -SensitivityTrajectories 2 `
  -SensitivityTopK 4 `
  -SkipOuterFolds
```

## Rancher validation

Reuse the checksum-verified Full100 baselines and submit only the selected
controller:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v3-scientific-validation-async.ps1 `
  -OptimizationDir .\results\controller-scientific-optimization\runs\<optimization-id>
```

For final scientific confirmation, create fresh paired Full100 and controller runs
for new seeds:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v3-scientific-validation-async.ps1 `
  -OptimizationDir .\results\controller-scientific-optimization\runs\<optimization-id> `
  -FreshSeeds 1,2,3
```

Use `-PlanOnly` first to inspect the complete Rancher job matrix without submitting
anything.

## Interpretation limits

Offline replay is a parameter-screening method. It assumes that training follows
the Full100 trajectory until the replayed stop epoch. It cannot measure controller
runtime overhead or prove performance on a new seed. Only the fresh Rancher
validation is used for the final scientific result.
