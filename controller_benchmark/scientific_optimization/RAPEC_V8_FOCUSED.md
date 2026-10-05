# Focused scientific optimization for RAPEC-v8

This workflow converts the focused Sobol-ALE sensitivity result into a constrained
multi-objective parameter search. Previous runs and immutable Full100 traces are
read-only inputs.

## Active parameters

- `min_epochs_floor`
- `utility_quantile`
- `noise_multiplier`
- `minimum_quality_floor`

`trend_window` remains fixed at the RAPEC-v8 base value `10`. Its overall Sobol
total-order indices were approximately `0.012` to `0.016` for quality regret,
energy saving, stop epoch, and joint success.

## Scientific method

1. Verify the immutable nine-dataset Full100 source bundle by checksum.
2. Replay RAPEC-v8 on every Full100 epoch trace.
3. Use Morris screening as a consistency check; all four sensitivity-supported
   parameters remain mandatory.
4. Optimize quality CVaR90 and lower-quartile energy saving with constrained
   qLogNEHVI.
5. Require bootstrap non-inferiority for every dataset group.
6. Choose one feasible Pareto point with Nash bargaining.
7. Evaluate task transfer using leave-one-task-type-out outer folds.
8. Confirm the selected controller with fresh paired Rancher runs.

The optimizer uses only training/epoch GPU energy because this is the energy that
the online controller can observe and influence directly.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\run-rapec-v8-focused-scientific-optimization.ps1 `
  -StartInBackground `
  -SkipDependencyInstall
```

The result is written below a new immutable directory:

```text
results/controller-v8-scientific-optimization/runs/<optimization-id>/
```

## Watch

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\watch-rapec-v8-focused-scientific-optimization.ps1 `
  -OptimizationId <optimization-id>
```

## Validate

Inspect the Rancher plan first:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v8-focused-scientific-validation-async.ps1 `
  -OptimizationDir .\results\controller-v8-scientific-optimization\runs\<optimization-id> `
  -PlanOnly
```

For the final result, use fresh paired seeds instead of only replaying the old
Full100 trajectories:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v8-focused-scientific-validation-async.ps1 `
  -OptimizationDir .\results\controller-v8-scientific-optimization\runs\<optimization-id> `
  -FreshSeeds 1,2,3
```
