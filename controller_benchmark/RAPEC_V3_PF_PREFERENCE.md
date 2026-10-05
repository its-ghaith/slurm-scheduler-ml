# RAPEC-v3-PF Preference Controller

This controller adds an explicit user preference and an optional task-
difficulty prior to the profile-free RAPEC-v3 warm-up gate. RAPEC-v3's shared
predictor, utility, uncertainty, quality guard, patience, and stop parameters
remain unchanged.

## Inputs

- `energy_priority_level`: integer `1..10`; `1` protects quality, `10`
  prioritizes energy saving.
- `task_difficulty_level`: optional integer `1..10`; omit it or pass `0` in
  PowerShell to estimate difficulty entirely from the current run.

The manual difficulty is a prior, not a permanent task profile. Its weight
decreases as current-run evidence accumulates but retains a configurable
minimum influence. No dataset, task, scenario, scratch/pretrained, or model
name is inspected.

## Online difficulty

The task-independent difficulty estimate combines four normalized current-run
signals:

```text
0.35 * quality attainment difficulty
+ 0.35 * learning-trajectory difficulty
+ 0.20 * validation noise relative to progress
+ 0.10 * relative training-loss persistence
```

Missing loss data uses a neutral value. Evidence confidence grows with the
number of observed epochs, and an exponential update prevents one noisy epoch
from changing the effective difficulty abruptly.

## Preference mapping

For normalized energy preference `e` and effective difficulty `d`:

```text
aggressiveness = 0.75 * e + 0.25 * (1 - d)
```

The score continuously interpolates only PF-specific warm-up controls between
a quality-protective and an energy-aggressive endpoint. It does not select one
of 100 hard-coded profiles. All confirmation uses exactly the latest five
epochs (`horizon_epochs=5`, multiplier `[1]`).

## Submission

Automatic difficulty and balanced preference:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v3-pf-preference-benchmark-async.ps1 `
  -EnergyPriority 5 `
  -RequireAllStages
```

Energy-oriented preference with a user difficulty prior:

```powershell
powershell -ExecutionPolicy Bypass -File .\controller_benchmark\scripts\submit-rapec-v3-pf-preference-benchmark-async.ps1 `
  -EnergyPriority 8 `
  -TaskDifficulty 7 `
  -RequireAllStages
```

Use `-PlanOnly` to inspect the run matrix without submitting Rancher jobs.
