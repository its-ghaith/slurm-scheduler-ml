from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.preference_conditioned_rapec_v3_pf import (
    PreferenceConditionedRapecV3PfController,
)


ROOT = Path(__file__).resolve().parents[2]


def _parameters(**overrides):
    document = json.loads(
        (
            ROOT
            / "controller_benchmark/config/rapec-v3-pf-preference-controller.json"
        ).read_text(encoding="utf-8")
    )
    values = dict(document["controllers"][0]["controller_parameters"])
    values.update(overrides)
    return values


def _context(
    *,
    task_type="unknown",
    scenario="unknown",
    model="unknown",
    **parameters,
):
    return ControllerContext(
        controller_id="rapec-v3-pf-preference-test",
        benchmark_version="preference-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters=_parameters(**parameters),
        metadata={"model_version": model},
    )


def _row(epoch: int, quality: float, *, loss: float | None = None) -> dict:
    row = {
        "epoch": epoch,
        "epoch_index": epoch,
        "quality_score": quality,
        "epoch_energy_wh": 1.0,
        "duration_seconds": 2.0,
        "gpu_util_avg_pct": 60.0,
        "gpu_power_avg_w": 150.0,
        "gpu_mem_used_avg_mb": 2048.0,
        "learning_rate": 0.001,
        "gradient_norm": 0.1,
        "model_parameter_count": 3_000_000,
        "model_flops": 8_000_000_000,
    }
    if loss is not None:
        row["train_loss"] = loss
    return row


def _observation(context, history: list[dict], current: dict):
    rows = [*history, current]
    qualities = [float(row["quality_score"]) for row in rows]
    epoch = int(current["epoch_index"])
    return EpochObservation(
        epoch=epoch,
        max_epochs=context.max_epochs,
        task_type=context.task_type,
        quality_metric=context.quality_metric,
        quality=qualities[-1],
        best_quality=max(qualities),
        delta_quality=(qualities[-1] - qualities[-2]) if len(qualities) > 1 else None,
        epoch_energy_wh=float(current["epoch_energy_wh"]),
        cumulative_energy_wh=sum(float(row["epoch_energy_wh"]) for row in rows),
        epoch_duration_seconds=float(current["duration_seconds"]),
        cumulative_duration_seconds=sum(float(row["duration_seconds"]) for row in rows),
        gpu_utilization_pct=float(current["gpu_util_avg_pct"]),
        history=tuple(history),
        raw_metrics=current,
    )


def _history(qualities: list[float]) -> list[dict]:
    return [
        _row(epoch, quality, loss=max(0.05, 1.5 - quality))
        for epoch, quality in enumerate(qualities, start=1)
    ]


def _profile(controller, context, rows):
    return controller._preference_profile(
        rows,
        _observation(context, rows[:-1], rows[-1]),
    )


class PreferenceConditionedRapecV3PfTests(unittest.TestCase):
    def test_shared_rapec_v3_parameters_are_identical(self):
        v3_document = json.loads(
            (
                ROOT / "controller_benchmark/config/rapec-v3-v8-controllers.json"
            ).read_text(encoding="utf-8")
        )
        v3 = v3_document["controllers"][0]["controller_parameters"]
        preference = _parameters()
        self.assertEqual({key: preference[key] for key in v3}, v3)

    def test_only_five_epoch_confirmation_is_used(self):
        context = _context()
        controller = PreferenceConditionedRapecV3PfController(context)
        self.assertEqual(controller._confirmation_horizons(), (5,))
        observation = _observation(context, [], _row(1, 0.1))
        self.assertEqual(controller._structural_history_floor(observation), 24)

    def test_energy_priority_monotonically_increases_aggressiveness(self):
        rows = _history([min(0.8, 0.04 * epoch) for epoch in range(1, 26)])
        low_context = _context(energy_priority_level=1, task_difficulty_level=5)
        high_context = _context(energy_priority_level=10, task_difficulty_level=5)
        low = _profile(
            PreferenceConditionedRapecV3PfController(low_context),
            low_context,
            rows,
        )
        high = _profile(
            PreferenceConditionedRapecV3PfController(high_context),
            high_context,
            rows,
        )
        self.assertGreater(high.aggressiveness, low.aggressiveness)
        self.assertLess(high.significance_z, low.significance_z)
        self.assertGreater(
            high.low_progress_quantile, low.low_progress_quantile
        )
        self.assertLessEqual(
            high.required_low_progress_windows,
            low.required_low_progress_windows,
        )
        self.assertLess(high.no_learning_multiplier, low.no_learning_multiplier)

    def test_higher_difficulty_is_more_conservative(self):
        rows = _history([min(0.7, 0.025 * epoch) for epoch in range(1, 26)])
        easy_context = _context(energy_priority_level=7, task_difficulty_level=1)
        hard_context = _context(energy_priority_level=7, task_difficulty_level=10)
        easy = _profile(
            PreferenceConditionedRapecV3PfController(easy_context),
            easy_context,
            rows,
        )
        hard = _profile(
            PreferenceConditionedRapecV3PfController(hard_context),
            hard_context,
            rows,
        )
        self.assertLess(hard.aggressiveness, easy.aggressiveness)
        self.assertGreater(hard.significance_z, easy.significance_z)
        self.assertGreater(
            hard.no_learning_multiplier, easy.no_learning_multiplier
        )

    def test_all_ten_by_ten_levels_are_monotonic(self):
        rows = _history([min(0.75, 0.03 * epoch) for epoch in range(1, 26)])
        grid = {}
        for energy_level in range(1, 11):
            for difficulty_level in range(1, 11):
                context = _context(
                    energy_priority_level=energy_level,
                    task_difficulty_level=difficulty_level,
                )
                grid[energy_level, difficulty_level] = _profile(
                    PreferenceConditionedRapecV3PfController(context),
                    context,
                    rows,
                ).aggressiveness
        for difficulty_level in range(1, 11):
            values = [
                grid[energy_level, difficulty_level]
                for energy_level in range(1, 11)
            ]
            self.assertEqual(values, sorted(values))
        for energy_level in range(1, 11):
            values = [
                grid[energy_level, difficulty_level]
                for difficulty_level in range(1, 11)
            ]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_optional_difficulty_uses_online_estimate(self):
        easy_rows = _history([min(0.95, 0.06 * epoch) for epoch in range(1, 21)])
        hard_rows = _history([0.02 + 0.003 * epoch for epoch in range(1, 21)])
        easy_context = _context(task_difficulty_level=None)
        hard_context = _context(task_difficulty_level=None)
        easy = _profile(
            PreferenceConditionedRapecV3PfController(easy_context),
            easy_context,
            easy_rows,
        )
        hard = _profile(
            PreferenceConditionedRapecV3PfController(hard_context),
            hard_context,
            hard_rows,
        )
        self.assertEqual(easy.difficulty_prior_weight, 0.0)
        self.assertEqual(hard.difficulty_prior_weight, 0.0)
        self.assertGreater(
            hard.automatic_difficulty, easy.automatic_difficulty
        )

    def test_manual_difficulty_is_a_prior_updated_by_run_data(self):
        rows = _history([min(0.9, 0.05 * epoch) for epoch in range(1, 31)])
        context = _context(task_difficulty_level=10)
        controller = PreferenceConditionedRapecV3PfController(context)
        profile = _profile(controller, context, rows)
        self.assertGreater(profile.difficulty_prior_weight, 0.35)
        self.assertLess(profile.difficulty_prior_weight, 1.0)
        self.assertLess(profile.effective_difficulty, 1.0)

    def test_invalid_levels_are_rejected(self):
        with self.assertRaises(ValueError):
            PreferenceConditionedRapecV3PfController(
                _context(energy_priority_level=11)
            )
        with self.assertRaises(ValueError):
            PreferenceConditionedRapecV3PfController(
                _context(task_difficulty_level=-1)
            )

    def test_source_does_not_read_task_scenario_or_model_profiles(self):
        source = inspect.getsource(PreferenceConditionedRapecV3PfController)
        for forbidden in (
            "context.task_type",
            "context.scenario",
            "context.metadata",
            '"scratch"',
            '"pretrained"',
        ):
            self.assertNotIn(forbidden, source)

    def test_identical_traces_ignore_task_and_dataset_labels(self):
        contexts = [
            _context(task_type="object_detection", scenario="dataset-a", model="a"),
            _context(task_type="image_classification", scenario="dataset-b", model="b"),
            _context(task_type="semantic_segmentation", scenario="dataset-c", model="c"),
        ]
        qualities = [
            min(0.72, 0.04 * epoch) if epoch <= 18 else 0.72
            for epoch in range(1, 35)
        ]
        runs = []
        for context in contexts:
            controller = PreferenceConditionedRapecV3PfController(context)
            history = []
            decisions = []
            for row in _history(qualities):
                decisions.append(
                    controller.evaluate(_observation(context, history, row))
                )
                history.append(row)
            runs.append(decisions)
        for decisions in zip(*runs):
            first = decisions[0]
            for decision in decisions[1:]:
                self.assertEqual(decision.stop, first.stop)
                self.assertEqual(decision.reason, first.reason)
                self.assertEqual(decision.diagnostics, first.diagnostics)


if __name__ == "__main__":
    unittest.main()
