from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from controller_benchmark.status_web import server


class StatusWebTests(unittest.TestCase):
    def _write_timeline(self, root: Path, rows: list[dict]) -> None:
        metrics = root / "run-1" / "metrics"
        metrics.mkdir(parents=True)
        timeline = metrics / "epoch_timeline_job_2.jsonl"
        timeline.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    def test_last_epoch_reads_current_direct_epoch_format(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_timeline(
                root,
                [
                    {"epoch": 51, "quality_score": 0.44},
                    {"epoch": 52, "quality_score": 0.45},
                ],
            )
            with patch.object(server, "ROOT", root):
                latest = server.last_epoch("run-1", "2")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["epoch"], 52)

    def test_last_epoch_remains_compatible_with_legacy_end_events(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_timeline(
                root,
                [
                    {"event": "start", "epoch": 1},
                    {"event": "end", "epoch": 1, "quality_score": 0.25},
                ],
            )
            with patch.object(server, "ROOT", root):
                latest = server.last_epoch("run-1", "2")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["epoch"], 1)

    def test_pretraining_status_reads_mirrored_log(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "current.out"
            log.write_text(
                json.dumps(
                    {
                        "source_case": "ettm1-time-series-forecasting-scratch",
                        "checkpoint": 10,
                        "pretraining_checkpoints": 20,
                        "raw_quality_metric": "nrmse",
                        "raw_quality_value": 0.138,
                        "quality_score": 0.878,
                        "train_loss": 0.004,
                        "learning_rate": 0.00058,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(server, "PRETRAINING_ROOT", root):
                text = server.pretraining_status_text()
        self.assertIn("ettm1-time-series-forecasting-scratch", text)
        self.assertIn("10/20", text)

    def test_latest_pretraining_log_supports_hardest_suite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "hardest-current.out"
            log.write_text('{"checkpoint": 1}\n', encoding="utf-8")
            with patch.object(server, "PRETRAINING_ROOT", root):
                selected = server.latest_pretraining_log()
        self.assertEqual(selected, log)

    def test_offline_replay_uses_run_specific_report_first(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            central = root / "offline-replays"
            central.mkdir(parents=True)
            (central / "rapec-g-v5-48-offline-replay.txt").write_text(
                "central report\n",
                encoding="utf-8",
            )
            run = root / "run-1"
            run.mkdir()
            (run / "rapec-g-v5-48-offline-replay.txt").write_text(
                "run report\n",
                encoding="utf-8",
            )
            with (
                patch.object(server, "ROOT", root),
                patch.object(server, "OFFLINE_REPLAY_ROOT", central),
            ):
                text = server.offline_replay_results_text("run-1")
        self.assertEqual(text, "run report")

    def test_offline_replay_reports_missing_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(server, "ROOT", root),
                patch.object(server, "OFFLINE_REPLAY_ROOT", root / "missing"),
            ):
                text = server.offline_replay_results_text("run-1")
        self.assertIn("noch kein Bericht", text)

    def test_html_uses_persistent_refresh_toggle_instead_of_meta_refresh(self) -> None:
        source = inspect.getsource(server.Handler.do_GET)
        self.assertIn('id=\\\"refresh-toggle\\\"', source)
        self.assertIn("localStorage.setItem", source)
        self.assertNotIn('http-equiv=\\\"refresh\\\"', source)


if __name__ == "__main__":
    unittest.main()
