from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from controller_benchmark.prepare_cross_domain_pretrained import prepare


class PretrainedResumeTests(unittest.TestCase):
    def test_existing_checkpoint_is_preserved_in_rebuilt_catalog(self) -> None:
        case_id = "existing-source"
        manifest = {
            "execution": {"max_epochs": 100},
            "stages": [
                {
                    "enabled": True,
                    "cases": [
                        {
                            "id": case_id,
                            "runner": "cross_domain_task",
                            "pretrained": False,
                        }
                    ],
                }
            ],
        }
        checkpoint = {
            "schema_version": 1,
            "model_state": {"weight": torch.tensor([1.0])},
            "source_case_id": case_id,
            "task_type": "test",
            "pretraining_checkpoints": 20,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            output_root = root / "checkpoints"
            output_root.mkdir()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            torch.save(checkpoint, output_root / f"{case_id}.pt")

            prepare(manifest_path, output_root, 20, force=False)

            catalog = json.loads((output_root / "catalog.json").read_text(encoding="utf-8"))
            self.assertEqual([case_id], [row["source_case_id"] for row in catalog["checkpoints"]])
            self.assertTrue((output_root / f"{case_id}.pt").exists())


if __name__ == "__main__":
    unittest.main()
