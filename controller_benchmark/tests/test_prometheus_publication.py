from __future__ import annotations

import unittest

from controller_benchmark.server.orchestrate import (
    normalize_prometheus_metric_types,
    prometheus_metric_types,
)


class PrometheusPublicationTests(unittest.TestCase):
    def test_metric_types_are_parsed(self) -> None:
        exposition = "# TYPE energy gauge\nenergy 1\n# TYPE quality untyped\nquality 2\n"
        self.assertEqual(
            prometheus_metric_types(exposition),
            {"energy": "gauge", "quality": "untyped"},
        )

    def test_existing_types_replace_or_complete_payload_metadata(self) -> None:
        payload = (
            "# HELP energy Energy\n"
            "# TYPE energy gauge\n"
            "energy{job_id=\"6\"} 1\n"
            "# HELP quality Quality\n"
            "quality{job_id=\"6\"} 2\n"
            "# TYPE new_metric counter\n"
            "new_metric 3\n"
        )

        normalized = normalize_prometheus_metric_types(
            payload, {"energy": "untyped", "quality": "gauge"}
        )

        self.assertIn("# TYPE energy untyped\n", normalized)
        self.assertIn("# TYPE quality gauge\nquality", normalized)
        self.assertIn("# TYPE new_metric counter\n", normalized)
        self.assertEqual(normalized.count("# TYPE energy "), 1)


if __name__ == "__main__":
    unittest.main()
