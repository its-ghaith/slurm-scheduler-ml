from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_snapshot, write_analysis, write_prometheus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_snapshot(args.snapshot, args.plan)
    output = args.snapshot / "controller_benchmark"
    write_analysis(result, output)
    write_prometheus(
        result,
        args.snapshot / "data" / "energy_metrics" / "node_exporter" / "controller_benchmark.prom",
    )
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
