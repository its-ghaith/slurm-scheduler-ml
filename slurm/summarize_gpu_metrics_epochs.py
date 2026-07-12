import argparse
import json
from pathlib import Path

from epoch_energy_controller import build_epoch_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch-timeline-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--price-eur-kwh", type=float, default=0.30)
    p.add_argument("--co2-kg-kwh", type=float, default=0.4)
    p.add_argument("--adaptive-enabled", action="store_true")
    p.add_argument("--controller-mode", choices=["none", "delta_mape", "uncertainty_aware"], default="none")
    p.add_argument("--comparison-strategy", default="unspecified")
    p.add_argument("--adaptive-monitor-metric", choices=["map50", "map50_95"], default="map50")
    return p.parse_args()


def main():
    args = parse_args()
    summary = build_epoch_summary(
        epoch_timeline_path=Path(args.epoch_timeline_jsonl),
        output_path=Path(args.output_json),
        job_id=args.job_id,
        price_eur_kwh=args.price_eur_kwh,
        co2_kg_kwh=args.co2_kg_kwh,
        adaptive_enabled=args.adaptive_enabled,
        adaptive_monitor_metric=args.adaptive_monitor_metric,
        controller_mode=args.controller_mode,
        comparison_strategy=args.comparison_strategy,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
