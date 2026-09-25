"""Print provider-reported usage for durable bounded-search attempts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from src.agent.cost_estimate import price_usage_report
from src.agent.provider_usage import optimizer_usage_report, usage_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=settings.compiled_dir,
                        help="Compiled artifact root, including search_cache and evaluations")
    parser.add_argument("--cache-only", action="store_true", help="Treat --root as a legacy search-cache root")
    args = parser.parse_args()
    report = usage_report(args.root) if args.cache_only else optimizer_usage_report(args.root)
    report["advisory_cost"] = price_usage_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
