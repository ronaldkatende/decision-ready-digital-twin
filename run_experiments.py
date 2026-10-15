from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback

from download_benchmarks import ensure_required_data
from dtwinexp.config import get_config, OUT, FIG, TAB, LOG
from dtwinexp.domains import build_domain
from dtwinexp.experiments import run_domain_suite
from dtwinexp.reporting import collect_tables, write_reports
from dtwinexp.plotting import make_all


def reset_results() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for p in [OUT, FIG, TAB, LOG]:
        p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the decision-ready digital-twin reproducibility suite."
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--domains",
        nargs="*",
        choices=["power", "water", "structure", "thermal"],
        default=None,
    )
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Do not clear an existing results directory before the run.",
    )
    args = parser.parse_args()

    cfg = get_config(args.mode)
    domains = args.domains if args.domains else list(cfg.domains)

    if "water" in domains:
        ensure_required_data()
    if not args.keep_results:
        reset_results()

    results = {}
    failures = {}
    t0 = time.time()

    for name in domains:
        print(f"\n===== {name.upper()} =====")
        try:
            domain = build_domain(name, cfg)
            print(f"d={domain.d}, candidate sensing units={domain.m}")
            res = run_domain_suite(domain, cfg)
            results[name] = res
            print(f"completed in {res['runtime_seconds']/60:.2f} min")
        except Exception as exc:
            failures[name] = {"error": repr(exc), "traceback": traceback.format_exc()}
            print(f"FAILED: {name}: {exc}")
            (LOG / f"{name}_failure.txt").write_text(
                failures[name]["traceback"], encoding="utf-8"
            )

    if not results:
        print("No domain completed.", file=sys.stderr)
        return 2

    tables = collect_tables(results)
    make_all(tables)
    checks = write_reports(results, tables, cfg)

    status = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "total_runtime_seconds": time.time() - t0,
        "completed_domains": sorted(results),
        "failed_domains": sorted(failures),
        "output_directory": "results",
        "theorem_checks": checks,
    }
    (LOG / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    print("\n===== RUN COMPLETE =====")
    print(f"Report: {OUT / 'RUN_REPORT.md'}")
    print(f"Tables: {TAB}")
    print(f"Figures: {FIG}")

    if failures and args.mode == "full":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
