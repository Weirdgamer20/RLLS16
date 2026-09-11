#!/usr/bin/env python3
"""
RLLS 16 — 3D Artificial-World Simulation System
Desktop Application Entry Point
"""

import sys
import argparse
from pathlib import Path

# Add project directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from rlls16.graphics.diagnostics import run_diagnostic_suite
from rlls16.app import launch
from rlls16.preview import preview


def main():
    parser = argparse.ArgumentParser(description="RLLS 16 — 3D Artificial-World Simulation System")
    parser.add_argument("--diagnostic", action="store_true", help="Run 10-stage graphics and data pipeline diagnostic suite")
    parser.add_argument("--headless-diagnostic", action="store_true", help="Run 10-stage diagnostic suite headlessly and exit")
    parser.add_argument("--preview", type=str, nargs="?", const="worlds/canonical_world", help="Launch direct 3D planetary preview")
    parser.add_argument("--world", type=str, default="worlds/canonical_world.npz", help="Path to canonical world dataset")

    args = parser.parse_args()

    if args.diagnostic or args.headless_diagnostic:
        report = run_diagnostic_suite(args.world, hidden_window=True)
        report.print_summary()
        sys.exit(0 if report.is_all_passed() else 1)

    if args.preview:
        preview(args.preview)
        return

    launch(args.world)


if __name__ == "__main__":
    main()
