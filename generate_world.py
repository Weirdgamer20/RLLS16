#!/usr/bin/env python3
"""
RLLS 16 — Authoritative Canonical 2D Earth Generator.
Generates deterministic 2D Earth simulation layers from real-world vector datasets.
"""

import sys
import argparse
from pathlib import Path

# Ensure package root is in sys.path
pkg_root = Path(__file__).resolve().parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from world_acquisition.generate_canonical_world import generate


def main():
    parser = argparse.ArgumentParser(
        description="RLLS 16 — Authoritative Canonical 2D Earth Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--width", type=int, default=512, help="Canonical grid longitude samples")
    parser.add_argument("--height", type=int, default=256, help="Canonical grid latitude samples")
    parser.add_argument("--seed", type=int, default=16001, help="Deterministic world seed")
    parser.add_argument("--raw-dir", type=str, default="world_data/raw", help="Path to raw vector datasets")
    parser.add_argument("--output-dir", type=str, default="world_data/canonical", help="Output directory for canonical package")
    parser.add_argument("--worlds-dir", type=str, default="worlds", help="Output directory for runtime worlds copy")

    args = parser.parse_args()

    raw_path = pkg_root / args.raw_dir
    canonical_path = pkg_root / args.output_dir
    worlds_path = pkg_root / args.worlds_dir

    generate(
        width=args.width,
        height=args.height,
        seed=args.seed,
        raw_dir=raw_path,
        canonical_dir=canonical_path,
        worlds_dir=worlds_path,
    )


if __name__ == "__main__":
    main()
