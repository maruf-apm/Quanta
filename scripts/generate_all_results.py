#!/usr/bin/env python3
"""
Batch script to run all 6 test cases and generate the full result suite.
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_CONFIGS = [
    "configs/tests/test1.yaml",
    "configs/tests/test2.yaml",
    "configs/tests/test3.yaml",
    "configs/tests/test4.yaml",
    "configs/tests/test5.yaml",
    "configs/tests/test6.yaml",
]


def run_comparison(config_path):
    cmd = [
        sys.executable,
        "scripts/compare_models.py",
        "--config", config_path,
        "--output_dir", "./results",
    ]
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    for cfg in TEST_CONFIGS:
        if os.path.exists(cfg):
            run_comparison(cfg)
        else:
            print(f"Config not found: {cfg}, skipping.")
    print("\nAll experiments completed. Check ./results/ for outputs.")


if __name__ == "__main__":
    main()
