#!/usr/bin/env python3
"""
Run all model interface tests in sequence.

Usage:
    python interface/test_all.py
"""

import subprocess
import sys
from pathlib import Path

TESTS = [
    "interface/test_3layer.py",
    "interface/test_2layer.py",
    "interface/test_simple.py",
    "interface/test_simple3layer.py",
]


def main():
    root = Path(__file__).resolve().parent.parent
    # Use the venv Python if available, otherwise fall back to sys.executable
    venv_python = root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'#' * 60}")
    print(f"  Running all model interface tests")
    print(f"{'#' * 60}\n")

    for test in TESTS:
        test_path = root / test
        print(f"\n>>> Running {test} ...")
        result = subprocess.run(
            [python, str(test_path)],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            passed += 1
            # Print last 3 lines (the PASSED summary)
            lines = result.stdout.strip().split("\n")
            for line in lines[-3:]:
                print(f"    {line}")
        else:
            failed += 1
            errors.append((test, result.stderr or result.stdout))
            print(f"    FAILED!")
            print(result.stderr[-500:] if result.stderr else result.stdout[-500:])

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(TESTS)}")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
