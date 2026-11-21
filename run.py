"""Run continual learning experiments."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.train import run_experiment

METHODS = ["baseline", "ewc", "si", "replay", "nsm"]


def main():
    results = {}
    for m in METHODS:
        print(f"\n=== {m} ===")
        results[m] = run_experiment(m, epochs=2)
    Path("viz").mkdir(exist_ok=True)
    (Path("viz") / "results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved viz/results.json")


if __name__ == "__main__":
    main()
