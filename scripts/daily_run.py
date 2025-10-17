from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    # 1) Optional: train (comment out if too heavy daily)
    # run([sys.executable, "-m", "src.train"])  # retrain

    # 2) Predict tomorrow
    run([sys.executable, "-m", "src.predict"])  # writes out/forecast_*.json

    # 3) Fetch snapshots (Borzen, BSP)
    run([sys.executable, "scripts/fetch_snapshots.py"])  # data/snapshots/YYYY-MM-DD/*.html

    # 4) Sync forecasts to docs and update manifest
    run([sys.executable, "scripts/sync_forecasts_to_docs.py"])  # docs/data + manifest


if __name__ == "__main__":
    main()


