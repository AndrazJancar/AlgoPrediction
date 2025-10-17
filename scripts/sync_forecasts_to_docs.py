"""
Synchronize forecast JSON files from `out/` to `docs/data/` and update manifest.

Rules:
- Copy all `forecast_YYYY-MM-DD.json` files from `out/` into `docs/data/`.
- Generate `docs/data/manifest.json` with files sorted by date descending.
- If no forecasts exist, create an empty manifest.

This script uses only Python's standard library.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


RE_FILENAME = re.compile(r"^forecast_(\d{4}-\d{2}-\d{2})\.json$")


@dataclass(frozen=True)
class ForecastFile:
    path: Path
    date: datetime


def find_forecasts(out_dir: Path) -> List[ForecastFile]:
    forecasts: List[ForecastFile] = []
    for p in out_dir.glob("forecast_*.json"):
        m = RE_FILENAME.match(p.name)
        if not m:
            continue
        date_str = m.group(1)
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        forecasts.append(ForecastFile(path=p, date=date))
    forecasts.sort(key=lambda f: f.date, reverse=True)
    return forecasts


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def copy_files(forecasts: List[ForecastFile], dst_dir: Path) -> None:
    for f in forecasts:
        target = dst_dir / f.path.name
        if not target.exists() or target.read_bytes() != f.path.read_bytes():
            target.write_bytes(f.path.read_bytes())


def write_manifest(forecasts: List[ForecastFile], dst_dir: Path) -> None:
    manifest_path = dst_dir / "manifest.json"
    files = [
        {
            "path": f"data/{f.path.name}",
            "label": f"Napoved {f.date.date()}",
        }
        for f in forecasts
    ]
    manifest = {"files": files}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "out"
    docs_data_dir = repo_root / "docs" / "data"

    ensure_dir(docs_data_dir)

    forecasts = find_forecasts(out_dir) if out_dir.exists() else []
    if forecasts:
        copy_files(forecasts, docs_data_dir)
    write_manifest(forecasts, docs_data_dir)


if __name__ == "__main__":
    main()


