"""
Fetch daily HTML snapshots from Borzen and BSP and store them under data/snapshots/YYYY-MM-DD/.
This is a lightweight archival step so we can later build scrapers or audits.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
import requests


BORZEN_URL = "https://borzen.si/sl-si/"
BSP_URL = "https://www.bsp-southpool.com/domov.html"


def save_snapshot(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)


def main() -> None:
    today = _dt.date.today().isoformat()
    dst_dir = Path("data") / "snapshots" / today
    save_snapshot(BORZEN_URL, dst_dir / "borzen.html")
    save_snapshot(BSP_URL, dst_dir / "bsp.html")


if __name__ == "__main__":
    main()


