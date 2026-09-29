"""Refresh the mean/variance tables in ``21_MEAN.md`` from ``collect_mean.py``.

Run after each multi-seed run finishes. Splices fresh tables between the ``<!-- MEAN:START/END -->``
markers.

Usage::

    nix develop . -c python finetuning/update_report21.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "finetuning" / "21_MEAN.md"


def main() -> None:
    tables = subprocess.run(
        ["python", "finetuning/collect_mean.py"], capture_output=True, text=True, cwd=str(REPO_ROOT)
    ).stdout.strip()
    txt = REPORT.read_text()
    txt = re.sub(
        r"<!-- MEAN:START -->.*<!-- MEAN:END -->",
        "<!-- MEAN:START -->\n" + tables + "\n<!-- MEAN:END -->",
        txt,
        flags=re.DOTALL,
    )
    REPORT.write_text(txt)
    print("21_MEAN.md tables refreshed")


if __name__ == "__main__":
    main()
