"""Check that notebooks are valid JSON and report their size and embedded outputs.

Exits 1 only for unreadable notebooks; oversized files are warnings.
"""

import json
import sys
from pathlib import Path

SIZE_WARN_BYTES = 500 * 1024


def count_png_outputs(nb: dict) -> int:
    return sum(
        1
        for cell in nb.get("cells", [])
        for out in cell.get("outputs", [])
        if "image/png" in out.get("data", {})
    )


def main(root: str = "notebooks") -> int:
    paths = sorted(Path(root).glob("*.ipynb"))
    if not paths:
        print(f"::error::no notebooks found under {root}/")
        return 1

    failed = False
    print(f"{'notebook':<28}{'KB':>8}{'cells':>7}{'png':>5}")
    for path in paths:
        size = path.stat().st_size
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"::error file={path}::invalid notebook JSON: {exc}")
            failed = True
            continue
        print(
            f"{path.name:<28}{size / 1024:>8.0f}"
            f"{len(nb.get('cells', [])):>7}{count_png_outputs(nb):>5}"
        )
        if size > SIZE_WARN_BYTES:
            print(f"::warning file={path}::{size / 1024:.0f} KB exceeds {SIZE_WARN_BYTES // 1024} KB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
