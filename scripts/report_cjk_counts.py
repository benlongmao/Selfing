#!/usr/bin/env python3
"""Print backend/*.py files ranked by count of CJK unified ideographs (U+4E00–U+9FFF)."""
import os
import re
import sys

CJK = re.compile(r"[\u4e00-\u9fff]")
ROOT = os.path.join(os.path.dirname(__file__), "..", "backend")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    counts: list[tuple[int, str]] = []
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            c = len(CJK.findall(text))
            if c:
                counts.append((c, os.path.abspath(path)))
    counts.sort(reverse=True)
    for c, path in counts[:n]:
        print(f"{c:5d}  {path}")


if __name__ == "__main__":
    main()
