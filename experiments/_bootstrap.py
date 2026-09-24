"""Configure local import paths for paper experiment entry points."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", Path(__file__).resolve().parent / "_internal"):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)
