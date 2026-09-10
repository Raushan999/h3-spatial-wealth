"""Maharashtra wrapper — same pipeline as `python run_state.py --state MH`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_state import main as run_state_main


def main() -> None:
    extra = sys.argv[1:]
    run_state_main(["--state", "MH", *extra])


if __name__ == "__main__":
    main()
