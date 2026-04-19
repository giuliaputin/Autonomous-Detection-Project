"""Project CLI entrypoint.

This file intentionally remains thin and delegates all argument handling and
mode dispatch to ``apps.entrypoints``.
"""

from __future__ import annotations

from apps.entrypoints import main as run_app

if __name__ == "__main__":
    raise SystemExit(run_app())
