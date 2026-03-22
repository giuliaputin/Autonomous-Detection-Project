"""Thin project entrypoint that delegates to application modules."""

from __future__ import annotations

from apps.entrypoints import main as run_app


if __name__ == "__main__":
    raise SystemExit(run_app())
