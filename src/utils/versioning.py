"""
src/utils/versioning.py

Shared run-versioning utility.

Usage in any script:
    from utils.versioning import resolve_run_dir

    out_dir = resolve_run_dir(Path(args.out_dir))
    # out_dir is now e.g. results/v2/gat/run_001/
    # next call on the same base → run_002/, run_003/, ...

The base directory (e.g. results/v2/gat/) acts as a container.
Each invocation of a script gets its own run_NNN/ subdirectory,
so results are never silently overwritten.

If --out_dir already ends in run_NNN (e.g. you pass an explicit run
directory for inspection), it is returned unchanged.
"""

from __future__ import annotations
import re
from pathlib import Path


_RUN_RE = re.compile(r"^run_(\d{3})$")


def resolve_run_dir(base: Path, *, exist_ok: bool = False) -> Path:
    """
    Given a base output directory, find or create the next run_NNN subdir.

    Parameters
    ----------
    base      : base directory (created if it doesn't exist)
    exist_ok  : if True, return the latest existing run dir instead of
                creating a new one (useful for read-only inspection)

    Returns
    -------
    Path to the resolved run directory (already created on disk)
    """
    base = Path(base)

    # If the caller already passed an explicit run dir, use it as-is
    if _RUN_RE.match(base.name):
        base.mkdir(parents=True, exist_ok=True)
        return base

    base.mkdir(parents=True, exist_ok=True)

    existing = sorted(d for d in base.iterdir() if d.is_dir() and _RUN_RE.match(d.name))

    if exist_ok and existing:
        return existing[-1]

    next_n = len(existing) + 1
    run_dir = base / f"run_{next_n:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [versioning] Run directory: {run_dir}")
    return run_dir


def latest_run_dir(base: Path) -> Path | None:
    """Return the most recent run_NNN subdir under base, or None."""
    base = Path(base)
    if not base.exists():
        return None
    candidates = sorted(
        d for d in base.iterdir() if d.is_dir() and _RUN_RE.match(d.name)
    )
    return candidates[-1] if candidates else None


def list_run_dirs(base: Path) -> list[Path]:
    """Return all run_NNN subdirs under base, sorted."""
    base = Path(base)
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and _RUN_RE.match(d.name))
