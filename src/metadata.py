from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def collect_metadata(
    *,
    command: list[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build a reproducibility metadata payload for a run directory."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or sys.argv,
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["branch", "--show-current"]),
            "dirty": bool(_run_git(["status", "--porcelain"])),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "settings": dict(settings or {}),
        "dataset": dict(dataset or {}),
        "notes": notes,
    }


def write_metadata(metadata: Mapping[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path
