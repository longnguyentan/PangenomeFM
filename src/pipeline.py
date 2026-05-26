from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


def run_module(module: str, args: dict[str, Any]) -> None:
    """Run an existing module as if called with ``python -m``.

    This keeps the new CLI small while the old research scripts are migrated
    into importable training/evaluation modules.
    """
    argv = [module]
    for key, value in args.items():
        flag = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend([flag, str(value)])

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def default_results_dir(dataset_name: str, task: str) -> Path:
    return Path("results") / dataset_name / task

