"""Command entry point for GraphGenome-FM.

The repo uses a flat ``src/`` tree for collaborator-friendly browsing:
``analysis/``, ``data/``, ``evaluation/``, ``graph/``, ``models/``,
``tasks/``, ``training/``, and ``utils/`` live directly under ``src/``.
This small module keeps ``python -m graphgenomefm`` as the stable command.
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    from cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
