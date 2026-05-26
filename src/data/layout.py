from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    """Simple on-disk dataset convention.

    Each cleaned dataset lives in one directory, for example:

    data/hprc/
      full_segments.csv
      full_links.csv
      benchmark/
      ccre/

    Future datasets should use the same convention after manual cleaning:

    data/hgsvc3/full_segments.csv.gz
    data/hgsvc3/full_links.csv.gz
    """

    data_dir: Path
    segments: Path
    links: Path

    @property
    def name(self) -> str:
        return self.data_dir.name

    @property
    def benchmark_dir(self) -> Path:
        return self.data_dir / "benchmark"

    @property
    def ccre_dir(self) -> Path:
        return self.data_dir / "ccre"


def _first_existing(data_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


def find_dataset(data_dir: str | Path) -> DatasetLayout:
    """Find required graph tables in a manually cleaned dataset directory."""
    data_dir = Path(data_dir).expanduser()
    segments = _first_existing(data_dir, ["full_segments.csv", "full_segments.csv.gz"])
    links = _first_existing(data_dir, ["full_links.csv", "full_links.csv.gz"])

    missing: list[str] = []
    if segments is None:
        missing.append("full_segments.csv or full_segments.csv.gz")
    if links is None:
        missing.append("full_links.csv or full_links.csv.gz")
    if missing:
        raise FileNotFoundError(
            f"{data_dir} is not a valid GraphGenome-FM dataset directory. "
            f"Missing: {', '.join(missing)}"
        )
    return DatasetLayout(data_dir=data_dir, segments=segments, links=links)

