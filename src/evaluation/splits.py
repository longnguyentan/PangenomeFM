from __future__ import annotations


def normalize_chrom(chrom: str) -> str:
    """Normalize ``chr22`` and ``GRCh38#0#chr22`` to ``chr22``."""
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom
    if "#" in chrom:
        return chrom.split("#")[-1]
    return chrom


def validate_chromosome_split(
    *,
    train_chrs: list[str] | tuple[str, ...] | None = None,
    val_chrs: list[str] | tuple[str, ...] | None = None,
    test_chrs: list[str] | tuple[str, ...] | None = None,
    allow_missing_train: bool = True,
) -> dict[str, list[str]]:
    """Validate that validation and test chromosomes are disjoint.

    Train can be omitted when it is defined as "all chromosomes except val/test".
    """
    train = {normalize_chrom(c) for c in (train_chrs or [])}
    val = {normalize_chrom(c) for c in (val_chrs or [])}
    test = {normalize_chrom(c) for c in (test_chrs or [])}

    errors: list[str] = []
    if val & test:
        errors.append(f"val/test overlap: {sorted(val & test)}")
    if train and train & val:
        errors.append(f"train/val overlap: {sorted(train & val)}")
    if train and train & test:
        errors.append(f"train/test overlap: {sorted(train & test)}")
    if not allow_missing_train and not train:
        errors.append("train_chrs must be provided")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "train_chrs": sorted(train),
        "val_chrs": sorted(val),
        "test_chrs": sorted(test),
    }

