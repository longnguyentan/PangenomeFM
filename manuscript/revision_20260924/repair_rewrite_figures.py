"""Repair labels in the supplied vector figures without changing plotted data.

Usage: python repair_rewrite_figures.py /path/to/extracted/PangenomeFM_NMI
Requires pypdf. The architecture diagram is rebuilt separately from its source
with build_method_figures.method("Fig2_architecture").
"""
from copy import deepcopy
import argparse
import hashlib
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ByteStringObject, ContentStream, FloatObject, NumberObject

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "Fig4_regulatory_elements.pdf": "37f01514bf6c4fd35438384632f8efa8b99280d2fa33143f9617588747f992f4",
    "SuppFig5_objective_development.pdf": "ddd288bdbe51c7bfac07b34d15de56b5a660e2b2e184b6dfaff8ebc3a4ef45d2",
}


def repair(source: Path) -> None:
    for name, expected in EXPECTED.items():
        path = source / "figures" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected source figure: {path}")
        reader = PdfReader(path)
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        page = writer.pages[0]
        content = ContentStream(page.get_contents(), writer)
        changed = 0
        if name.startswith("Fig4"):
            for i, (operands, operator) in enumerate(content.operations):
                if operator == b"TJ" and operands == [["b"]]:
                    # Copy the existing b label's font and x position; use c's
                    # baseline for the lower row. Every plot operation is kept.
                    label = deepcopy(content.operations[i - 5:i + 3])
                    assert label[0][1] == b"q" and label[1][1] == b"cm"
                    assert label[3][0] == ["/F2", 16]
                    label[1][0][5] = FloatObject(264.5512967557)
                    label[5][0][0] = ArrayObject([ByteStringObject("d".encode("utf-16-be"))])
                    label.insert(1, ([NumberObject(0)], b"g"))
                    content.operations.extend(label)
                    changed += 1
                    break
        else:
            for operands, operator in content.operations:
                if operator == b"TJ" and operands == [["Untrained (same initialization)"]]:
                    operands[0] = ArrayObject([ByteStringObject("Untrained (same architecture)".encode("utf-16-be"))])
                    changed += 1
        if changed != 1:
            raise ValueError(f"Expected exactly one label repair for {name}; got {changed}")
        page.replace_contents(content)
        writer.add_metadata({"/Creator": "PangenomeFM vector label correction; plot data unchanged"})
        output = ROOT / "figures" / name
        writer.write(output)
        print(f"Repaired {output.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    repair(parser.parse_args().source)
