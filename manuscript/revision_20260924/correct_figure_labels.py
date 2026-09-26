"""Correct copied vector-figure target labels without changing any plotted value."""

from pathlib import Path
import hashlib
import json
import pymupdf

root = Path(__file__).resolve().parent / "figures"
replacements = {
    "figure2_method": [
        ("SV breakpoint", "SV insertion/", False),
        ("classification", "deletion type", False),
        ("graph", "coord", False),
        ("coord", "graph", False),
        ("graph", "fused", False),
        (
            "weighted binary cross-entropy (no biological labels)",
            "class-adaptive focal loss (no biological labels)",
            False,
        ),
    ],
    "figure4_sv": [
        (
            "Structural-variant breakpoint classification",
            "Insertion versus deletion from SV segment pairs",
            True,
        ),
        (
            "Graph organization contributes strongly to structural-variant breakpoint prediction",
            "Frozen graph representations improve insertion/deletion classification",
            True,
        ),
    ],
}
records = []
for name, edits in replacements.items():
    source = root / "source" / f"{name}.pdf"
    output = root / f"{name}.pdf"
    doc = pymupdf.open(source)
    page = doc[0]
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]
    inserts = []
    for original, new, bold in edits:
        matches = [s for s in spans if s["text"] == original]
        if original in {"coord", "graph"}:
            # The formula superscripts and the frozen representation subscript
            # have distinct positions in the preserved source vector figure.
            lower, upper = (438, 440) if new == "fused" else (347, 349)
            matches = [s for s in matches if lower < s["bbox"][1] < upper]
        if len(matches) != 1:
            raise ValueError(f"Expected unique label: {original}")
        span = matches[0]
        rect = pymupdf.Rect(span["bbox"])
        page.add_redact_annot(rect, fill=(1, 1, 1))
        inserts.append((rect, span, new, bold))
    page.apply_redactions(images=0, graphics=0)
    for rect, span, new, bold in inserts:
        padding = 0 if span["text"] in {"coord", "graph"} else 10
        box = pymupdf.Rect(
            rect.x0 - padding, rect.y0 - 1, rect.x1 + padding, rect.y1 + 4
        )
        font = "hebo" if bold else "helv"
        size = span["size"]
        # Preflight using font widths; no clipped replacement is accepted.
        while pymupdf.get_text_length(new, fontname=font, fontsize=size) > box.width:
            size -= 0.1
        width = pymupdf.get_text_length(new, fontname=font, fontsize=size)
        page.insert_text(
            ((box.x0 + box.x1 - width) / 2, span["origin"][1]),
            new,
            fontsize=size,
            fontname=font,
            color=tuple(((span["color"] >> shift) & 255) / 255 for shift in (16, 8, 0)),
        )
    doc.save(output)
    doc.close()
    records.append(
        dict(
            source=str(source.relative_to(root.parent)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output=str(output.relative_to(root.parent)),
            output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            edits=[dict(old=a, new=b) for a, b, _ in edits],
            plotted_values_changed=False,
        )
    )
(root.parent / "evidence/figure_label_audit.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
