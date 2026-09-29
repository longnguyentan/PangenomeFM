"""Package the current manuscript and its dependencies without generating a PDF."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / 'output' / 'source' / 'PangenomeFM_NMI_revised_LaTeX.zip'

README = """# PangenomeFM: consolidated LaTeX source

Open `main.tex`. It contains the entire main article, Supplementary Information,
all tables, both reference lists, and the preserved professor/editorial comments.
There is exactly ONE .tex file. Only the ten vector figures remain external.
No compiled manuscript PDF, extra .tex file or .bib file is required or included.

## Compile

Upload the entire ZIP to Overleaf and select `main.tex` with pdfLaTeX. Locally:

    pdflatex main.tex
    pdflatex main.tex
    pdflatex main.tex

There is no BibTeX step: the references are inline. No data, Python package,
checkpoint or server access is required. The Codex single-document compiler does
not currently load companion figure PDFs; use a full LaTeX project compiler.

## Organization and editorial decisions

Main sections: Abstract, Introduction (six paragraphs), Results, Discussion,
Methods, Data availability, Code availability, References and author statements.
The Supplementary Information follows, with 11 notes, Supplementary Methods,
consortium-list placeholders and its own S-numbered reference list.

Four main figures: paradigms (1), architecture (2), structural variants (3),
regulatory elements (4). Two main tables: three-class SVs and trained/untrained
controls. Reconstruction controls are supplementary. Figure filenames need not
match their final automatically assigned supplementary numbers.

The author's instruction to keep findings out of the abstract and Introduction
is retained. Completed results and null findings remain in Results. New
single-fold development findings are labelled as such. No incomplete chromosome
replication result is used. Author-confirmation items remain visible in red.
See `PARAGRAPH_REVIEW_20260928.html` (searchable side-by-side) or its Markdown
version for the complete paragraph-level writing review with Vietnamese reasons.
See `REVISION_NOTES.md` for the evidence checks, reference corrections and open
scientific/author-input items.

## Comments and integrity

All 118 original professor comments and the 25-line follow-up are preserved
verbatim inside `main.tex`, together with earlier and supplied rewrite comments.
The historical comment archive follows the document end marker; it does not
print. Search `% Long Note:` for editorial responses and current dispositions.
Improved genotype calls remain evidence-open, not marked as a favorable result.

Optional preservation check (Python standard library only):

    python check_editorial_comments.py

`package_manifest.json` records the SHA-256 of each delivered file. The older
modular sources in the repository are historical and are not dependencies.
"""


def main() -> None:
    files = {"main.tex": ROOT / "main.tex",
             "check_editorial_comments.py": ROOT / "check_editorial_comments.py",
             "editorial_comment_manifest.json": ROOT / "editorial_comment_manifest.json",
             "consolidation_provenance.json": ROOT / "consolidation_provenance.json",
             "REVISION_NOTES.md": ROOT / "REVISION_NOTES.md",
             "PARAGRAPH_REVIEW_20260928.md": ROOT / "PARAGRAPH_REVIEW_20260928.md",
             "PARAGRAPH_REVIEW_20260928.html": ROOT / "PARAGRAPH_REVIEW_20260928.html",
             "prose_revision_provenance.json": ROOT / "prose_revision_provenance.json"}
    from check_editorial_comments import without_comments
    text = without_comments((ROOT / "main.tex").read_text()).split(r"\end{document}", 1)[0]
    figures = set(re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", text))
    for name in figures:
        files[name] = ROOT / name
    payload = {name: path.read_bytes() for name, path in files.items()}
    payload['README.md'] = README.encode()
    manifest = {
        'entrypoint': 'main.tex',
        'compiler': 'pdfLaTeX; inline main and supplementary bibliographies; no BibTeX step',
        'scope': 'Source package only; no new experiment or compiled manuscript PDF.',
        'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(payload.items())},
    }
    payload['package_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        for name, data in payload.items():
            assert archive.read(name) == data, name
    print(f'Packaged {len(payload)} files: {OUTPUT}')
    print(f'SHA-256: {hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
