"""Package the current manuscript and its dependencies without generating a PDF."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / 'output' / 'source' / 'PangenomeFM_NMI_revised_LaTeX.zip'

README = '''# PangenomeFM: revised LaTeX source

Open `main.tex`. This is the revised manuscript, with four main figures, two main
tables, and a narrative supplement. Its supporting TeX files and figure PDFs are
included. No compiled manuscript PDF is included.

## Compile

Upload this entire ZIP to Overleaf, select `main.tex` and use pdfLaTeX. In a local
TeX installation, run `latexmk -pdf main.tex`. If using commands individually:

    pdflatex main.tex
    bibtex main
    bibtex S
    pdflatex main.tex
    pdflatex main.tex

The two BibTeX runs build the main and supplementary reference lists. Figures are
already supplied as vector PDFs. No data, checkpoints, Python packages or server
access are needed to compile the manuscript.

## Editorial comments

All 118 original comments remain verbatim in `editorial_comment_archive.tex`,
which is included by `main.tex`. Search `Long Note: Done` in the main source and
archive for completed requests. The 25-comment follow-up repeats original
comments; its 13 groups have R01-R13 identifiers. Twelve are Done. R08 records
that improved genotype calls have not been demonstrated.

See `editorial/LATEX_COMMENT_COMPLETION_20260928.md` for the current checklist,
including the six remaining scientific/author-input items in the full original
inventory. `editorial/NMI_STRUCTURE_REVISION_20260928.md` explains the four NMI
examples and the main/supplement organization. Neither favorable results nor
journal acceptance are guaranteed by the editorial revision.

Optional preservation check (Python standard library only):

    python check_editorial_comments.py

The native single-file editor cannot load this multi-file project; the source
package is intended for a complete LaTeX project compiler. SHA-256 identities of
all packaged files are recorded in `package_manifest.json`.
'''


def main() -> None:
    files = {path.name: path for pattern in ('*.tex', '*.bib')
             for path in ROOT.glob(pattern)}
    files['check_editorial_comments.py'] = ROOT / 'check_editorial_comments.py'
    files['editorial_comment_manifest.json'] = ROOT / 'editorial_comment_manifest.json'
    for path in ROOT.glob('*provenance.json'):
        files[path.name] = path
    figures = set()
    for path in ROOT.glob('*.tex'):
        # Ignore commented-out examples; all active figure filenames are literal.
        text = '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0]
                         for line in path.read_text().splitlines())
        figures.update(re.findall(r'\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}', text))
    for name in figures:
        files[name] = ROOT / name
    for name in ('LATEX_COMMENT_COMPLETION_20260928.md',
                 'LATEX_COMMENT_RESPONSE_20260928.md',
                 'NMI_STRUCTURE_REVISION_20260928.md'):
        files['editorial/' + name] = REPO / 'docs' / name
    payload = {name: path.read_bytes() for name, path in files.items()}
    payload['README.md'] = README.encode()
    manifest = {
        'entrypoint': 'main.tex',
        'compiler': 'pdfLaTeX with main and S BibTeX bibliographies',
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
