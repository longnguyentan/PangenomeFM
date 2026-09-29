"""Package the current manuscript and its dependencies without generating a PDF."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / 'output' / 'source' / 'PangenomeFM_NMI_revised_LaTeX.zip'

README = (ROOT / 'README.md').read_text()


def main() -> None:
    files = {"main.tex": ROOT / "main.tex",
             "references.bib": ROOT / "references.bib",
             "build_manuscript.py": ROOT / "build_manuscript.py",
             "check_bibliography.py": ROOT / "check_bibliography.py",
             "bibliography_audit_20260929.json": ROOT / "bibliography_audit_20260929.json",
             "BIBLIOGRAPHY_AUDIT_20260929.md": ROOT / "BIBLIOGRAPHY_AUDIT_20260929.md",
             "MERGE_REVIEW_20260929.md": ROOT / "MERGE_REVIEW_20260929.md",
             "merge_revision_provenance_20260929.json": ROOT / "merge_revision_provenance_20260929.json",
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
        'compiler': 'pdfLaTeX and BibTeX bu1/bu2; external references.bib',
        'scope': 'Source package only; no new experiment or compiled manuscript PDF.',
        'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(payload.items())},
    }
    payload['package_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 29, 0, 0, 0))
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
