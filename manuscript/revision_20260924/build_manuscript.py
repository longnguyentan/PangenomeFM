#!/usr/bin/env python3
"""Build the one-TeX manuscript with two reference lists from references.bib."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def executable(name: str) -> str:
    found = shutil.which(name)
    mac = Path('/Library/TeX/texbin') / name
    if found:
        return found
    if mac.is_file():
        return str(mac)
    raise SystemExit(f'{name} is required (TeX Live/MacTeX or Overleaf).')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'output/pdf')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['BIBINPUTS'] = str(ROOT) + os.pathsep + env.get('BIBINPUTS', '')
    env['TEXINPUTS'] = str(out) + os.pathsep + env.get('TEXINPUTS', '')

    def run(command: list[str], cwd: Path, label: str) -> None:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        (out / f'{label}.stdout.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise SystemExit(f'{label} failed; see {out / (label + ".stdout.log")}\n'
                             + result.stdout[-3000:])

    latex = [executable('pdflatex'), '-interaction=nonstopmode', '-halt-on-error',
             '-jobname=PangenomeFM_manuscript', f'-output-directory={out}', 'main.tex']
    run(latex, ROOT, 'latex_1')
    for unit in ('bu1', 'bu2'):
        run([executable('bibtex'), unit], out, 'bibtex_' + unit)
        log = (out / (unit + '.blg')).read_text()
        if 'Warning--' in log:
            raise SystemExit(f'BibTeX warnings remain in {out / (unit + ".blg")}')
    for number in (2, 3, 4):
        run(latex, ROOT, f'latex_{number}')
    log = (out / 'PangenomeFM_manuscript.log').read_text()
    problems = [line for line in log.splitlines() if re.search(
        r'undefined|multiply defined|Overfull|LaTeX Warning:|Package .* Warning:|'
        r'destination with the same identifier', line)]
    if problems:
        raise SystemExit('Resolve final compilation warnings:\n' + '\n'.join(problems))
    print(out / 'PangenomeFM_manuscript.pdf')


if __name__ == '__main__':
    main()
