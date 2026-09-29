# PangenomeFM manuscript: revised 29 September 2026

`main.tex` is the one editable TeX source for the main article, Supplementary
Information, tables and preserved professor/editorial comments. All bibliographic
entries are in **`references.bib`**, never embedded in `main.tex`. Ten figure PDFs
are external. The source ZIP contains the entire project needed for compilation.

## Compile

Upload the complete source ZIP to Overleaf and select `main.tex` with pdfLaTeX.
With an existing TeX Live/MacTeX installation, the reproducible local build is:

```bash
python3 check_bibliography.py
python3 check_editorial_comments.py
python3 build_manuscript.py
```

The last command creates `output/pdf/PangenomeFM_manuscript.pdf`, runs BibTeX for
both reference lists and rejects unresolved citations/references and layout
warnings. It uses Python's standard library; no model/data/server is needed.
The equivalent manual sequence, from this folder, is:

```bash
pdflatex main.tex
bibtex bu1
bibtex bu2
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

The Codex single-document compiler currently cannot load this project's companion
figure PDFs or bibliography. The saved source remains editable in its existing
tab; the full-project PDF is the verified rendered artifact.

## Current editorial decisions

The latest author feedback prioritizes manuscript quality over word count and
restores findings in the Abstract, final Introduction paragraph and Discussion.
Results use the authors' voice, with the main SV task named insertion-versus-
deletion classification. The more precise masking, initialization, post hoc and
replication caveats are retained. Completed EN-TEx weighting and multiplicity
sensitivities accompany the primary estimates.

Main sections are Abstract, Introduction (six paragraphs), Results, Discussion,
Methods, availability statements, References and author statements. The Methods
open with graph representation, encoder, objective, loss and optimization.
Supplementary Information contains 11 notes, Supplementary Methods and its own
S-numbered references. The main article has four figures and two tables.

See `MERGE_REVIEW_20260929.md` for the new old/new comparisons and editorial
choices, and `BIBLIOGRAPHY_AUDIT_20260929.md` for all 45 distinct works and the
Google Scholar access limitation. All 52 active keys resolve; seven intentional
`supp__` aliases give repeated works distinct anchors in the two reference lists.

`PARAGRAPH_REVIEW_20260928.html` and its Markdown version are the previous
paragraph-level review, retained as history. `REVISION_NOTES.md` clearly separates
the current changes from the older editorial passes.

## Comments and unresolved author inputs

All 118 original professor comments, 25 follow-up comment lines and later source
comments are preserved. Search `% Long Note:` for responses. The comment archive
after the document end marker does not print. A request to demonstrate improved
SV genotyping remains evidence-open; it has not been converted into a claim.
Grant identifiers, consortium approval/member lists, contributions and publication
repository/DOI information still require author confirmation and remain visible.

## Integrity and source package

`merge_revision_provenance_20260929.json` identifies the current revision;
`bibliography_audit_20260929.json` pins the reviewed bibliography. Older provenance
files describe earlier revisions, not the current source hash.

```bash
python3 package_latex.py
```

This creates `output/source/PangenomeFM_NMI_revised_LaTeX.zip` with exactly one
TeX file, one bibliography, figures, checks and revision notes. Its internal
`package_manifest.json` records every delivered file hash. Compiled PDFs and
historical modular TeX files are not dependencies of the source ZIP.

The separate, comprehensive team report is
`../../docs/ENTEX_COMPLETE_TEAM_REPORT_20260929.md` from this directory. It reviews
450 completed EN-TEx configurations, 3,150 feature-specific metric records,
source QC, sensitivity analyses and evidence limits; it is not a new model run.
