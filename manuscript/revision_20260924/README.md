# PangenomeFM: consolidated LaTeX source

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

## Paragraph-level review

The current writing pass reviewed 120 content blocks, revised 75 and retained 45;
six headings were revised. Open `PARAGRAPH_REVIEW_20260928.html` for a searchable
old/new table, or `PARAGRAPH_REVIEW_20260928.md` for the complete Markdown version.
Explanations are in Vietnamese. `prose_revision_provenance.json` records the
current manuscript hash and checks; consolidation provenance is historical.

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

## Repository provenance

`consolidation_provenance.json` identifies the supplied rewrite and numerical
evidence. Older `*_provenance.json`, modular TeX sources and PDF snapshots
document previous revisions; they do not describe the current active source.

To rebuild the source-only delivery ZIP:

    python package_latex.py

To regenerate corrected figure labels from the supplied extracted rewrite:

    python repair_rewrite_figures.py /path/to/PangenomeFM_NMI

The architecture vector is generated with
`build_method_figures.method("Fig2_architecture")`. Neither procedure fits a model.
