# NMI exemplar review and manuscript revision

Verified: 28 September 2026. Scope: writing, organization and comment preservation;
no new experiments, metric recomputation or incomplete test results imported.
Base: `3e1032c34c7ced0477b4fd40c28bf67f8166190f` on
`codex/v2-evidence-review-20260927`.

## What the four examples teach

Page anchors below refer to the supplied PDFs, not printed journal page numbers.
These are examples of published article organization, not mandatory NMI rules or
evidence that a particular framing will secure acceptance. The first two supplied
PDFs retain placeholder publication dates. Separate supplementary files were not
supplied; observations about their contents use explicit main-paper references.
No prose, diagrams or numerical findings from the examples are imported into our
manuscript.

### 1. Task-structured modularity emerges in artificial networks and aligns with brain architecture

[Wu et al., DOI 10.1038/s42256-026-01306-9](https://doi.org/10.1038/s42256-026-01306-9).
The abstract and introduction identify a missing explanation, formulate three
questions, and connect them to controlled training paradigms (PDF pp. 1–2).
Results proceed from representation to performance, intervention and biological
alignment (pp. 2–8). The central random/prior-mask controls remain in main Fig. 5,
including pairing and density-matching definitions (p. 6). More extensive
robustness analysis is supplementary. Discussion limits the biological analogy
(p. 9). **Application:** organize PangenomeFM around questions and keep the random
encoder comparison in the main argument. Computational experiments do not justify
claims about causal biological mechanisms.

### 2. Guiding large language models to predict edit sequences for molecular synthesizability optimization

[Li and Lai, DOI 10.1038/s42256-026-01304-x](https://doi.org/10.1038/s42256-026-01304-x).
The introduction moves from a practical bottleneck to the limitations of current
representations, then explains the editing formulation (pp. 1–2). Results begin
with the primary benchmark and its fairness conditions, followed by explanatory
examples and applications (pp. 2–8). Detailed ablations move to the supplement,
but their conclusions remain beside the benchmark (p. 2). Limitations accompany
application claims (p. 6). **Application:** explain why oriented segments and two
neighborhoods are needed before presenting outcomes; lead with the strongest
completed SV comparison. Attribute standard components rather than equating their
combination with invention of each component.

### 3. Causal evidence that language models use confidence to drive behaviour

[Kumaran et al., DOI 10.1038/s42256-026-01293-x](https://doi.org/10.1038/s42256-026-01293-x).
The abstract previews a four-phase chain of evidence. The introduction defines a
precise uncertainty and a design that separates possible explanations (pp. 1–2).
Results move from measurement through alternative predictors to interventions
and internal representations (pp. 2–8). Important alternatives remain in the main
text; additional model and regression details are supplementary. The conceptual
figure and Discussion constrain mechanistic interpretation (pp. 8–10).
**Application:** distinguish graph information beyond sequence from learned
weights beyond graph statistics and random encoders. PangenomeFM has no equivalent
causal intervention, so this paper's causal language is not transferred.

### 4. NucleicBERT interprets RNA sequence space through self-supervised language modelling

[Upadhyay et al., DOI 10.1038/s42256-026-01295-9](https://doi.org/10.1038/s42256-026-01295-9).
The paper frames a biological data bottleneck and a defining input constraint,
while identifying BERT as its architectural basis (pp. 1–2). The Results opening
separates pretrained/random and frozen/fine-tuned regimes. Main Table 1 retains
these controls alongside benchmarks (p. 4). Failed frozen fitness probes remain
beside successful fine-tuning (p. 6); nuisance-control details move to Extended
Data while their interpretation stays in Results (pp. 4–5, 16). Discussion
connects representation and prediction evidence (pp. 9–10). **Application:** give
H/random controls a main table, preserve null endpoints, and describe frozen
reuse accurately. Fine-tuning outcomes in this example are not comparable to
PangenomeFM's frozen protocol.

## Revised main argument

1. **Introduction:** six paragraphs, retaining the requested sequence: population
   representation problem; biological motivation; sequence-model context; two
   graph/coordinate neighborhoods; related paradigms and the attribution gap;
   our design, tests and bounded contribution.
2. **Results: integrating genomic position and native connectivity.** Three
   design rationales connect representation, layerwise coupling and frozen reuse
   to Fig. 2 and the opening Methods. The requested sequence-coverage heading stays.
3. **Results: known SV types.** The largest original conditional gain leads the
   biological evidence, followed by the controlled three-class extension.
4. **Results: regulatory reuse.** cCRE provides the complementary sequence-led
   setting; EN-TEx tests activity and includes the null as well as positive outcomes.
5. **Results: attribution.** Main Table 2 exposes all trained/random comparisons,
   including the unfavorable strict cCRE result. The degree shortcut remains explicit.
6. **Results: external limits.** HG008, TraitGym and COSIGT have concise main-text
   outcomes; exact contrasts and sensitivity details remain available.
7. **Discussion:** contribution; task-dependent information; attribution;
   representation limits; next discriminating methodological test.
8. **Methods:** the representation, coupled encoder, masking/scorer/loss and
   frozen reuse still precede datasets and task-specific implementation.

## Main/supplement movement

| Item | Final placement | Reason and protection |
|---|---|---|
| Paradigms and architecture diagrams | Main Figs. 1–2 | Explain the representation choice and mechanism |
| Original SV and cCRE figures | Main Figs. 3–4, reordered | Strongest structural comparison precedes regulatory contrast; assets unchanged |
| Natural-frequency SV table | Main Table 1 | Controls the expanded task for length and graph statistics; uncertain INV and absent random arm remain explicit |
| Full four-row trained-minus-random table | Main Table 2, moved from supplement | Determines the central attribution claim; all intervals and p/q values retained |
| Historical reconstruction/transfer plot | Supplementary Fig. S1, moved from main Fig. 5 | Shortcut-sensitive objective cannot carry the biological headline; exact scores/caveat retained |
| EN-TEx complete inventory and sensitivities | Supplement plus main outcome paragraph | All endpoints retained, including nulls; avoids organizing by job counts |
| SV run receipts and exact class counts | Supplement plus exact main table | Execution detail supports reproducibility without leading the argument |
| Masked-feature E development | Supplement; separate pending sentence in Discussion | Development success is not completed independent validation or a replacement for T |
| External nulls | Main and supplement | Movement must not conceal the limits of transfer |

The main text has **four figures and two tables**. Labels are stable even when
numbers change: `tab:s-hr` now resolves to main Table 2, and
`fig:generalization-transfer` resolves to Supplementary Fig. S1. Historical asset
filenames retain their original numbers to avoid unnecessary renaming.

## Comments and provenance

- Existing comments are retained independently in each edited source file.
- `editorial_comment_archive.tex` restores all **118 original attached-LaTeX
  comment occurrences**, with original line anchors and verbatim comment suffixes.
  It is included from `main.tex` but contains comments only, so it produces no PDF
  text. Original requests remain source material, not unverified scientific facts.
- New decisions are annotated inline as `% Long Note:`. Major changes link to
  this page-anchored review, including why controls stay in the main text.
- `editorial_comment_manifest.json` and `check_editorial_comments.py` verify the
  archived comments and preserve the pre-revision per-file comment multisets.
- The earlier full comment disposition ledger remains in
  `LATEX_COMMENT_RESPONSE_20260928.md`.
- `nmi_revision_provenance.json` records input identities, build and verification.

## Verification checklist

- [x] All four supplied examples read; representative main pages/figures rendered.
- [x] Main narrative revised and Methods-first organization preserved.
- [x] All original comments restored; prior inline comments retained; Long Note annotations added.
- [x] Main attribution controls and negative outcomes retained.
- [x] Completed-study tables regenerated from verified snapshots; numerical rows unchanged.
- [x] Both bibliographies rebuilt; source compiles without unresolved references or box warnings.
- [x] Complete revised PDF rendered and visually checked.
- [x] Changes committed and pushed to the existing evidence-review branch.

Build details, counts and hashes are in the manuscript provenance file. The checked
build/review/push items describe the delivered revision, not newly run experiments.

## Claim boundary

This is a clearer presentation of existing evidence, **not a demonstration of new
architecture performance**. It does not claim universal learned-weight superiority,
causal regulatory effects, donor-held-out validation, successful SV genotyping, or
journal readiness/acceptance. More forceful prose cannot repair those evidence gaps.

## Subsequent comment-resolution pass

The follow-up in `LATEX_COMMENT_COMPLETION_20260928.md` marks 12 of the 13 supplied
requests Done and retains the genotyping evidence gap. It also annotates all 38
original groups. The saved source has small subsequent prose/engine-compatibility
edits; the 32-page PDF and visual checks above describe the preceding NMI writing
snapshot. The latest source passed full-project non-PDF checks; the native editor
cannot load its companion project files. No new PDF was generated in that follow-up.
