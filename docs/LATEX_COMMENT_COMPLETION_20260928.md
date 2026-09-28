# LaTeX comment completion checklist

Updated 28 September 2026. Applies to the **existing open `main.tex`** in
`manuscript/revision_20260924`; no replacement manuscript or tab was created.

## Latest consolidated-source revision

- [x] Review the supplied NMI rewrite against saved evidence and merge into the existing `main.tex`.
- [x] Inline the article, supplement, tables, references and preserved comments; one active TeX file.
- [x] Retain the six-paragraph Introduction and an aim-focused abstract without findings.
- [x] Lead Results with known SV classification; retain sequence coverage and trained/untrained controls.
- [x] Keep Methods beginning with representation, encoder, objective, loss and optimization.
- [x] Keep four main figures and two tables; reconstruction controls remain supplementary.
- [x] Correct Figure 2's repair annotation, Figure 4's missing d and the development figure's initialization legend.
- [x] Verify added control/development measurements; exclude incomplete chromosome-level replication.
- [x] Preserve original, earlier and supplied rewrite comments inside the single source.

Main figures are paradigms 1, architecture 2, structural variants 3 and regulatory
elements 4. The diagrams introduce the model before Results; methods and detailed
specification are in Methods. See `manuscript/revision_20260924/REVISION_NOTES.md`
for the current full review. Earlier dated placement notes remain in the comment
archive for traceability. No evidence-open item changes status through editing.

## Status rules

**Done** means the requested writing, figure, explanation or supported analysis is
present. It never means a favorable scientific outcome was obtained by editing.
The original comments remain verbatim, followed by `% Long Note:` responses.

The latest excerpt contains **25 comment lines in 13 request groups**. All 25
already occur in the preserved 118-comment archive. Twelve requests are **Done**.
The SV-genotyping question has been answered, but its stronger scientific goal
remains **Evidence open**. It is not marked Done as an improved-genotyping result.

## Latest supplied requests

Source: `b5bfe652-938c-46d4-bc89-ded5ace1c7fb/Pasted text.txt`.
SHA-256: `25c3913810344f46331064766d096e30fbbb37df35445fb3c2234e8d01732351`.
Line numbers refer to that immutable excerpt; the opaque number on line 1 is not
manuscript text or a task.

| ID | Excerpt lines | Status | Resolution in the saved source |
|---|---:|---|---|
| R01 | 5 | **Done** | Six Introduction paragraphs follow representation, biology, sequence models, graph context, prior work/gap and PangenomeFM study aims; findings are confined to Results. |
| R02 | 9–10 | **Done** | Macias-Velasco et al. is cited; its functional-genomics and pipeline-dependent scope is now explained explicitly. |
| R03 | 12 | **Done** | Opening starts with genome representation and coordinate systems before variation. |
| R04 | 15 | **Done** | Four NMI examples reviewed; clear Results, Discussion and method-first Methods; four main figures/two main tables, with key controls retained. |
| R05 | 19 | **Done** | Introduction cites the paradigm figure and incorporates related work/background. |
| R06 | 22–24 | **Done** | Redesigned Figure 1 uses concise parallel rows ending at learned representations; caption explains reuse. |
| R07 | 40–47 | **Done** | Binary cCRE, PLS/pELS/dELS/CTCF-only and fixed complexity strata retained. Results now leads with SV findings under the latest section-boundary request. |
| R08 | 52 | **Evidence open; question answered** | Completed COSIGT analysis and its negative result are explicit. Improving genotype calls, particularly complex/CNV calls, remains untested. |
| R09 | 58 | **Done** | Heading changed to “Much of the graph contribution does not require pretraining.” Shortcut and trained/random controls remain visible. |
| R10 | 68 | **Done** | Methods opens with oriented segments and the coupled encoder, followed by objective/optimization and frozen reuse; the conceptual architecture figure is introduced before Results. Latest author instructions supersede the old Results-design placement. |
| R11 | 78 | **Done** | Exact focal loss, ordered scorer, class weighting, optimizer, learning-rate schedule, clipping and stopping are specified. |
| R12 | 83–86 | **Done** | Dedicated sequence-coverage Results subsection explains 303,425 covered segments, zero missing features and 26.3% end sampling; detailed QC stays supplementary. |
| R13 | 91 | **Done** | Supplement has narrative sections, figures, tables and separate references; S-prefixes match in text and bibliography. |

Search `% Long Note: Done [R` in `main.tex` for the local resolutions, and
`% Long Note: Evidence open [R08]` for the exception. The comment archive carries
the same R identifiers immediately after the corresponding original requests.

The requested citation was rechecked against the primary
[Nature Communications article](https://www.nature.com/articles/s41467-026-73663-3).
It compares genome representations across functional assays with their associated
pipelines; it supports the motivation for studying representation choice, not a
claim that PangenomeFM improves every assay or that graphs always outperform
linear references. No new experiment is inferred from the citation.

## Genotyping: what is answered and what is not

The completed COSIGT analysis covers 265 genomic loci, 30 runs and 900 evaluations.
Its primary endpoint is donor-aggregated alignment-quality fraction at released,
evaluable sample–locus pairs. The training-median MAE is **0.046169**, compared with
**0.053351** for C+S and **0.052699/0.053824** for C+S+T (strict/one-hop).
All original topology MAE intervals cross zero. The validation-only fallback does
not establish a topology benefit.

This answers whether the present features improved that **quality-prediction**
endpoint: no established benefit. It does not answer whether they can improve
actual genotype calls. That claim requires genotype truth, an all-callable
comparison set and independently evaluated complex/CNV strata. The manuscript
now states that distinction beside the result. No successful genotyping result
has been manufactured or substituted for classification.

Evidence: `docs/COSIGT_GENOTYPING_QUALITY_20260927.md` and the completed manuscript
source tables; the existing independent replay receipt remains at
`results/foundation_evidence_20260927/cosigt_quality_local_replay/audit.json`.

## Full original comment inventory

All **38 original request/technical groups** now also have explicit status
responses in `editorial_comment_archive.tex` under their S/M/A/T identifiers.
The original 118 comment occurrences and all 23 baseline source comments remain.
**32 groups are Done** for their stated scope. Six require an honest exception:

| ID | Status | Remaining condition |
|---|---|---|
| S09 | Optional, not done | A transparent locus vignette requires a selection rule and verified tracks/predictions; no vignette is claimed. |
| S12 | Partial, evidence open | Exact donor provenance of the SV-resolution training graph is unverified; known graph/cache/path-resource counts are reported separately. |
| M01 / R08 | Evidence open | Improved genotyping has not been demonstrated. |
| A01 | Author input needed | Approved consortium authorship wording, membership and permission. |
| A02 | Author input needed | Shi-lab repository destination, release license and checkpoint hosting. |
| A03 | Author input needed | Exact grant numbers, recipients and required acknowledgement wording. |

The exhaustive line-to-group mapping remains in
`docs/LATEX_COMMENT_RESPONSE_20260928.md`. The new checklist is the current status;
the earlier document retains the detailed history and rationale.

## Verification and editor limitation

- Comment checker verifies 118 original comments, 23 baseline source comments,
  all 25 follow-up comment lines and the 13 explicit R statuses.
- Six generated numerical row files and all tracked numerical figure assets
  remain unchanged from the pinned pre-writing base.
- The pdfLaTeX-only compression command is now guarded so XeTeX can read it.
- The requested built-in compile was run after editing. It now passes that engine
  error and stops because it receives only `main.tex`, without the companion
  `editorial_comment_archive.tex`. The tool does not support multi-file projects;
  the manuscript also requires modular text, bibliography files and vector figures.
- Two full-project **draft-mode** pdfLaTeX passes validate source and references
  without generating or replacing a PDF. The prior PDF is a verified snapshot
  from before this follow-up, not a rendering of the latest small prose edits.
- The current editor stays open. No alternate tab, replacement source or separate
  PDF was created in this follow-up.

The source-preservation manifest and `comment_resolution_provenance.json` record
verification and the native compiler limitation. A source request can be resolved
while a compiler lacks project support; these are separate statuses.
