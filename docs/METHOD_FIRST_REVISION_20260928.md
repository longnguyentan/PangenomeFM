# Method-focused manuscript revision — 28 September 2026

## Outcome

The manuscript now leads with the representation problem, the PangenomeFM design
and the evidence that tests it. It preserves the completed biological results,
negative findings and limits on learned-weight attribution. This is a writing,
method-description and figure revision; it adds no experimental measurements.

The editable source is `manuscript/revision_20260924/main.tex`. The supplied LaTeX
is an older version with the same SHA-256 as the previously supplied source; it
was used as an editorial reference, not copied over the completed-evidence draft.
Input identities are recorded in `method_revision_provenance.json` alongside the
manuscript. Private meeting transcripts are not copied into the repository.

## Checklist

- [x] Read all three meeting transcripts, including unrelated project discussions.
- [x] Read and inventory all 118 actual `%` comment lines, including repeated and
  technical comments; preserve exact attachment anchors in the response ledger.
- [x] Rewrite the abstract around the problem, method, representative validation
  and supported conclusion.
- [x] Develop the six-paragraph Introduction: representation → biological relevance
  → sequence models → graph information → existing approaches and gap → this method.
- [x] Organize Results into five questions that connect method design to evidence.
- [x] Open Methods with representation/architecture, masking/loss/optimization,
  and frozen feature comparisons, before resource and task details.
- [x] Correct shared branch inputs, layerwise residual gating, node-rank context,
  ordered scorer, focal weighting and archived optimizer settings against code.
- [x] Distinguish historical directed-edge masking and execution labels from later
  reverse-complement masking and deterministic-seeding repairs.
- [x] Rebuild both conceptual figures as editable vector artwork and align their
  panels with the opening Methods subsections.
- [x] Keep the sequence coverage/end-sampling result under a named Results heading.
- [x] Give Supplementary Information an introduction, topical narratives and its
  own separately linked reference list.
- [x] Retain primary nulls, negative controls, all SV-type populations and incomplete
  development status; move detail without promoting preliminary evidence.
- [x] Keep five main figures and one main table; retain the original primary summary
  table and SV-complexity figure in Supplementary Information.
- [x] Final compile and rendered-page inspection: all 31 pages checked, including
  the corrected method diagram, equations, tables and supplementary references.

## Revised argument

### Introduction

There are six paragraphs and no printed Introduction heading. They introduce
linear and graph representations before variation, explain why representation
matters in biology, and establish the methodological question before results.
Macias-Velasco et al. remains cited for functional-genomics motivation. DeepGene
and PangenomeX are contextual comparisons, not same-task performance baselines;
PangenomeX's phylogeny-guided CNV network is distinguished from native segment
adjacency.

### Results

1. **PangenomeFM couples genomic position and connectivity for frozen reuse.**
   Method overview plus a named coverage subsection that distinguishes complete
   segment joins from visibility of every nucleotide.
2. **Regulatory transfer is complementary to sequence and task dependent.**
   Original cCRE, subclasses and EN-TEx, including primary nulls and ascertainment.
3. **Graph representations inform known SV types beyond sequence.**
   Known INS/DEL classification, strata and the explicit INS/DEL/INV extension.
4. **Structural controls distinguish graph information from learned-weight gains.**
   Degree shortcut, trained/random/H controls, scaling and a scoped E development
   summary. Full E development remains in the supplement.
5. **Cross-resource and external evaluations define the limits of reuse.**
   Historical reconstruction transfer, prospective HG008, TraitGym and COSIGT.

### First three Methods subsections

1. Oriented-segment representation and coupled coordinate–graph encoder.
2. Query masking, connection scoring and optimization.
3. Frozen segment pooling and conditional information comparisons.

The contribution is the pangenome representation and controlled reuse framework.
Graph attention, rotary encoding, gating, focal loss, AdamW and GraphMAE-style
reconstruction are attributed/adopted components. The text does not present these
as newly invented operators, or turn the pending E comparison into a validated
replacement for original T.

## Meeting-to-change map

Paragraph numbers below are 1-based nonempty blocks after splitting Unicode line
and paragraph separators, not timestamps. The transcripts have no speaker labels;
roles are inferred from context. Input counts: 89, 265 and 197 paragraphs.

| Relevant source | Request | Implemented response |
|---|---|---|
| 11 Sep P23–26; 15 Sep P77–85 | Finish the available draft; assess method/architecture/training contribution; unfinished tasks may have removable sections | Completed results drive the main text; development and missing evidence are explicitly scoped, with no invented findings |
| 11 Sep P27–34; 15 Sep P71–78 | Broaden functional evaluation, especially ENCODE and EN-TEx | Completed cCRE subtype and EN-TEx results remain together, including null endpoints |
| 11 Sep P35 | Acknowledge T2T feedback and grants | General T2T feedback acknowledgement added; NHGRI U24/NIGMS R01 award details remain author-confirmed |
| 15 Sep P83–90, P165–166 | Clarify what graph information and model design contribute; use a methods-journal tone | Representation question first; conditional features and learned-weight controls explicitly separated |
| **22 Sep P1–3** | “more method first”; explain graph representation, architecture, real novel components; objective/optimization “if we do anything” | Opening Methods reordered and audited; standard components cited, exact implementation stated |
| **22 Sep P4–7** | Connect figure panels to methodological sections; expand the Introduction | Rebuilt Figures 1/2, caption-to-Methods mapping, expanded six-paragraph Introduction |
| 22 Sep P11 | Examine multiple SV types and stratify | Binary and three-class evidence integrated with uncertainty for rare INV retained |

The following were not silently converted into current manuscript claims:
compression (11 Sep P61–63), Parkinson applications (15 Sep P150–152), and future
cross-source harmonization (15 Sep P219–227). The Hi-C sample-scaling discussion,
HGSVC3 functional analyses and agentic diagnosis figures belong to other projects.
The “model size, on the model side” passage is a spoken correction, not an
unqualified demand to increase parameter count.

## Scientific corrections discovered during the writing pass

- Both streams consume the same evolving hidden state; the coordinate branch is
  not isolated from the seven shared numeric inputs.
- The gate is applied at every layer, followed by residual addition and LayerNorm.
- Adaptive coordinate attention uses a sorted-node-rank window driven by branching
  fraction, not a base-pair radius or generic graph density.
- Original v1 uses incoming-edge messages, without edge attributes in the principal
  launcher; E's bidirectional/sequence-input changes are not assigned to T.
- Original v1 checkpoints resolve through `full_multicohort_server_20260806`.
  Archived code `4f9e59a` masks exact directed rows; canonical RC-aware handling was
  added in `204fbba`. The earlier audit found no detected reciprocal duplicate link
  rows in the released main graphs, but dense exported IDs do not prove absence
  of every reverse-equivalent candidate leak. That residual limitation is explicit.
- Original seed labels controlled split/order generators but did not completely
  seed Torch initialization/DropEdge. They identify independent archived executions.
  Later deterministic campaigns do not retroactively repair historical training.
- The archived launcher/parser supports the stated learning rate, weight decay,
  dropout, minibatches, accumulation, clipping and unusual post-epoch scheduler
  ordering. These are code/config records, not newly inspected server receipts.

Source anchors for these distinctions include `docs/NEXT_STAGE_PHASE0_AUDIT_20260814.md`
(Training/evaluation section), historical `src/training/pretrain.py`, and
`src/models/{gat,dual_stream_gat}.py`. Historical numerical results are unchanged.

## Journal structure references

The [NMI Article guidance](https://www.nature.com/natmachintell/content) specifies
an unheaded Introduction followed by Results, Discussion and Methods, a maximum
150-word abstract, 3,500 main-text words excluding Methods and other excluded
matter, and six combined figures/tables. This pass follows that structure; it is
not a claim of acceptance or complete submission readiness.

Two primary article examples informed the presentation:

- [GROVER, NMI (2024)](https://www.nature.com/articles/s42256-024-00872-0): its Results
  introduce the representation/training choice before downstream validation and
  examine simple input statistics as controls.
- [SegmentNT, Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02881-2):
  its Results open with model design and the corresponding overview figure.

These are structural examples, not new comparison experiments or borrowed claims
about PangenomeFM's performance. The standard-method references were checked at
the primary [focal-loss](https://arxiv.org/abs/1708.02002),
[RoFormer](https://arxiv.org/abs/2104.09864) and
[AdamW](https://arxiv.org/abs/1711.05101) papers.

## Items a writing pass cannot resolve

The exhaustive ledger, `docs/LATEX_COMMENT_RESPONSE_20260928.md`, keeps separate:

- Exact approved consortium author wording, grant numbers/recipients and CRediT.
- Shi-lab repository/weight-hosting destination, license and archival release.
- The optional transparently selected locus vignette.
- Independent E replication and superiority claims; exact donor provenance of
  the processed graph; DUP/complex labels; measured improvements to genotype calls.

These are evidence or author decisions, not silently completed editorial tasks.
No external messages, publication submission, repository transfer or new server
experiment is part of this revision.

## Verification record

- `latexmk -pdf`: success, **31 pages**, zero warnings, unresolved references,
  duplicate citation anchors or overfull/underfull boxes.
- All 31 pages rendered with Poppler and visually inspected; method figure and
  equations also checked at full page scale. Widow/orphan controls applied.
- TeXcount: **141 abstract words; 2,428 main narrative words**, plus 62 heading
  words; captions and Methods excluded. Main displays: five figures and one table.
- All **118** source comment lines covered exactly once in **38** inventory entries.
- All LaTeX labels unique and all cross-reference targets present.
- Completed-evidence builder verified six source-backed numerical tables. Their
  row files are unchanged; evidence and bibliography rebuilds are idempotent.
- Both new Python builders pass Ruff and Python compilation.
- Eight supplementary bibliography entries derive from the maintained bibliography;
  key namespaces prevent main/supplement citation collisions.
- Output: `manuscript/revision_20260924/output/pdf/PangenomeFM_working_revision_20260928.pdf`.
- PDF SHA-256: `38ae621e61bd0288a9e0fa30384a79ff245189b8ab3b4dee436de5b77093c466`.
- Source branch: `codex/v2-evidence-review-20260927`; unrelated worktree files and
  existing untracked result directories were left untouched.

No retraining or new experiment run is claimed by these document checks.
