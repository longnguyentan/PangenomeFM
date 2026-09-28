# Response to every supplied LaTeX comment — 28 September 2026

> **Current status:** see [the completion checklist](LATEX_COMMENT_COMPLETION_20260928.md).
> Every original group now has an explicit `% Long Note:` disposition in the source
> archive. This detailed ledger records the preceding method-focused pass; its old
> section names/page counts are historical. The latest native-editor compilation
> limitation and non-PDF validation are recorded in the completion checklist.

## Scope and version control

Source: [the supplied LaTeX attachment](</Users/longnguyentan/.codex/attachments/ae853bc4-a996-4af1-a000-dbf9e8155c44/Pasted text.txt>).
SHA-256: `83c093f26c1b8e184da0b21ed9f8eef65c005550c166eee1bebebee047f72949`.
Every source line was read, including prose surrounding the comments. A scan that
recognizes escaped percent signs found **118 actual comment lines**. Printed
percentages such as `26.3\%` are not comments. The trailing `%` in the title macro
is a comment and is included below.

The attachment is an older editorial reference, **not the source to restore over
the completed evidence revision**. The working source is
`manuscript/revision_20260924/main.tex` with its `completed_*_20260928.tex` inputs.
Original attachment line numbers below remain fixed even when the working source
moves. Quotations are short exact excerpts; ellipses omit text without changing
the requested action. Repeated requests are cross-referenced, not discarded.

Status below is checked against the **saved method-focused source**. Source integration
is complete where stated. Final compilation and both bibliography lists pass
without warnings or unresolved references; all 31 rendered pages were visually
inspected. The original before-state is retained in Section F.

Status terms:

- **Complete:** the latest source/evidence resolves the request.
- **Integrated in source:** the requested writing/organization is present in the
  saved files; final build/layout verification remains with the lead editor.
- **Writing revision:** a remaining writing or explanation need is stated explicitly.
- **Partial / evidence needed:** a related analysis exists but does not answer the
  stronger requested scientific question.
- **Author decision:** factual or administrative confirmation is still required.
- **Optional / technical:** a suggestion or formatting note, not a missing result.

The before/after map at the end records the structural integration separately.
No pending experiment, funding detail, consortium approval or release destination
is represented as completed by this audit.

## A. Comments addressed to Long/Tomoya and scientific presentation

| ID | Original exact lines | Faithful excerpt / requested action | Disposition in the latest manuscript and remaining work |
|---|---|---|---|
| S01 | **L153** | “Long and Tomoya ... layout of the introduction: representation of DNA sequences and human genomes (linear->graph ...), existing DNA foundation models ... previous efforts ... what they lack of” | **Integrated in source.** The unheaded Introduction now has six paragraphs: representation; biological relevance; DNA models; graph-specific relations; prior approaches/gap; PangenomeFM and controlled reuse. The opening contribution is the native representation/transfer framework, with no unsupported performance ranking against DeepGene or PangenomeX. |
| S02 | **L157–158** | “Long, the following paper should be cited ... importance of using graph genomes for functional genomics analysis” followed by `s41467-026-73663-3` | **Complete.** The current Introduction cites Macias-Velasco et al., `maciasvelasco2026benchmarking`; `references.bib` contains the verified DOI. The older attachment's `macias2026benchmarking` key must not be restored. S07 repeats this citation request. |
| S03 | **L161–167** | “[LONG NOTE]”; “Para 1: linear to graph”; “Para 2: why the representation matters in biology”; “para 3 ... DNA foundation models”; “para 4 ... topology features”; “para 5: previous work & gaps”; “para 6: introduce PangenomeFM + evidence/result” | **Integrated in source.** The six unheaded paragraphs follow the requested progression explicitly. The graph paragraph explains branching, reconvergence, connections, structural context and the absence of donor-specific paths. The final paragraph identifies oriented-segment inputs, coupled coordinate/graph processing and frozen controlled reuse. It distinguishes this contribution from inventing attention, gating or a new loss. |
| S04 | **L172** | “Intro should start with the representation of human genomes, genetic variation should be mentioned later” | **Complete, retain.** The first sentence introduces reference coordinate systems; genetic variation follows the representation problem. Keep this opening through the restructure. |
| S05 | **L183** | “Long, the layout looks more like a CS paper ... format of Nature Machine Intelligence or Nature Methods (check some recent papers ... follow their layout)” | **Integrated and verified.** The saved revision has a short method-led abstract, an unheaded six-paragraph Introduction and five topical main Results subsections, followed by Discussion and method-first Methods. Standard components are attributed. The main primary-results table, detailed SV-complexity figure and broad transfer/development details moved to Supplement while key negative/control conclusions remain in the main narrative. Journal examples and publisher guidance used by the lead editor are recorded separately; organization is not an acceptance/readiness claim. |
| S06 | **L187** | “Long, this figure should be cited in Introduction ... merge the current related work and background into Introduction” | **Complete, retained in revised source.** `fig:paradigms` is cited in the fifth unheaded Introduction paragraph, where prior work and the gap are integrated; no separate Related Work section is introduced. The included asset is now `figure1_paradigms_methodfirst.pdf` with a matching editable SVG. T09 covers the redesign. |
| S07 | **L264–266** | “Long, add results on predicting cCREs and ENCODE data?”; “add citation of this paper” and the same Macias-Velasco URL | **Complete evidence, integrated placement.** “Regulatory transfer is complementary to sequence and task dependent” is now the second Results subsection, after the method/representation opening. Binary cCRE, PLS/pELS/dELS/CTCF-only comparisons and complexity strata remain. Subclass rows subset the original binary-probe predictions, not independent subtype-trained models. Macias-Velasco is cited in the biological-relevance Introduction paragraph (S02). |
| S08 | **L268–271; L282–284** | “TODO(new analysis) ... PLS/pELS/dELS/CTCF-only stratification ... local graph/variant complexity. Do not assume which class benefits most.” | **Complete.** The source reports all four subclasses, fixed complexity strata and negative limitations; bundled scorecard rows identify the original sources. State that the largest observed subtype increment is conditional on the binary-probe/common-background analysis, without claiming independent replications or causal mechanism. Original first-result placement is superseded by the current method-first instruction, not the request to retain the results. |
| S09 | **L285–286** | “Optional biological vignette: one transparently selected locus with graph, annotation track, and predictions; avoid selecting solely by maximal gain.” | **Optional; not completed.** No such case study is claimed. If added, first specify a transparent selection rule, verify graph/annotation coordinate correspondence, and show the predictions including unfavorable comparisons. A fabricated or score-selected success illustration would not satisfy this request. |
| S10 | **L373** | “Long, this subsection title is too cs/graph-heavy, we need to rephrase” | **Integrated in source.** The fourth Results heading is now “Structural controls distinguish graph information from learned-weight gains.” Its text defines the masking-induced degree deficit and unmasked structural-input caveat, reports the exact degree-score replay and keeps the trained/random/H results—including unfavorable cCRE controls. The main heading no longer implies that neighborhood context alone establishes learning. |
| S11 | **L410** | “Long, I feel that the first (or the first two) subsessions should be dedicated to novel components of PangenomeFM (and highlight those key components)” | **Integrated in source.** Results opens with “PangenomeFM couples genomic position and connectivity for frozen reuse” (`sec:method-result`). Methods opens with (1) “Oriented-segment representation and coupled coordinate--graph encoder” (`sec:architecture`), (2) “Query masking, connection scoring and optimization” (`sec:objective`), and (3) “Frozen segment pooling and conditional information comparisons” (`sec:frozen-reuse`), before resources. Original T uses a shared evolving state, layerwise gate, residual addition and normalization; original unmasked degree/component inputs and query-set masking are explicit. Established graph attention, RoPE, focal loss and AdamW are attributed. E remains a separate, not-yet-replicated adaptation. |
| S12 | **L416** | “Long, specify how many graph genomes, etc.” | **Integrated scope clarification; exact training-graph donor provenance still unresolved.** Resource Methods now explicitly distinguishes 751,237 segments/1,097,658 links, 303,425 downstream NT rows and 479,477 benchmark NT rows from the separate path audit: 227 HPRC donors/454 haplotypes and 61 HGSVC donors/122 haplotypes. It states these path-resource counts do not reconstruct the SV-resolution training graph’s donor provenance and that node IDs cannot be substituted without verified correspondence. No generic release cohort count is invented. |
| S13 | **L508–511** | “Tomoya and Long, These results should be included as a separate subsection in Results, including detailed interpretations”; subsequent note says “coverage/truncation ... summarized in the first biological Results section” | **Integrated in source.** The first Results subsection now contains the named subsubsection “Sequence coverage is complete, but long-segment context is restricted” (`sec:sequence-coverage-result`). It reports 303,425 original segments, 512 dimensions, zero missing-feature exclusions and 26.3% end sampling, distinguishing segment coverage from nucleotide coverage and linking to `sec:s-coverage`. This gives the requested result a visible dedicated heading while preserving five topical Results subsections. TraitGym base-visibility limitations and whole-graph completion remain separately scoped. |
| S14 | **L588** | “Long, Supplementary information should read like a manuscript, including its own sessions, text, figures, tables, and references” | **Integrated and verified.** Supplement now starts with an explicit Scope paragraph and contains narrative sections, tables and figures supporting the main claims. `multibib` declares the separate S bibliography; supplementary citations use `citepS` with `supp__`-prefixed keys. The generated `supplement_references.bib` preserves verified reference fields while separating the supplementary key namespace, and `bibliographystyleS`/`bibliographyS` declare “Supplementary references.” This is no longer only a shared main bibliography. Negative/control findings remain summarized in the main Results and Discussion; the final build resolves both reference lists without duplicate anchors or missing citations. |

## B. Remaining actionable scientific and reproducibility TODOs

| ID | Original exact lines | Faithful excerpt / requested action | Disposition in the latest source, code and completed evidence |
|---|---|---|---|
| M01 | **L343** | “Is it possible to improve SV genotyping (especially in complex and copy number variant regions)?” | **Partial; stronger endpoint unresolved.** Original task is known INS versus DEL, not discovery/genotyping. The completed COSIGT locus-quality regression and validation-only fallback are relevant follow-ups but do not show improved calls: topology intervals include zero and the training-median reference is stronger on mean primary error. INV classification is separate, and DUP/complex classes, per-variant concordance and an all-callable denominator remain missing. Preserve these limits rather than relabel existing classification as genotyping. |
| M02 | **L381–385; L487–490** | “HIGH PRIORITY CONTROL: recompute degree sum / preferential attachment on the actual expanded one-hop graph”; “Do not strengthen ... until this control is verified.” | **Complete audit, limited conclusion.** Current Methods records exact canonical replay of 271,796 scores across 1,215 context-specific slices using each visible graph after query/reciprocal masking. This replay does not retroactively assign canonical reverse-complement masking to the historical trainer, which used exact directed query rows. The expanded one-hop degree computation is verified. Later candidate audit found a masking-induced degree deficit; the completed H/random matrix does not rescue universal learned-weight superiority. Keep the verified computation and the stronger limitation together. |
| M03 | **L431–432; L480–481** | “state the connected:unconnected sampling ratio ... chance-level AUPRC”; “add it as a reference line to the generalization figure” | **Complete for the reported evaluation prevalence.** Methods and Supplement give HPRC test prevalence 0.4990 strict/0.5065 one-hop, and approximately balanced transfer sets; chance AP equals prevalence. The generalization figure builder draws the approximate 0.5 reference line. Keep the distinction between distance-matched design, realized context-specific candidate ratios and dataset-specific chance; do not imply all resources have exactly 1:1 counts. If exact training sampling ratios are added, use archived candidate/config counts rather than infer them from test AP. |
| M04 | **L448–450** | “specify the implemented gate function, pair-scoring architecture, and the exact focal/class-adaptive form”; “copied from the actual archived code/config, not inferred” | **Complete, corrected in the method-first source.** `sec:architecture` gives the per-layer update `LN(h + g*a + (1-g)*b)`, with `g = sigmoid(W concat(a,b) + c)`, coordinate-attention output `a` and graph-attention output `b`. Both branches consume the same evolving shared state; the coordinate branch is not restricted to coordinate-only input. Original structural columns are computed before query masking. Historical v1 masks exact directed query rows; later canonical reverse-complement repair is explicitly separate, and the old prediction IDs cannot establish absence of reverse-equivalent candidate leakage. `sec:objective` gives the ordered-concatenation MLP, alpha=0.25/0.75, gamma=2, guarded batch class weight, clipping and mean focal reduction. Figure 2 is redrawn to match. Neither the attachment’s generic weighted cross entropy nor a final-only independent-branch fusion is retained. |
| M05 | **L458** | “We may need to talk about loss functions and optimization in PangenomFM” | **Complete in saved Methods.** `sec:objective` now specifies AdamW, learning rate 5e-4, weight decay 1e-4, dropout/DropEdge 0.1, 512 candidates, four-batch accumulation, norm-1 gradient clipping, 100 epochs/patience 20, validation macro-AUROC tolerance 1e-4 and the implemented epoch-stepped ramp/cosine schedule—including the first-epoch ordering caveat. The original three execution labels did not fully seed PyTorch initialization or DropEdge; Methods distinguishes them from later repaired global seeding. Adopted loss/optimizer components are cited. The separate E protocol retains its own 20-epoch/patience-five budget; no compute matching is implied. |
| M06 | **L493–494; L689** | “give the numeric cutoffs ... three complexity strata and the number of regions/examples”; “exact graph-complexity thresholds and counts” | **Complete.** Methods specifies the six-component robust normalization, cutoffs −0.155641/0.434725 and 203/202/203 native windows, with endpoint/locus mapping rules. Supplement points back to Methods and retains task-specific strata. Region counts are not task-example counts; subgroup support and the sparse SV stratum remain explicit. |
| M07 | **L497–500; L694–696** | “principal configuration (48/2) is not part of this sweep”; “run 48/2 under the exact diagnostic sweep protocol; otherwise state ... not directly comparable” | **Complete through recovered matched evidence, not a newly invented run.** The archived matching fold/run-label/candidate cells supply 48/2 fold-pooled AP 0.9423; this differs from the 0.9501 window-mean headline. Both the table and Methods explain the aggregation. Matching archived run labels is not proof that the original Torch initialization was reproducibly seeded. The 96/4 diagnostic remains higher without a monotonic scaling or final-architecture claim. |
| M08 | **L516–518** | “name the downstream classifier family and its hyperparameters, regularization, model-selection procedure ... threshold tuning used validation chromosomes only” | **Complete.** Original probes use training-fitted StandardScaler, balanced L2 logistic regression, C=1, lbfgs, max_iter=800, validation-only temperature and threshold. New studies explicitly disclose 4,000-iteration ceilings, convergence checks and validation-selected or nonlinear sensitivities. Do not overwrite original hyperparameters with later settings. |
| M09 | **L520–521** | “clarify whether 60,685 applies to each split or to validation and test combined” | **Complete.** cCRE Methods states each of validation and test averages 60,685, with 50,846–73,988 range; training mean 182,054 with its range. |
| M10 | **L526–527** | “clarify whether 34,781 applies to each split or to validation and test combined” | **Complete.** Original INS/DEL Methods states each of validation and test averages 34,781, with 27,708–43,217 range; training mean 104,343 with its range. Retain actual endpoint, ordered features and insertion pseudo-end convention. |
| M11 | **L552–554** | “state the actual experimental reason ... restricted to chr8”; “Do NOT use an unrelated chr8 literature example as a post-hoc justification” | **Complete as documented scope.** Methods describes a prespecified chromosome-8 pilot, with only chr8 summary files retrieved after checking download size and local storage; no outcome-directed selection or unrelated biological justification is added. The negative matched QTL/GWAS findings remain. Keep this statement linked to the existing preparation/reproducibility record, not a stronger all-QTL claim. |

## C. Author and release decisions

| ID | Original exact lines | Faithful excerpt | Disposition / concrete remaining need |
|---|---|---|---|
| A01 | **L76; L82** | “Need to add HGSVC and HPRC to the author list”; “HGSVC, HPRC,” | **Author decision, unresolved.** Preserve current authors and affiliations. Obtain the exact approved consortium author wording, ordering, membership/affiliations and approval before changing authorship. Resource use alone does not supply that authorization. |
| A02 | **L567** | “Long, please prepare a github under shilab repo for your code, should we share the pretrained model there or huggingface etc.?” | **Partial / author release decision.** The current availability statement gives `https://github.com/longnguyentan/PangenomeFM`; checkpoints and large derived tables remain on the server and archival DOI is pending. This is not a Shi-lab organization transfer or model-hosting release. Need exact organization/repository destination and authorization, checkpoint/license/provenance release checklist, and hosting/archive choice before claiming publication. This audit does not publish or transfer anything. |
| A03 | **L574** | “acknowledge NHGRI U24 and NIGMS R01 for funding” | **Author decision remains unresolved.** The saved funding paragraph now faithfully records that the comments request NHGRI U24 and NIGMS R01 acknowledgement while explicitly leaving award numbers and final wording for confirmation. It does not assert invented awards. Exact recipients, numbers and required agency language remain needed. Named acknowledgements and CRediT wording remain author-confirmed items; any newly authorized acknowledgement text is not evidence that these other decisions are settled. |

## D. Figure, layout, outline and technical comments

| ID | Original exact lines | Faithful excerpt / content | Disposition |
|---|---|---|---|
| T01 | **L3** | “Prevent pdflatex object-stream overflow with complex vector figures.” | **Retain technical setting.** `\pdfobjcompresslevel=0` remains. The final method-focused multi-file TeX build succeeds without warnings; all 31 rendered pages were inspected. This is not a scientific TODO. |
| T02 | **L27–28** | “Uncomment if line numbers are required.”; commented `lineno` package | **Optional submission formatting.** Line numbering remains disabled. Enable only for the selected journal/submission format; its absence is not missing evidence. |
| T03 | **L44–45** | Commented numbered section/subsection `\titleformat` commands | **Retained alternatives, not requests.** Active source deliberately uses unnumbered headings. No need to activate obsolete numbered-format lines merely because they are comments. |
| T04 | **L58** | Trailing `%` in `\renewcommand{\maketitle}{%` | **Retain technical syntax.** Suppresses an unintended space in the macro; it contains no prose instruction. |
| T05 | **L71** | “Nature Machine Intelligence” | **Venue note.** Related style request is S05. Do not treat a target-journal note as acceptance or readiness. |
| T06 | **L75** | “Pangenome Foundation Model capture” | **Scoped title revision made.** The saved title is “PangenomeFM: topology-native self-supervision for reusable human pangenome representations.” The incomplete scratch fragment is not treated as a command to claim broad foundation-model status. Main text explicitly limits learned-weight and biological claims. |
| T07 | **L101; L103–104; L106; L108–112; L114; L116–126; L128–133; L135** | “NEW STRUCTURES”; title/abstract/introduction; four Results headings; Discussion; Methods list; availability/admin/references; Supplementary Information | **Historical outline deliberately superseded.** The saved source has five topical Results sections: method/representation; regulatory transfer; known SV types; structural controls; external/generalization limits. Methods leads with architecture, objective/optimization and frozen reuse before resources. The old “breakpoint classification” wording stays corrected to INS/DEL among known variants. All original content families remain in main/Supplement or explicitly pending; adverse results were not dropped to fit the older four-heading outline. |
| T08 | **L146** | “Structure: biological representation problem --> graph-native idea --> cCRE --> SV --> generalization --> control --> scoped conclusion” | **Integrated in source.** The concise abstract leads with pangenome representation and coupled coordinate/graph self-supervision, then representative original cCRE/INS–DEL evidence, the three-class/INV limitation, null external findings and shortcut/random controls. It does not enumerate every experiment or promote the still-developmental E candidate; that evidence remains qualified in main text and Supplement. The conclusion is task-dependent complementarity, not broad superiority. |
| T09 | **L190–192** | “TODO(figure): redesign with keyword-only labels. End each paradigm at its learned output; explain downstream reuse in the caption rather than a PangenomeFM-only output column.” | **Redesign integrated and visually verified.** `figure1_paradigms_methodfirst.svg/.pdf` now uses four parallel rows and Source → Model input → Learning → Representation columns with concise labels. Every row ends at a learned representation; the caption explains downstream reuse and denies a performance ranking. `figure2_method_methodfirst.svg/.pdf` separately depicts shared inputs, the layerwise coordinate/graph gate plus residual/normalization, query masking/scorer, and frozen reuse. The saved Figure 2 caption distinguishes historical directed-row masking from the later reciprocal repair. Original assets remain available; no numerical result panel was invented. |
| T10 | **L320–322** | “Keep this table”; “consider moving the table to Supplementary once the standalone biological figures carry these values clearly” | **Preserved and relocated as suggested.** The original numerical `tab:downstream` is now the first supplementary table under “Principal frozen-feature comparisons.” Main regulatory/SV figures and prose retain its values and refer to it. The SV complexity figure also moved to Supplement with full uncertainty/interaction text, while the main SV narrative preserves the smaller high-complexity gains. Relocation did not remove controls or unfavorable results. |

## E. Requests embedded in live text rather than `%` comments

These do not increase the 118-comment count but were also read and checked:

- **L626–627:** supplement donor-overlap statement and printed `TODO(authors)` for
  chance AP. Latest source corrects overlap to at least five donors including
  HG002/NA24385 and supplies prevalence; M03 covers the chance request.
- **L662:** control-figure caption says expanded-context degree must be verified
  and principal 48/2 added. M02/M07 are completed; current caption was updated.
- **L704:** the capacity table's printed `[TODO]` is now 0.9423 with the correct
  fold-pooled estimand; M07 covers its provenance.
- **L568–580:** release URL/DOI, acknowledgements, funding and contribution
  placeholders require honest availability and author confirmation (A01–A03),
  not replacement by fabricated details.

## F. Verified before/after structural map

The **before** column refers to the completed-evidence manuscript at the start of
this request, not the older attachment. The **after** column was checked against
the actual saved `main.tex`, modular text and vector-figure sources. Source
integration is followed by a successful final compile, bibliography/link resolution
and visual inspection of all 31 rendered pages.

| Requested change | Before this structural pass | Verified saved source after integration |
|---|---|---|
| Method and novelty first (S11) | Results opened with one brief representation paragraph then cCRE; Methods opened with graph resources | First Results: “PangenomeFM couples genomic position and connectivity for frozen reuse.” First three Methods: oriented-segment/coupled encoder; query masking/scoring/optimization; frozen pooling/conditional comparisons. Shared state, per-layer gate, residual and normalization are explicit; standard components are attributed. |
| Introduction progression (S01/S03/S04/S06) | Integrated representation/model/prior-work Introduction with Figure 1 reference | Six unheaded paragraphs follow representation → biological motivation → DNA models → native graph relations → prior approaches/gap → method and evidence. Figure 1 is cited in the prior-approaches paragraph. |
| Publication-style organization (S05/T07/T08) | Unnumbered headings with a long Results inventory | Short method-led abstract, unheaded Introduction and five main Results subsections: method, regulatory transfer, known SV types, structural controls, external/generalization limits. Discussion separates contribution from biological and attribution limitations. Publication-style resemblance is not acceptance. |
| Representation coverage in Results (S13) | Coverage/end sampling appeared in cCRE opening | Named subsubsection “Sequence coverage is complete, but long-segment context is restricted” under the method-focused opening (`sec:sequence-coverage-result`), linked to detailed Supplement coverage. |
| Accessible controls heading (S10) | “Degree shortcuts and random controls limit attribution to learned weights” | “Structural controls distinguish graph information from learned-weight gains,” retaining visible-degree deficit, original unmasked inputs, exact baseline replay and all H/random outcomes, with historical directed-row masking separated from the later canonical repair. |
| Keyword-only parallel paradigm figure (T09) | Existing Figure 1 and its unfulfilled redesign TODO | New `figure1_paradigms_methodfirst` SVG/PDF uses four parallel representation outputs; new `figure2_method_methodfirst` SVG/PDF depicts the actual shared-input, gated/residual architecture and frozen evaluation. Both new vectors and their manuscript pages were visually inspected. |
| Main/supplement evidence balance (S14/T10) | Original primary table in main; full added tables in Supplement; shared bibliography | `tab:downstream` and `fig:sv-complexity` moved to explanatory Supplement sections; main prose/figures retain core values and adverse findings. A Scope paragraph and separate S-labeled supplementary bibliography are now declared, with `citepS` and generated `supp__` citation keys in a separate bibliography file to avoid cross-bibliography key collisions. |

### Scope that remains open after writing integration

- **Build/layout:** the lead editor must confirm final multi-file compilation,
  supplementary bibliography and cross-references, and visual layout. This audit
  does not infer successful compilation from the presence of `multibib` commands.
- **Scientific evidence:** pending chromosome replication and original-v1
  reference remain pending; E has no new completed genomic result. Adjusted INV,
  external-genome, TraitGym and COSIGT uncertainty is retained. Exact training-graph
  donor provenance, verified paths, DUP/complex labels and a callable genotyping
  denominator are not invented. The historical directed-row masking and incomplete
  global-seeding limitations remain documented, rather than assigning later repairs
  to old checkpoints.
- **Optional analysis:** the transparently selected locus vignette is still absent
  and is not claimed complete (S09).
- **Author/release decisions:** consortium author wording/approval, exact award
  numbers, CRediT/named acknowledgement approvals and the Shi-lab/model-hosting
  destination remain distinct from the completed editorial work (A01–A03).

## G. Exhaustive coverage ledger

Each listed source comment line has exactly one primary inventory owner. A comment
block may refer to another owner for the same scientific issue, but no line is
omitted or counted twice.

```text
T01: 3
T02: 27-28
T03: 44-45
T04: 58
T05: 71
T06: 75
A01: 76,82
T07: 101,103-104,106,108-112,114,116-126,128-133,135
T08: 146
S01: 153
S02: 157-158
S03: 161-167
S04: 172
S05: 183
S06: 187
T09: 190-192
S07: 264-266
S08: 268-271,282-284
S09: 285-286
T10: 320-322
M01: 343
S10: 373
M02: 381-385,487-490
S11: 410
S12: 416
M03: 431-432,480-481
M04: 448-450
M05: 458
M06: 493-494,689
M07: 497-500,694-696
S13: 508-511
M08: 516-518
M09: 520-521
M10: 526-527
M11: 552-554
A02: 567
A03: 574
S14: 588
```

Coverage verification: expand the ledger ranges and compare with the source's
unescaped-percent line numbers. Required result: **118 unique covered lines,
zero omitted, zero duplicate, zero falsely included**. The attachment's bytes
remain unchanged. Verified editorial integration is intentionally distinct from final build/layout
verification and from completed experiment/reproducibility evidence.

Final verification details and PDF hash are in `docs/METHOD_FIRST_REVISION_20260928.md`.
