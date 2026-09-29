# EN-TEx and PangenomeFM: explained for a 15-minute team discussion

**Results reviewed: 29 September 2026.** This briefing explains the biological targets, model comparison and interpretation. The detailed tables and quality checks remain in the same document after the timed discussion.

## 0–1 min · Main finding

Adding information from the pangenome graph gives **small improvements in some biological measurements**, but not a consistent improvement across all EN-TEx tasks. Some positive findings weaken when we account for how often a genomic location was measured.

The completed analysis covers **450 experiment settings across 15 datasets**, with seven model-input combinations in each setting. These are repeated model evaluations, not 450 independent donors. EN-TEx contains only four donors in this analysis.

The current results support further testing of graph information. They do **not yet show that pretraining the graph encoder is better than using an otherwise identical encoder with random weights**.

## 1–3 min · The biology behind each task

EN-TEx combines personal genomes with functional measurements across tissues from four donors.

People usually carry two copies of each autosomal genomic region, one inherited from each parent. At a position where the copies differ, sequencing reads can sometimes be assigned to one copy or the other. **Allele-specific imbalance** means that the supplied statistical test detects unequal activity between those copies. It does not automatically identify a disease-causing variant.

We use three types of prediction:

1. **Regulatory regions prone to unequal activity.** Candidate cis-regulatory elements are DNA regions with evidence of a gene-regulatory role. A region is positive if EN-TEx reports significant imbalance in at least one informative experiment; measured regions with no significant call are negative. There are 250,722 regions, including 28,092 positives, or 11.20%.
2. **Active versus repressed enhancers.** Enhancers help regulate gene expression. We restrict this comparison to annotated distal enhancer-like regions, rather than comparing promoters with enhancers. Each tissue has its own classifier. The final tissue average gives equal importance to all five tissues.
3. **Imbalance measured at single-base variants.** A single-nucleotide variant is a difference at one DNA base. Here it helps distinguish the two alleles when counting reads. The task predicts unequal biological signal at an informative variant; it does not predict whether the variant exists.

The six variant-based tasks measure different biological signals:

| Measurement | Biological meaning |
|---|---|
| CTCF protein binding | CTCF is a DNA-binding protein involved in organizing chromosome contacts. The assay measures its association with DNA |
| H3K27ac histone signal | A chemical modification of a DNA-packaging protein, often associated with active enhancers and promoters |
| RNA production | RNA sequencing measures gene-expression output; unequal RNA reads can indicate unequal expression of the two alleles |
| DNA accessibility, measured by ATAC-seq | Measures how accessible DNA is to the assay enzyme, reflecting how tightly it is packaged |
| H3K4me3 histone signal | A histone modification commonly associated with active gene promoters |
| H3K27me3 histone signal | A histone modification commonly associated with repressed chromatin |

**For every row, the target is imbalance between alleles, not simply the presence of binding, expression or a histone mark.** The model gives one prediction per genomic location; it cannot specify which allele is favoured in a particular donor.

## 3–5 min · Data checks and the model comparison

**Data issues resolved before analysis**

- The initially supplied high-confidence variant file contains **RNA measurements only**. Other assays come from the complete accessible-variant file, with 22,310,439 measurements across 12 assays. The six analysed tasks contain 1.29–3.27 million measurements each, with only 1.44–4.50% labelled positive.
- A newer regulatory-element registry matched only **31.6% of active and 41.6% of repressed records**. The older ENCODE version 2 registry matched all supplied identifiers, avoiding unequal loss of the two classes.
- Five enhancer tissues were selected by sample size, before examining performance: thyroid gland, tibial nerve, body of pancreas, gastroesophageal sphincter and Peyer's patch. Conflicting active/repressed labels were excluded. Each tissue retained approximately 206,000–261,000 regions.
- All regulatory-region and variant locations mapped to the original HPRC release 2 graph. Here HPRC means Human Pangenome Reference Consortium. About 1.33% of regulatory regions overlap more than one graph segment. The original enhancer-task record also reports complete mapping.
- The four initial tables had no recorded missing values. Checks covered coordinates, counts and labels. We did not independently repeat sequencing-library quality control.

**Three score columns, one controlled comparison**

- **Without graph:** genomic position/structural descriptors plus DNA-sequence features from the Nucleotide Transformer model.
- **Graph within window:** the same inputs, plus PangenomeFM features using the graph inside each genomic window. This is called “strict” in the detailed tables.
- **Graph with neighbours:** the same inputs, with graph features also using directly connected segments. This is called “one-hop”.

An encoder converts sequence or graph structure into numerical features for prediction. Both pretrained encoders stay **frozen**, meaning their weights are unchanged. Only the same simple downstream classifier is fitted with biological labels. We test on five groups of chromosomes withheld from fitting, with three pretraining runs per graph context. All observations at the same location stay in the same partition, including repeats across donors and tissues.

## 5–8 min · Results, with the table explained

Every number below is **average precision**, a score summarizing how well the model retrieves true positive examples as the prediction threshold changes. Higher is better. It is **not classification accuracy**: a score of 0.05 does not mean “5% of predictions are correct”.

For a random ranking, the expected score is approximately the positive fraction. Because that fraction differs between tasks, compare models **within a row**, not raw scores between biological tasks.

| Biological prediction | Without graph | Graph within window | Graph with neighbours |
|---|---|---|---|
| Regulatory regions prone to unequal activity between alleles | 0.1514 | 0.1516 | 0.1508 |
| Active versus repressed enhancers, average across five tissues | 0.5699 | 0.5708 | 0.5707 |
| Unequal CTCF protein binding between alleles | 0.0616 | 0.0630 | 0.0638 |
| Unequal H3K27ac signal between alleles | 0.0497 | 0.0504 | 0.0530 |
| Unequal RNA production between alleles | 0.0255 | 0.0271 | 0.0270 |
| Unequal DNA accessibility between alleles | 0.0473 | 0.0470 | 0.0476 |
| Unequal H3K4me3 signal between alleles | 0.0803 | 0.0847 | 0.0893 |
| Unequal H3K27me3 signal between alleles | 0.0915 | 0.0949 | 0.0988 |

**Worked example:** for unequal H3K27ac signal, the score rises from 0.0497 without graph information to 0.0530 with graph neighbours. The paired mean improvement is approximately 0.0033, or 0.33 percentage point of average precision. This is a small improvement in ranking rare positive events; it is not a 0.33-point increase in classification accuracy.

**Uncertainty matters.** A 95% interval describes uncertainty in the estimated improvement. If it includes zero, these runs do not establish a clear benefit. We estimate intervals by resampling chromosome groups and runs while keeping each model comparison paired.

- **Regulatory-region susceptibility, enhancer state, RNA, accessibility and H3K27me3:** overall improvement remains inconclusive.
- **CTCF binding:** positive primary intervals in both graph settings, but sensitive to how repeated measurements are weighted.
- **H3K27ac:** positive primary interval with neighbours; the within-window primary interval includes zero.
- **H3K4me3:** positive primary intervals in both settings, but the follow-up checks below weaken that result.

The enhancer average is inconclusive even though tibial nerve has a positive within-window interval. The other tissue/settings include zero. We retain every tissue, including negative estimates for Peyer's patch.

## 8–11 min · Follow-up checks and their meaning

**Frequently measured locations have more influence.** A location measured 20 times can contribute 20 observations, while another contributes only one. Our follow-up gives each location the same total weight during evaluation. The classifier is not retrained in this check.

| Follow-up | Observed change | Interpretation |
|---|---|---|
| Equal weight per location: CTCF binding | Improvements shrink to +0.00018 within-window and +0.00049 with neighbours; both intervals include zero | The primary finding is not stable under this weighting choice |
| Equal weight per location: H3K27ac | Improvements +0.00199 and +0.00221; both intervals remain above zero | A small signal worth independent confirmation |
| Equal weight per location: H3K4me3 | Improvements shrink to +0.00107 and +0.00115; both intervals include zero | Frequently measured locations influence the larger primary gain |
| Regulatory regions matched by number of measurements | Improvements +0.00148 and +0.00200; both intervals above zero | A different, balanced population with 50% positives; coordinate features alone still outperform the largest feature combination |

We also prepared regulatory-region tasks restricted to H3K27ac or CTCF before fitting. Positive intervals occur only in the within-window H3K27ac analysis and the neighbour-expanded CTCF analysis, rather than consistently across settings.

**Testing many comparisons increases the chance of a positive-looking result.** In the additional accessibility/H3K4me3/H3K27me3 panel, none of the average-precision improvements passes the multiple-comparison check. That check has limited resolution because only five chromosome groups contribute. Exact intervals and adjusted values are retained in Appendix D.

**More graph complexity does not mean more model benefit.** The fraction of regulatory regions with imbalance rises from 9.04% to 10.20% to 13.23% across low, medium and high complexity. However, adding graph features does not show a clear gain within the high-complexity group. With neighbours included, the low- and medium-complexity gains are negative.

The CTCF/H3K27ac weighting analyses were designed after seeing the initial results and remain exploratory. The three additional assays had their follow-up protocol fixed before fitting. Reweighting suggests sensitivity to measurement frequency; it does not prove that measurement frequency caused the observed gains.

## 11–13 min · Limits of the current evidence

- **Shared location-level features:** cannot distinguish donors, tissues or two alleles at the same position. Many observations from four donors do not become many independent individuals.
- **Sequence coverage:** long segments use sampled terminal sequence rather than every base; the sequence features are not centred on each tested allele. A stronger sequence baseline could change the graph comparison.
- **Pretraining:** the original reconstruction task contained a masking shortcut. Random initialization and the random removal of graph edges during training were not fully controlled by the recorded run identifiers. Frozen extraction does not remove these limitations.
- **Generalization:** withheld chromosomes are not withheld donors. Further model choices made on repeatedly examined folds need untouched confirmation data.
- **Feature coverage:** complete in 180 core settings; at least 99.9983% in 150 enhancer settings. Later 120 RNA/additional-assay settings lack a uniform exported feature audit, so every-run coverage is not independently re-certified here.
- **Completed work:** 450 settings and 3,150 feature-specific metric records, plus 150 evaluations of existing predictions. Saved means and paired differences were checked; this report did not retrain models or recompute confidence intervals.

## 13–15 min · Potential ideas

- **Separate graph information from learned weights:** compare trained encoders, matched random-weight encoders and simple graph statistics on the same examples. Include the newer sequence-conditioned encoder, which also receives sequence information.
- **Change training weights as well as evaluation weights:** prevent heavily measured locations from dominating fitting; retain the existing chromosome partitions.
- **Represent the actual alleles:** compare local sequence around each variant with the current segment representation; verify that the variant base enters the sequence model.
- **Retain the full six-assay panel:** use H3K27ac as a hypothesis for confirmation and H3K4me3 as a measurement-frequency sensitivity example; keep the null results.
- **Strengthen confirmation and audit coverage:** reserve external or donor/tissue-aware validation, complete the late-run feature audit, and test length-weighted pooling for multi-segment regulatory regions.

**Take-home message:** graph information may help identify some locations prone to allelic imbalance. The evidence is small and depends on the assay and evaluation choices. The next step is to test whether learned graph representations add information beyond simpler inputs on independent data.

---

# Detailed reference: retained in this document

The 15-minute discussion ends above. The following glossary explains the notation used in the exact tables. All detailed scientific tables from the preceding version remain unchanged, including confidence intervals, null results, sample counts and quality checks.

## Reading the detailed tables

### Biological labels and task codes

| Term in the tables | Meaning in this study |
|---|---|
| Endpoint / task | The biological outcome being predicted |
| Locus | A genomic location or interval; the same locus can be measured repeatedly |
| Allele / heterozygous | One of the sequence alternatives at a locus; heterozygous means the two inherited copies differ |
| AS | Allele-specific imbalance: a supplied significant difference in signal between the two alleles |
| AS-prone | A location with at least one significant imbalance call across its informative measurements |
| cCRE | Candidate cis-regulatory element: a DNA region with evidence of a regulatory role |
| dELS | Distal enhancer-like signature: the annotated enhancer-like class used in the active/repressed task |
| P0 | AS-prone regulatory-region prediction; P0b uses its predictions for graph-complexity comparisons |
| P1 | Active versus explicitly repressed distal enhancer-like regions, analysed separately in five tissues |
| P2 | Variant-based imbalance tasks initially defined for CTCF and H3K27ac; later assays extend this design |
| SNV | Single-nucleotide variant: a difference at one DNA base; informative heterozygous sites allow allele-specific read counting |
| CTCF | CCCTC-binding factor, a DNA-binding protein involved in genome organization; here the target is unequal binding signal |
| H3K27ac | Acetylation at lysine 27 of histone H3; associated with active regulatory regions. The task predicts unequal signal between alleles |
| H3K4me3 | Trimethylation at lysine 4 of histone H3; commonly associated with active promoters. The task predicts unequal signal |
| H3K27me3 | Trimethylation at lysine 27 of histone H3; commonly associated with repression. The task predicts unequal signal |
| RNA ASE | Allele-specific expression measured using RNA sequencing; unequal expression of the two alleles |
| ATAC-seq | Assay for transposase-accessible chromatin using sequencing; measures DNA accessibility. The task predicts unequal accessibility signal |
| ChIP-seq | Chromatin immunoprecipitation followed by sequencing; profiles protein binding or histone modifications |
| Prevalence / positive fraction | Fraction of examples labelled positive; for example, 3.51% of CTCF measurements |
| Exposure / informative measurements | Number of experiments in which a locus had a measurable, testable observation; not environmental exposure |

### Model inputs and evaluation

| Symbol / term | Plain-language interpretation |
|---|---|
| C | Coordinate and structural descriptors: segment position, length and orientation |
| K | Counts of short DNA words, called k-mers; a simple sequence-composition baseline |
| S | DNA-sequence features produced by the frozen Nucleotide Transformer model |
| T | Graph features produced by the frozen PangenomeFM encoder |
| C+S | The main model-input baseline: coordinates plus sequence, without learned graph features |
| C+S+T | The same baseline with PangenomeFM graph features added |
| C+T | Coordinates plus graph features, used to estimate the extra contribution of sequence |
| H / R / E | Simple handcrafted graph statistics / a random-weight encoder / the newer sequence-conditioned encoder. Their matched EN-TEx comparison is not completed here |
| Frozen encoder | A representation model whose parameters are not changed using EN-TEx labels |
| Probe | The downstream classifier trained on those fixed representations |
| Strict / one-hop | Graph inside the selected window / graph expanded to directly connected neighbours |
| Fold / chromosome-held-out | One chromosome partition reserved for evaluation; observations at a locus stay together |
| Seed / run identifier | Recorded repeat identifier; historical encoder randomness was not fully controlled |
| RNG / DropEdge | Random-number generation / random removal of graph edges during training |
| Five-tissue macro | Compute each tissue's score, then average the five scores with equal tissue weight |
| Donor/tissue macro | Equal-weight average across eligible donor or tissue groups; does not imply those groups were withheld during fitting |
| Measurement weighting | Every measurement counts once, so frequently measured loci can count more often |
| Equal-locus weighting | Each locus has equal total evaluation weight; a locus measured 20 times gives each observation weight 1/20 |
| Exposure matched | Positive and negative loci paired by chromosome and number of informative measurements |
| Graph complexity | The predefined measure of how structurally variable a graph region is; bins were not chosen using prediction results |

### Scores and uncertainty

| Metric / notation | Meaning and reading rule |
|---|---|
| AP / AUPRC | Average precision, called AUPRC in this code: summarizes precision across recall thresholds. Higher is better; not accuracy |
| Precision / recall | Fraction of predicted positives that are true positives / fraction of true positives retrieved |
| F1 | Harmonic mean of precision and recall at the selected decision threshold |
| Balanced accuracy | Average of sensitivity and specificity; gives the positive and negative classes equal importance |
| AUROC | Area under the receiver operating characteristic curve; ranking measure with random baseline 0.5 |
| ΔT / ΔAP | AP with coordinates + sequence + graph minus AP with coordinates + sequence; positive values favour adding graph features |
| ΔS | AP with coordinates + sequence + graph minus AP with coordinates + graph; estimates the extra sequence contribution |
| Normalized AP | (AP − prevalence)/(1 − prevalence); adjusts the chance baseline but does not make different tasks directly comparable |
| Mean / SD | Average across runs / standard deviation describing variation across runs |
| 95% CI | A 95% confidence interval for the estimate. An interval including zero does not establish a clear positive gain |
| Paired comparison | Compare models on the same fold, run and examples before summarizing differences |
| Hierarchical bootstrap | Resample chromosome groups, then runs within groups, to estimate uncertainty while preserving the comparison pairs |
| Pointwise interval | Uncertainty for one comparison; does not adjust for inspecting many comparisons |
| Multiplicity / BH q | Multiple-comparison analysis / Benjamini–Hochberg adjusted P value; applied to the stated comparison family only |
| Sign-flip test | Repeatedly reverse the signs of fold-level differences to evaluate the null of no consistent directional gain |
| Inconclusive | The evaluated runs do not establish a clear improvement; this is not proof that the true effect is exactly zero |
| Post hoc / exploratory | Designed after inspecting earlier results; requires independent confirmation |
| QC / feature coverage | Quality control / fraction of examples retaining the required model inputs |
| HPRC / ENCODE / SCREEN | Human Pangenome Reference Consortium / Encyclopedia of DNA Elements / its candidate-regulatory-element registry browser |
| GRCh38 | The human reference assembly used for genomic coordinates |
| SHA256 | A file fingerprint used to verify that the same resource version was analysed |

## Appendix A. Source data and QC

### A.1 Source files and schemas

| Source | Bytes | Rows | Unique loci/IDs | Compression | Recorded missing values |
|---|---|---|---|---|---|
| cCREs_default_AS.tsv | 745298142 | 5330335 | 250722 | none | 0 |
| hetSNVs_high-confidence_AS.tsv | 186460228 | 1640580 | 466867 | none | 0 |
| active.combined_set.txt.zip | 17660571 | 5646598 | 635906 | zip | 0 |
| repressed.combined_set.txt.zip | 14834438 | 4620982 | 778533 | zip | 0 |
| hetSNVs_default_AS.tsv | 2675757003 | 22310439 | assay-specific below | none | validated during preparation; no all-column missingness inventory |

The four initial files were already downloaded and were not replaced. ZIP inspection includes member integrity; `__MACOSX` metadata is not parsed as observations. Each annotation ZIP has a headerless tab-delimited data member (`ccre_id,state,tissue`). Both AS TSVs have headers and are tab-delimited. Coordinates are GRCh38, `chr`-prefixed, zero-based and half-open; an SNV has `end = start + 1`. Retained prepared loci are autosomal; five-fold definitions also list X/Y for consistency with the manuscript.

**cCRE schema:** `chr, start, end, region_id, hap1_count, hap2_count, experiment_accession, donor, tissue, assay, hap1_allele_ratio, p_betabinom, imbalance_significance`.

**SNV schema:** `chr, ref_start, ref_end, ref_allele, hap1_allele, hap2_allele, experiment_accession, donor, tissue, assay, cA, cC, cG, cT, ref_allele_ratio, p_betabinom, imbalance_significance`.

Read counts and supplied p-values remain provenance fields. Labels use `imbalance_significance`, not a new p-value cutoff. Validation rejects malformed coordinates, unknown chromosomes, missing required fields, invalid/non-integer/negative counts, zero-read measurements, non-binary labels, and p-values outside [0,1]. Exact repeated records are deduplicated; conflicting duplicate identities fail. The four inspected files record zero missing values. This is table-level QC: we did not independently reprocess FASTQs, assess antibody specificity, FRiP/TSS enrichment, phasing error or residual allele-mapping bias.

**Representative cCRE:** chr1:817080–817403, `EH38D2115333_PLS,CTCF-bound`, haplotype counts 16/16, donor ENC-001, thoracic aorta, H3K27ac, supplied p=1 and AS=0. **Representative SNV:** chr1:17385–17386, ref G and haplotypes A/G, RNA-seq, counts A/C/G/T=7/0/7/0, AS=0. Examples demonstrate schema, not a locus selected for model success.

### A.2 Assay-specific SNV data

| Assay/source | Measurements | Unique loci | AS+ measurements | AS− measurements | Prevalence | Donors | Tissues |
|---|---|---|---|---|---|---|---|
| ctcf | 2391015 | 596650 | 83911 | 2307104 | 3.5094% | 4 | 30 |
| h3k27ac | 3168525 | 713418 | 74833 | 3093692 | 2.3618% | 4 | 28 |
| rna | 1640580 | 466867 | 23546 | 1617034 | 1.4352% | 4 | 29 |
| atac | 3265155 | 1498771 | 133227 | 3131928 | 4.0803% | 4 | 24 |
| h3k4me3 | 1659748 | 265099 | 74771 | 1584977 | 4.5050% | 4 | 29 |
| h3k27me3 | 1291316 | 702509 | 25667 | 1265649 | 1.9877% | 4 | 24 |

All six prepared SNV datasets record zero identical-duplicate exclusions. CTCF and H3K27ac come from the full accessible source; the initially supplied high-confidence file contains RNA-seq only. The full source contains 22,310,439 measurements across 12 assays and was parsed in chunks, once into compact caches. RNA here uses the high-confidence subset (1,640,580 measurements), not all 2,531,272 RNA measurements in the default file. Source selection therefore differs by endpoint.

### A.3 cCREs, tissue selection and registry mismatch

P0 retains 250,722 loci: 28,092 positive and 222,630 negative (11.2044%), from 5,330,335 informative measurements. No source measurements were excluded. The cCRE table spans four donors, 30 tissues and 11 assays. “Any significant call” gives loci with more measurements more opportunities to become positive; this is why exposure matching is essential.

The active and repressed annotations span 28 tissues. Initial SCREEN v4 joins matched only 1,783,533/5,646,598 active rows and 1,921,649/4,620,982 repressed rows. This outcome-dependent loss was not accepted. Registry ENCFF924IMH (ENCODE V2) matched all supplied IDs. We required V2 dELS and EN-TEx distal status; promoter/proximal elements were not used to create an easy promoter-versus-enhancer task.

| Selected tissue | Loci | Active | Repressed | Active prevalence | Conflicting locus/tissue states excluded |
|---|---|---|---|---|---|
| Peyers_patch | 234737 | 93921 | 140816 | 0.4001 | 26773 |
| body_of_pancreas | 227552 | 100064 | 127488 | 0.4397 | 25486 |
| gastroesophageal_sphincter | 205788 | 94745 | 111043 | 0.4604 | 14565 |
| thyroid_gland | 260898 | 115611 | 145287 | 0.4431 | 32569 |
| tibial_nerve | 233508 | 101438 | 132070 | 0.4344 | 17605 |

Tissues were ranked by the smaller class count, then total loci and tissue name, with at least 1,000 examples per class; model performance never entered selection. Separate classifiers were fitted per tissue. The macro result gives equal weight to these five fixed tissues within each fold/run, then bootstraps folds/runs; it does not resample tissues or claim all-tissue generalization. Conflicts were excluded because a single static binary state could not be assigned faithfully after collapsing the supplied annotations.

### A.4 Mapping and feature coverage

All tasks use the same checksum-pinned HPRC R2 SV graph as the manuscript. The existing half-open interval mapper was reused. P0 maps 250,722/250,722 loci: 247,382 overlap one segment and 3,340 overlap multiple segments (1.3322%). The default mean pools segment features; length-weighted pooling is implemented, but a completed length-weighted sensitivity is not established by the reviewed outputs. SNVs use only the containing segment; no neighbourhood pooling extension was added. All six SNV datasets map 100% and have no multi-segment SNVs. The original P1 completion record reports 100% mapping; its per-run feature-coverage receipts are available locally, but the separate full mapping receipt was not re-exported for this review.

Joint C/K/S/T coverage is 100% across the 180 P0/sensitivity/CTCF/H3K27ac configurations. Across 150 P1 configurations, common coverage is at least 99.9982869966%; two to four examples per run are excluded under the existing topology-extraction rules. They are removed from every feature arm, not just T. Do not sum these repeated fold/context exclusions into a count of unique loci. RNA/extension preparation, smoke and completed summary/follow-up audits are available; this review does not have a uniform 120-run raw feature-audit export for those later fits, so their every-run feature coverage is not independently re-certified here.

### A.5 Resource identities

Graph segment SHA256: `e0d832a820969403797662f9af267440599898f068ba9df4069b114d669a3347`. NT: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, revision `81b29e5786726d891dbf929404ef20adca5b36f1`. Registry SHA256: `16fe76cbbc1f24e38a5476ce44fa4b81a9612f61a23517860d6e61f5619615ec`. No release or checkpoint substitution is allowed.

| Source | SHA256 |
|---|---|
| cCREs_default_AS.tsv | 679d916cf78008b0f6c279a30ef7131d8e6a5645c7bb5e6051a7906c0500ba28 |
| hetSNVs_high-confidence_AS.tsv | 01b6b7d27446010e214a8cdf1569312cb253c6bedc064a942554b9e28f0927e1 |
| active.combined_set.txt.zip | 676e62e3e2fbeae71d1ff39d09121a5b574ccd4079b9609c27d1dcc8afde9637 |
| repressed.combined_set.txt.zip | 0096f4c5d0fce555bf67d3e8dddf52413e471be4a38ab86c8ede937dbdcc3770 |
| hetSNVs_default_AS.tsv | e59a83a1595cbe12ba56714af79c297ac9f31b593e13ee44966c732d30d4daa8 |

## Appendix B. Definitions, completeness and evaluation

EN-TEx combines personal genomes and functional assays across tissues from four individuals. Its published resource contains 1,635 datasets; our experiments use released, processed accessible/testable AS calls and active/repressed cCRE annotations, not all raw experiments. Source: Rozowsky et al., Cell 186, 1493–1511.e40 (2023), and the released EN-TEx annotations.

Primary objective: measure the incremental AP of frozen topology T beyond C+S on identical examples: `ΔT = AP(C+S+T) − AP(C+S)`. The reciprocal contrast is `ΔS = AP(C+S+T) − AP(C+T)`. A positive topology increment is neither proof of causal biological information nor proof that pretraining was necessary.

### Experimental units and labels

| Task | Unit and positive | Negative / exclusions |
|---|---|---|
| P0 | One cCRE locus; any supplied significant AS call among informative measurements | Measured/testable with no significant call; absent/unmeasured elements never become negative |
| P0b | Existing P0 predictions within predefined graph-complexity strata | No re-binning or selection using performance |
| P0 sensitivities | Exact exposure matching, H3K27ac-only, CTCF-only | Definitions fixed before P0 biological fitting; matching after common feature coverage |
| P1 | A distal enhancer-like cCRE in one tissue, explicitly active | Explicitly repressed; require V2 dELS and supplied distal state; exclude conflicting locus/tissue labels |
| P2 / extensions | A measured heterozygous SNV/experiment, supplied AS significance=1 | Accessible/informative SNV measurement with significance=0; no absence-based negatives |

SNV features and predictions are constant for every occurrence of a locus within a run. Training aggregates identical locus/label examples with counts while retaining original measurement class weights and scaler occurrence weights. A locus may have both AS and non-AS measurements across experiments. Thus the task measures propensity for imbalance within the sampled measurements; it does not predict the favoured allele, its direction of effect, a causal variant, or a donor/tissue-specific response.

### Completed programme

| Programme | Configurations | Status |
|---|---|---|
| P0 + 3 sensitivities | 120 | Complete |
| P1, 5 tissues | 150 | Complete |
| CTCF/H3K27ac SNVs | 60 | Complete |
| RNA ASE | 30 | Complete |
| ATAC/H3K4me3/H3K27me3 SNVs | 90 | Complete |
| Total | 450 | 3,150 feature-specific classifier metric records |

Here “configuration” means one task × fold × run identifier × graph context, with seven separately fitted feature-set probes. It is not 450 donors or biological replicates. We checked all 3,150 per-run metric rows, five folds, three run identifiers, two contexts, and all seven features; recomputed summary means and paired ΔT from the saved metrics agree within 1e-12. CIs are the saved hierarchical-bootstrap estimates, not newly bootstrapped predictions.

Completed follow-ups re-evaluate 150 existing prediction configurations (60 CTCF/H3K27ac and 90 extension runs); they are not additional training runs. Full prediction arrays and large graph/cache files remain on the lab server. The present review checks locally saved result tables, QC and audit receipts, and does not claim a new server rerun.

### Features, splits and uncertainty

- C: log1p segment reference offset, log1p segment length and orientation; pooled using the same rules as other segment features. These are not freshly redefined locus coordinates.
- K: normalized mono/di/tri-nucleotide composition from the manuscript procedure.
- S: frozen NT final hidden-state mean over non-special/non-padding tokens. Long segments use balanced terminal sequence, at most 6,000 raw bases, with tokenizer truncation at 1,000 tokens.
- T: frozen original PangenomeFM representations under strict/one-hop context. The EN-TEx results below do not evaluate the newer sequence-conditioned E model.
- Seven feature sets: C, K, S, T, C+S, C+T, C+S+T. Only the existing standardized logistic probes receive biological labels; validation alone selects calibration/decision thresholds.

| Fold | Test chromosomes | Validation block |
|---|---|---|
| a | 1, 6, 11, 16, 21 | b |
| b | 2, 7, 12, 17, 22 | c |
| c | 3, 8, 13, 18, X | d |
| d | 4, 9, 14, 19, Y | e |
| e | 5, 10, 15, 20 | a |

All remaining chromosomes train the probe. Locus repeats across donors/tissues/assays stay in the same chromosome partition. Run identifiers are 42, 314159 and 20260806. Historical encoder initialization and DropEdge were not fully seeded: these identifiers should not be described as complete RNG reproducibility. Chromosome separation is not unseen-donor validation, since each donor contributes observations on multiple chromosomes.

Primary metric: AP. Secondary metrics: AUROC, balanced accuracy, F1, precision, recall and normalized AP `(AP − prevalence)/(1 − prevalence)`. Random-ranking AP equals prevalence. Normalized AP adjusts the baseline but is not fully prevalence invariant. The summaries resample five folds and then three runs within folds for 10,000 hierarchical draws, pairing feature contrasts before resampling. Repeated training sets and only five chromosome blocks limit uncertainty estimates. The intervals are pointwise; an interval above zero is not a family-wise significance claim.

## Appendix C. Exact primary and sensitivity results

Each row averages 15 fold/run pairs. The prevalence column is the mean held-out-fold prevalence and can differ from the global source prevalence above. All null and negative results are shown.

| Task | Context | Test prevalence | AP C+S | AP C+S+T | ΔT [95% CI] | AUROC C+S | AUROC C+S+T |
|---|---|---|---|---|---|---|---|
| P0 AS-prone cCRE | strict | 0.111392 | 0.151416 | 0.151625 | +0.000208 [-0.000317, +0.000746] | 0.584674 | 0.584673 |
| P0 AS-prone cCRE | 1hop | 0.111392 | 0.151416 | 0.150762 | -0.000654 [-0.002092, +0.000466] | 0.584674 | 0.583952 |
| P0 exposure matched | strict | 0.500000 | 0.533617 | 0.535099 | +0.001482 [+0.000098, +0.002925] | 0.529444 | 0.530224 |
| P0 exposure matched | 1hop | 0.500000 | 0.533617 | 0.535618 | +0.002002 [+0.000844, +0.003278] | 0.529444 | 0.531259 |
| P0 H3K27ac only | strict | 0.045663 | 0.062326 | 0.062910 | +0.000584 [+0.000084, +0.001065] | 0.560453 | 0.563766 |
| P0 H3K27ac only | 1hop | 0.045663 | 0.062326 | 0.062938 | +0.000612 [-0.000533, +0.001689] | 0.560453 | 0.563272 |
| P0 CTCF only | strict | 0.065020 | 0.078094 | 0.078354 | +0.000261 [-0.000370, +0.000952] | 0.534003 | 0.533580 |
| P0 CTCF only | 1hop | 0.065020 | 0.078094 | 0.078819 | +0.000725 [+0.000217, +0.001301] | 0.534003 | 0.534729 |
| P1 thyroid_gland | strict | 0.442483 | 0.571855 | 0.572962 | +0.001107 [-0.000376, +0.002996] | 0.649640 | 0.649764 |
| P1 thyroid_gland | 1hop | 0.442483 | 0.571855 | 0.573284 | +0.001430 [-0.000742, +0.003184] | 0.649640 | 0.650337 |
| P1 tibial_nerve | strict | 0.431937 | 0.584226 | 0.586990 | +0.002764 [+0.000573, +0.005121] | 0.670585 | 0.671973 |
| P1 tibial_nerve | 1hop | 0.431937 | 0.584226 | 0.585834 | +0.001608 [-0.002043, +0.004163] | 0.670585 | 0.671347 |
| P1 body_of_pancreas | strict | 0.438851 | 0.556333 | 0.557195 | +0.000862 [-0.000206, +0.002003] | 0.631421 | 0.631909 |
| P1 body_of_pancreas | 1hop | 0.438851 | 0.556333 | 0.556806 | +0.000474 [-0.000445, +0.001361] | 0.631421 | 0.632017 |
| P1 gastroesophageal_sphincter | strict | 0.459414 | 0.571966 | 0.572458 | +0.000492 [-0.001782, +0.002691] | 0.632661 | 0.632985 |
| P1 gastroesophageal_sphincter | 1hop | 0.459414 | 0.571966 | 0.572568 | +0.000603 [-0.004254, +0.004279] | 0.632661 | 0.632560 |
| P1 Peyers_patch | strict | 0.397241 | 0.565275 | 0.564497 | -0.000778 [-0.001683, +0.000126] | 0.689070 | 0.688550 |
| P1 Peyers_patch | 1hop | 0.397241 | 0.565275 | 0.565183 | -0.000092 [-0.001569, +0.001113] | 0.689070 | 0.689334 |
| P2 CTCF | strict | 0.034568 | 0.061559 | 0.063033 | +0.001473 [+0.000096, +0.003549] | 0.544256 | 0.544928 |
| P2 CTCF | 1hop | 0.034568 | 0.061559 | 0.063805 | +0.002245 [+0.000958, +0.003482] | 0.544256 | 0.548724 |
| P2 H3K27ac | strict | 0.023245 | 0.049680 | 0.050373 | +0.000693 [-0.000567, +0.001920] | 0.580121 | 0.584331 |
| P2 H3K27ac | 1hop | 0.023245 | 0.049680 | 0.053018 | +0.003338 [+0.000120, +0.007962] | 0.580121 | 0.594022 |
| RNA ASE | strict | 0.013921 | 0.025530 | 0.027099 | +0.001568 [-0.000075, +0.004401] | 0.607324 | 0.612091 |
| RNA ASE | 1hop | 0.013921 | 0.025530 | 0.026996 | +0.001465 [-0.000218, +0.004303] | 0.607324 | 0.607461 |
| atac | strict | 0.040577 | 0.047280 | 0.046971 | -0.000309 [-0.000714, +0.000020] | 0.528504 | 0.527620 |
| atac | 1hop | 0.040577 | 0.047280 | 0.047599 | +0.000319 [-0.000672, +0.001780] | 0.528504 | 0.529634 |
| h3k4me3 | strict | 0.044848 | 0.080250 | 0.084671 | +0.004420 [+0.000116, +0.010614] | 0.561953 | 0.567680 |
| h3k4me3 | 1hop | 0.044848 | 0.080250 | 0.089251 | +0.009000 [+0.000618, +0.018619] | 0.561953 | 0.586314 |
| h3k27me3 | strict | 0.019837 | 0.091497 | 0.094883 | +0.003386 [-0.000456, +0.009559] | 0.682334 | 0.683667 |
| h3k27me3 | 1hop | 0.019837 | 0.091497 | 0.098773 | +0.007276 [-0.001910, +0.021325] | 0.682334 | 0.682379 |


### P1 equal-tissue macro

| Context | AP C+S | AP C+S+T | ΔT [95% CI] |
|---|---|---|---|
| strict | 0.569931 | 0.570820 | +0.000889 [-0.000389, +0.002326] |
| 1hop | 0.569931 | 0.570735 | +0.000805 [-0.001531, +0.002467] |

### Interpretation of the complete task panel

- **P0:** global gain is inconclusive; one-hop has a negative mean. The positive exposure-matched contrast concerns a different 50%-positive population. C alone performs better than the largest feature set in that matched population, so “positive incremental gain” is not “best classifier”.
- **P1:** macro gain is inconclusive. Individual tissue estimates are descriptive and correlated; a positive tibial-nerve estimate does not establish a general enhancer-state benefit.
- **CTCF / H3K27ac:** small assay/context-dependent increments; investigate weighting before claiming robust locus-level reuse.
- **RNA:** the positive fold-A smoke did not produce a positive multi-fold interval. The full 30-run result supersedes it.
- **H3K4me3:** positive pointwise intervals in both contexts, but the effect is smaller under equal-locus weighting and does not survive the existing six-contrast multiplicity analysis.
- **ATAC / H3K27me3:** global intervals cross zero; no consistent improvement is established.

## Appendix D. Measurement exposure, donor/tissue macros and multiplicity

Three P0 sensitivities were fixed before biological fitting: exact 1:1 matching within chromosome × number-of-informative-measurements strata, H3K27ac-only calls and CTCF-only calls. Matching is without replacement with seed 20260806 and is applied after common feature coverage; every arm uses the same selected loci. It controls the measured exposure count, not all donor/tissue/assay ascertainment.

CTCF/H3K27ac follow-ups were specified after inspecting primary results and are exploratory. The additional three-assay follow-up protocol was fixed before those fits. Equal-locus weighting assigns each measurement weight `1 / measurements at its locus` during evaluation only. It does not refit a classifier with equal-locus training weights. Donor/tissue subsets require at least 200 measurements, 10 positives and 10 negatives in every held-out fold; all eligible groups are retained. Macros weight eligible groups equally.

| Assay | Evaluation weighting/group macro | Context | ΔAP [95% CI] |
|---|---|---|---|
| ctcf | donor_macro | 1hop | +0.002172 [+0.000854, +0.003423] |
| ctcf | donor_macro | strict | +0.001038 [-0.000041, +0.002338] |
| ctcf | equal_locus_weight | 1hop | +0.000493 [-0.000343, +0.001323] |
| ctcf | equal_locus_weight | strict | +0.000183 [-0.000542, +0.000955] |
| ctcf | measurement_weight | 1hop | +0.002245 [+0.000932, +0.003455] |
| ctcf | measurement_weight | strict | +0.001473 [+0.000133, +0.003519] |
| ctcf | tissue_macro | 1hop | +0.003299 [+0.001279, +0.005534] |
| ctcf | tissue_macro | strict | +0.001915 [+0.000417, +0.003968] |
| h3k27ac | donor_macro | 1hop | +0.003696 [+0.000425, +0.008464] |
| h3k27ac | donor_macro | strict | +0.000766 [-0.000656, +0.002163] |
| h3k27ac | equal_locus_weight | 1hop | +0.002206 [+0.000653, +0.004192] |
| h3k27ac | equal_locus_weight | strict | +0.001990 [+0.001105, +0.002731] |
| h3k27ac | measurement_weight | 1hop | +0.003338 [+0.000118, +0.008003] |
| h3k27ac | measurement_weight | strict | +0.000693 [-0.000589, +0.001975] |
| h3k27ac | tissue_macro | 1hop | +0.003563 [-0.000183, +0.008526] |
| h3k27ac | tissue_macro | strict | +0.000149 [-0.001244, +0.001542] |
| atac | donor_macro | 1hop | +0.000267 [-0.000884, +0.001734] |
| atac | donor_macro | strict | -0.000336 [-0.000679, -0.000054] |
| atac | equal_locus_weight | 1hop | -0.000019 [-0.000289, +0.000331] |
| atac | equal_locus_weight | strict | -0.000010 [-0.000121, +0.000112] |
| atac | measurement_weight | 1hop | +0.000319 [-0.000661, +0.001728] |
| atac | measurement_weight | strict | -0.000309 [-0.000709, +0.000018] |
| atac | tissue_macro | 1hop | +0.000084 [-0.002036, +0.001995] |
| atac | tissue_macro | strict | -0.000390 [-0.001349, +0.000449] |
| h3k4me3 | donor_macro | 1hop | +0.009195 [+0.000716, +0.018964] |
| h3k4me3 | donor_macro | strict | +0.004613 [-0.000090, +0.011221] |
| h3k4me3 | equal_locus_weight | 1hop | +0.001149 [-0.000538, +0.002810] |
| h3k4me3 | equal_locus_weight | strict | +0.001068 [-0.000049, +0.002431] |
| h3k4me3 | measurement_weight | 1hop | +0.009000 [+0.000711, +0.018416] |
| h3k4me3 | measurement_weight | strict | +0.004420 [+0.000112, +0.010603] |
| h3k4me3 | tissue_macro | 1hop | +0.009722 [+0.000709, +0.019526] |
| h3k4me3 | tissue_macro | strict | +0.004797 [+0.000084, +0.011501] |
| h3k27me3 | donor_macro | 1hop | +0.005474 [-0.003312, +0.018503] |
| h3k27me3 | donor_macro | strict | +0.002218 [-0.004661, +0.008754] |
| h3k27me3 | equal_locus_weight | 1hop | +0.001161 [-0.000731, +0.004024] |
| h3k27me3 | equal_locus_weight | strict | +0.000856 [-0.000295, +0.002251] |
| h3k27me3 | measurement_weight | 1hop | +0.007276 [-0.001867, +0.021231] |
| h3k27me3 | measurement_weight | strict | +0.003386 [-0.000449, +0.009522] |
| h3k27me3 | tissue_macro | 1hop | +0.004527 [-0.005772, +0.017457] |
| h3k27me3 | tissue_macro | strict | +0.001113 [-0.004547, +0.007043] |

Equal-locus weighting makes CTCF inconclusive and preserves a small H3K27ac signal in both contexts. The stronger measurement-weighted H3K4me3 estimates attenuate markedly under equal-locus weighting. These patterns are consistent with a contribution from measurement multiplicity, but the weighting comparison does not isolate a causal ascertainment mechanism. Donor and tissue strata are not independently held-out donors/tissues.

### Existing multiplicity sensitivity for the three-assay extension

| Assay | Weighting | Context | ΔAP | Exact fold sign-flip P | BH q |
|---|---|---|---|---|---|
| atac | equal_locus_weight | 1hop | -0.000019 | 1.0000 | 1.0000 |
| atac | equal_locus_weight | strict | -0.000010 | 0.8750 | 1.0000 |
| atac | measurement_weight | 1hop | 0.000319 | 1.0000 | 1.0000 |
| atac | measurement_weight | strict | -0.000309 | 0.1875 | 0.3750 |
| h3k27me3 | equal_locus_weight | 1hop | 0.001161 | 0.7500 | 1.0000 |
| h3k27me3 | equal_locus_weight | strict | 0.000856 | 0.3750 | 0.7500 |
| h3k27me3 | measurement_weight | 1hop | 0.007276 | 0.4375 | 0.5250 |
| h3k27me3 | measurement_weight | strict | 0.003386 | 0.2500 | 0.3750 |
| h3k4me3 | equal_locus_weight | 1hop | 0.001149 | 0.3125 | 0.7500 |
| h3k4me3 | equal_locus_weight | strict | 0.001068 | 0.2500 | 0.7500 |
| h3k4me3 | measurement_weight | 1hop | 0.009000 | 0.1875 | 0.3750 |
| h3k4me3 | measurement_weight | strict | 0.004420 | 0.1875 | 0.3750 |

BH adjustment is across the six assay/context contrasts separately for each metric and weighting scheme. None of these AP contrasts passes q<0.05. The minimum attainable two-sided sign-flip P with five folds is 0.0625. This is an exploratory sensitivity, not a correction covering every task, stratum and follow-up in this report.

## Appendix E. P0 graph-complexity results

Use the existing label-free native complexity v2 categories and locus-start regional assignment, fixed independently of EN-TEx performance. AS prevalence increasing with complexity does not show that adding T helps. The following P0 comparison uses the same examples within each stratum; normalized AP and AUROC appear alongside AP.

| Context | Stratum | Mean test n | Mean prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 1hop | high | 21414.8 | 0.130428 | 0.172560 | 0.172162 | -0.000398 [-0.002298, +0.001443] | -0.000501 [-0.003058, +0.002180] | -0.000430 [-0.002589, +0.001689] |
| 1hop | low | 12477.6 | 0.090691 | 0.117755 | 0.116619 | -0.001136 [-0.001836, -0.000457] | -0.001553 [-0.004014, +0.000475] | -0.001249 [-0.002014, -0.000504] |
| 1hop | medium | 16252.0 | 0.101949 | 0.129840 | 0.128429 | -0.001412 [-0.002348, -0.000620] | -0.001470 [-0.002552, -0.000287] | -0.001569 [-0.002604, -0.000691] |
| strict | high | 21414.8 | 0.130428 | 0.172560 | 0.173060 | +0.000500 [-0.000659, +0.001569] | +0.000806 [-0.001104, +0.002586] | +0.000579 [-0.000753, +0.001813] |
| strict | low | 12477.6 | 0.090691 | 0.117755 | 0.117306 | -0.000449 [-0.001383, +0.000453] | -0.000879 [-0.003325, +0.001509] | -0.000499 [-0.001528, +0.000497] |
| strict | medium | 16252.0 | 0.101949 | 0.129840 | 0.129344 | -0.000496 [-0.001436, +0.000664] | -0.001442 [-0.004379, +0.001293] | -0.000550 [-0.001594, +0.000742] |

P0 does not establish a positive high-complexity topology gain. In one-hop context, low- and medium-complexity intervals are negative. Prevalence-normalization does not turn this into evidence of beneficial topology. All task-specific complexity contrasts appear in Appendix I below; no bin was chosen for a favorable outcome.

## Appendix F. All seven feature combinations

AP is mean [pointwise 95% CI]; other metrics below are means. The exact sample sizes are in Appendix H. AP uncertainty is shown for every feature combination; secondary metrics here are descriptive means.

### P0 AS-prone cCRE

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.132805 [0.121161, 0.143423] | 0.542210 | 0.517221 | 0.202349 | 0.115894 | 0.803939 |
| strict | K | 0.145363 [0.137591, 0.151716] | 0.576773 | 0.554796 | 0.217581 | 0.134196 | 0.583222 |
| strict | S | 0.150806 [0.140999, 0.158879] | 0.584222 | 0.557887 | 0.219702 | 0.136140 | 0.587866 |
| strict | T | 0.134611 [0.122442, 0.145136] | 0.547338 | 0.520938 | 0.201609 | 0.118670 | 0.702771 |
| strict | C+S | 0.151416 [0.141568, 0.159597] | 0.584674 | 0.558075 | 0.219886 | 0.136090 | 0.587681 |
| strict | C+T | 0.138026 [0.125350, 0.149569] | 0.554082 | 0.530633 | 0.203914 | 0.124136 | 0.602089 |
| strict | C+S+T | 0.151625 [0.141883, 0.159925] | 0.584673 | 0.557740 | 0.219026 | 0.139190 | 0.541403 |
| 1hop | C | 0.132805 [0.121161, 0.143423] | 0.542210 | 0.517221 | 0.202349 | 0.115894 | 0.803939 |
| 1hop | K | 0.145363 [0.137591, 0.151716] | 0.576773 | 0.554796 | 0.217581 | 0.134196 | 0.583222 |
| 1hop | S | 0.150806 [0.140999, 0.158879] | 0.584222 | 0.557887 | 0.219702 | 0.136140 | 0.587866 |
| 1hop | T | 0.133529 [0.121161, 0.144457] | 0.547912 | 0.523394 | 0.202662 | 0.119398 | 0.696917 |
| 1hop | C+S | 0.151416 [0.141568, 0.159597] | 0.584674 | 0.558075 | 0.219886 | 0.136090 | 0.587681 |
| 1hop | C+T | 0.135678 [0.122743, 0.147053] | 0.553098 | 0.533353 | 0.205190 | 0.124081 | 0.614624 |
| 1hop | C+S+T | 0.150762 [0.140080, 0.159917] | 0.583952 | 0.557775 | 0.219052 | 0.139035 | 0.537209 |


### P0 exposure matched

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.537015 [0.534492, 0.539398] | 0.532326 | 0.500012 | 0.666580 | 0.500006 | 0.999587 |
| strict | K | 0.523155 [0.516640, 0.530122] | 0.519214 | 0.499779 | 0.666470 | 0.499890 | 0.999558 |
| strict | S | 0.529301 [0.524859, 0.533742] | 0.525322 | 0.499978 | 0.666498 | 0.499989 | 0.999283 |
| strict | T | 0.535487 [0.529062, 0.541452] | 0.528739 | 0.500000 | 0.666432 | 0.500000 | 0.998945 |
| strict | C+S | 0.533617 [0.529470, 0.537764] | 0.529444 | 0.499823 | 0.666123 | 0.499911 | 0.997919 |
| strict | C+T | 0.536912 [0.532708, 0.541190] | 0.530373 | 0.500023 | 0.666548 | 0.500012 | 0.999421 |
| strict | C+S+T | 0.535099 [0.530269, 0.540111] | 0.530224 | 0.499939 | 0.666364 | 0.499969 | 0.998763 |
| 1hop | C | 0.537015 [0.534492, 0.539398] | 0.532326 | 0.500012 | 0.666580 | 0.500006 | 0.999587 |
| 1hop | K | 0.523155 [0.516640, 0.530122] | 0.519214 | 0.499779 | 0.666470 | 0.499890 | 0.999558 |
| 1hop | S | 0.529301 [0.524859, 0.533742] | 0.525322 | 0.499978 | 0.666498 | 0.499989 | 0.999283 |
| 1hop | T | 0.536743 [0.532721, 0.541131] | 0.531420 | 0.500022 | 0.666593 | 0.500011 | 0.999623 |
| 1hop | C+S | 0.533617 [0.529470, 0.537764] | 0.529444 | 0.499823 | 0.666123 | 0.499911 | 0.997919 |
| 1hop | C+T | 0.537656 [0.534647, 0.540757] | 0.532272 | 0.500026 | 0.666475 | 0.500013 | 0.999088 |
| 1hop | C+S+T | 0.535618 [0.531954, 0.539195] | 0.531259 | 0.500015 | 0.666558 | 0.500008 | 0.999480 |


### P0 H3K27ac only

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.063906 [0.057548, 0.070265] | 0.559279 | 0.534597 | 0.099585 | 0.064154 | 0.265263 |
| strict | K | 0.061478 [0.057720, 0.065236] | 0.562459 | 0.541270 | 0.102994 | 0.062751 | 0.296487 |
| strict | S | 0.061933 [0.055358, 0.068509] | 0.558565 | 0.532043 | 0.096079 | 0.063376 | 0.266097 |
| strict | T | 0.064841 [0.056087, 0.073302] | 0.564913 | 0.542543 | 0.105711 | 0.069888 | 0.265229 |
| strict | C+S | 0.062326 [0.055843, 0.068784] | 0.560453 | 0.536402 | 0.100575 | 0.064646 | 0.251043 |
| strict | C+T | 0.066922 [0.059001, 0.074713] | 0.570133 | 0.547562 | 0.109421 | 0.069813 | 0.278189 |
| strict | C+S+T | 0.062910 [0.056327, 0.069527] | 0.563766 | 0.537661 | 0.101676 | 0.064859 | 0.265573 |
| 1hop | C | 0.063906 [0.057548, 0.070265] | 0.559279 | 0.534597 | 0.099585 | 0.064154 | 0.265263 |
| 1hop | K | 0.061478 [0.057720, 0.065236] | 0.562459 | 0.541270 | 0.102994 | 0.062751 | 0.296487 |
| 1hop | S | 0.061933 [0.055358, 0.068509] | 0.558565 | 0.532043 | 0.096079 | 0.063376 | 0.266097 |
| 1hop | T | 0.064067 [0.055269, 0.071998] | 0.564519 | 0.541376 | 0.105241 | 0.067843 | 0.258003 |
| 1hop | C+S | 0.062326 [0.055843, 0.068784] | 0.560453 | 0.536402 | 0.100575 | 0.064646 | 0.251043 |
| 1hop | C+T | 0.064758 [0.056524, 0.071754] | 0.568793 | 0.541641 | 0.106256 | 0.071740 | 0.236266 |
| 1hop | C+S+T | 0.062938 [0.055596, 0.070231] | 0.563272 | 0.539368 | 0.100492 | 0.062695 | 0.316054 |


### P0 CTCF only

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.076821 [0.073963, 0.080070] | 0.529971 | 0.515019 | 0.122354 | 0.068201 | 0.631543 |
| strict | K | 0.074346 [0.070660, 0.078135] | 0.522130 | 0.507330 | 0.121189 | 0.066657 | 0.736304 |
| strict | S | 0.077759 [0.073142, 0.083638] | 0.533162 | 0.521437 | 0.123812 | 0.070924 | 0.501507 |
| strict | T | 0.076703 [0.071694, 0.082069] | 0.526578 | 0.513236 | 0.120768 | 0.070077 | 0.580396 |
| strict | C+S | 0.078094 [0.073623, 0.083740] | 0.534003 | 0.522638 | 0.124634 | 0.071280 | 0.506036 |
| strict | C+T | 0.076918 [0.072713, 0.081463] | 0.528113 | 0.514084 | 0.120111 | 0.070454 | 0.560791 |
| strict | C+S+T | 0.078354 [0.073496, 0.084308] | 0.533580 | 0.519480 | 0.123686 | 0.070026 | 0.553050 |
| 1hop | C | 0.076821 [0.073963, 0.080070] | 0.529971 | 0.515019 | 0.122354 | 0.068201 | 0.631543 |
| 1hop | K | 0.074346 [0.070660, 0.078135] | 0.522130 | 0.507330 | 0.121189 | 0.066657 | 0.736304 |
| 1hop | S | 0.077759 [0.073142, 0.083638] | 0.533162 | 0.521437 | 0.123812 | 0.070924 | 0.501507 |
| 1hop | T | 0.077428 [0.072312, 0.083034] | 0.529717 | 0.513493 | 0.122066 | 0.069898 | 0.619457 |
| 1hop | C+S | 0.078094 [0.073623, 0.083740] | 0.534003 | 0.522638 | 0.124634 | 0.071280 | 0.506036 |
| 1hop | C+T | 0.077175 [0.072192, 0.082272] | 0.531089 | 0.515549 | 0.122437 | 0.070034 | 0.621145 |
| 1hop | C+S+T | 0.078819 [0.074212, 0.084356] | 0.534729 | 0.521624 | 0.124458 | 0.070415 | 0.554197 |


### P1 thyroid_gland

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.442412 [0.423950, 0.459112] | 0.501936 | 0.500167 | 0.613243 | 0.442566 | 0.999738 |
| strict | K | 0.543577 [0.521735, 0.559612] | 0.621599 | 0.550521 | 0.620922 | 0.471670 | 0.909883 |
| strict | S | 0.570836 [0.550378, 0.585200] | 0.648358 | 0.570896 | 0.628613 | 0.485814 | 0.890411 |
| strict | T | 0.456702 [0.431863, 0.472394] | 0.520378 | 0.500104 | 0.613145 | 0.442535 | 0.999366 |
| strict | C+S | 0.571855 [0.550010, 0.585959] | 0.649640 | 0.572257 | 0.629100 | 0.486703 | 0.889567 |
| strict | C+T | 0.458307 [0.434304, 0.475933] | 0.522015 | 0.500084 | 0.613035 | 0.442525 | 0.998766 |
| strict | C+S+T | 0.572962 [0.550913, 0.586731] | 0.649764 | 0.574532 | 0.628006 | 0.489006 | 0.878315 |
| 1hop | C | 0.442412 [0.423950, 0.459112] | 0.501936 | 0.500167 | 0.613243 | 0.442566 | 0.999738 |
| 1hop | K | 0.543577 [0.521735, 0.559612] | 0.621599 | 0.550521 | 0.620922 | 0.471670 | 0.909883 |
| 1hop | S | 0.570836 [0.550378, 0.585200] | 0.648358 | 0.570896 | 0.628613 | 0.485814 | 0.890411 |
| 1hop | T | 0.464041 [0.432211, 0.485126] | 0.528878 | 0.500133 | 0.613014 | 0.442549 | 0.998596 |
| 1hop | C+S | 0.571855 [0.550010, 0.585959] | 0.649640 | 0.572257 | 0.629100 | 0.486703 | 0.889567 |
| 1hop | C+T | 0.463634 [0.433295, 0.484176] | 0.528698 | 0.500315 | 0.612969 | 0.442639 | 0.997885 |
| 1hop | C+S+T | 0.573284 [0.549414, 0.588606] | 0.650337 | 0.569822 | 0.628279 | 0.484996 | 0.892148 |


### P1 tibial_nerve

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.429523 [0.407883, 0.448786] | 0.497600 | 0.500068 | 0.602676 | 0.431970 | 0.998974 |
| strict | K | 0.546452 [0.515823, 0.571282] | 0.632742 | 0.555557 | 0.612872 | 0.465094 | 0.899042 |
| strict | S | 0.582020 [0.556721, 0.598919] | 0.667658 | 0.584621 | 0.621839 | 0.488742 | 0.858554 |
| strict | T | 0.448809 [0.422538, 0.465985] | 0.525314 | 0.500513 | 0.602310 | 0.432191 | 0.995554 |
| strict | C+S | 0.584226 [0.557031, 0.601323] | 0.670585 | 0.589319 | 0.626143 | 0.491253 | 0.865968 |
| strict | C+T | 0.449720 [0.423386, 0.470138] | 0.524896 | 0.500078 | 0.602363 | 0.431977 | 0.996916 |
| strict | C+S+T | 0.586990 [0.560049, 0.602962] | 0.671973 | 0.589344 | 0.626063 | 0.490899 | 0.867018 |
| 1hop | C | 0.429523 [0.407883, 0.448786] | 0.497600 | 0.500068 | 0.602676 | 0.431970 | 0.998974 |
| 1hop | K | 0.546452 [0.515823, 0.571282] | 0.632742 | 0.555557 | 0.612872 | 0.465094 | 0.899042 |
| 1hop | S | 0.582020 [0.556721, 0.598919] | 0.667658 | 0.584621 | 0.621839 | 0.488742 | 0.858554 |
| 1hop | T | 0.451550 [0.412924, 0.478016] | 0.524973 | 0.500709 | 0.601978 | 0.432291 | 0.992605 |
| 1hop | C+S | 0.584226 [0.557031, 0.601323] | 0.670585 | 0.589319 | 0.626143 | 0.491253 | 0.865968 |
| 1hop | C+T | 0.448693 [0.411954, 0.476195] | 0.522120 | 0.500330 | 0.602280 | 0.432100 | 0.995712 |
| 1hop | C+S+T | 0.585834 [0.555641, 0.604041] | 0.671347 | 0.590910 | 0.625631 | 0.492935 | 0.859010 |


### P1 body_of_pancreas

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.444842 [0.425747, 0.463938] | 0.511625 | 0.500026 | 0.609824 | 0.438864 | 0.999808 |
| strict | K | 0.538959 [0.513938, 0.560297] | 0.612803 | 0.528028 | 0.612447 | 0.454100 | 0.940705 |
| strict | S | 0.555421 [0.531870, 0.579314] | 0.629889 | 0.554903 | 0.615026 | 0.472558 | 0.883212 |
| strict | T | 0.458543 [0.438158, 0.478893] | 0.526120 | 0.500052 | 0.609761 | 0.438877 | 0.999376 |
| strict | C+S | 0.556333 [0.532888, 0.580507] | 0.631421 | 0.553650 | 0.616713 | 0.470773 | 0.894425 |
| strict | C+T | 0.458797 [0.437225, 0.481287] | 0.526776 | 0.500117 | 0.609776 | 0.438909 | 0.999306 |
| strict | C+S+T | 0.557195 [0.534191, 0.581232] | 0.631909 | 0.552409 | 0.616672 | 0.469828 | 0.898433 |
| 1hop | C | 0.444842 [0.425747, 0.463938] | 0.511625 | 0.500026 | 0.609824 | 0.438864 | 0.999808 |
| 1hop | K | 0.538959 [0.513938, 0.560297] | 0.612803 | 0.528028 | 0.612447 | 0.454100 | 0.940705 |
| 1hop | S | 0.555421 [0.531870, 0.579314] | 0.629889 | 0.554903 | 0.615026 | 0.472558 | 0.883212 |
| 1hop | T | 0.459381 [0.435394, 0.481996] | 0.524966 | 0.500475 | 0.609365 | 0.439089 | 0.996091 |
| 1hop | C+S | 0.556333 [0.532888, 0.580507] | 0.631421 | 0.553650 | 0.616713 | 0.470773 | 0.894425 |
| 1hop | C+T | 0.458383 [0.434652, 0.481587] | 0.524582 | 0.500506 | 0.609382 | 0.439102 | 0.996006 |
| 1hop | C+S+T | 0.556806 [0.533070, 0.581034] | 0.632017 | 0.555872 | 0.616441 | 0.472995 | 0.887511 |


### P1 gastroesophageal_sphincter

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.464881 [0.454783, 0.475440] | 0.504475 | 0.500090 | 0.628932 | 0.459458 | 0.997095 |
| strict | K | 0.539470 [0.521622, 0.551653] | 0.596107 | 0.522562 | 0.630700 | 0.471579 | 0.952461 |
| strict | S | 0.569260 [0.553983, 0.579587] | 0.629345 | 0.547165 | 0.637367 | 0.486572 | 0.924283 |
| strict | T | 0.475445 [0.461305, 0.486460] | 0.521027 | 0.500292 | 0.629276 | 0.459559 | 0.998222 |
| strict | C+S | 0.571966 [0.555469, 0.582780] | 0.632661 | 0.557694 | 0.637465 | 0.494578 | 0.898231 |
| strict | C+T | 0.472179 [0.459781, 0.482554] | 0.516606 | 0.500214 | 0.628967 | 0.459520 | 0.996860 |
| strict | C+S+T | 0.572458 [0.557786, 0.582869] | 0.632985 | 0.554091 | 0.638222 | 0.491630 | 0.910914 |
| 1hop | C | 0.464881 [0.454783, 0.475440] | 0.504475 | 0.500090 | 0.628932 | 0.459458 | 0.997095 |
| 1hop | K | 0.539470 [0.521622, 0.551653] | 0.596107 | 0.522562 | 0.630700 | 0.471579 | 0.952461 |
| 1hop | S | 0.569260 [0.553983, 0.579587] | 0.629345 | 0.547165 | 0.637367 | 0.486572 | 0.924283 |
| 1hop | T | 0.474511 [0.447859, 0.490650] | 0.518650 | 0.500356 | 0.628783 | 0.459593 | 0.995414 |
| 1hop | C+S | 0.571966 [0.555469, 0.582780] | 0.632661 | 0.557694 | 0.637465 | 0.494578 | 0.898231 |
| 1hop | C+T | 0.470608 [0.445820, 0.486513] | 0.514200 | 0.500225 | 0.628966 | 0.459527 | 0.996712 |
| 1hop | C+S+T | 0.572568 [0.551816, 0.586058] | 0.632560 | 0.550973 | 0.636017 | 0.490238 | 0.908620 |


### P1 Peyers_patch

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.407671 [0.373928, 0.436341] | 0.514990 | 0.500687 | 0.566719 | 0.397581 | 0.991391 |
| strict | K | 0.534869 [0.496408, 0.564444] | 0.660735 | 0.591413 | 0.594947 | 0.458126 | 0.850787 |
| strict | S | 0.564537 [0.530532, 0.588008] | 0.688107 | 0.622946 | 0.610026 | 0.485780 | 0.820239 |
| strict | T | 0.419320 [0.385869, 0.445620] | 0.534508 | 0.500862 | 0.566533 | 0.397676 | 0.989335 |
| strict | C+S | 0.565275 [0.530581, 0.589554] | 0.689070 | 0.623645 | 0.611073 | 0.485621 | 0.824497 |
| strict | C+T | 0.425694 [0.389217, 0.457184] | 0.539412 | 0.501295 | 0.567627 | 0.397866 | 0.993707 |
| strict | C+S+T | 0.564497 [0.529363, 0.588931] | 0.688550 | 0.621497 | 0.610338 | 0.483776 | 0.827745 |
| 1hop | C | 0.407671 [0.373928, 0.436341] | 0.514990 | 0.500687 | 0.566719 | 0.397581 | 0.991391 |
| 1hop | K | 0.534869 [0.496408, 0.564444] | 0.660735 | 0.591413 | 0.594947 | 0.458126 | 0.850787 |
| 1hop | S | 0.564537 [0.530532, 0.588008] | 0.688107 | 0.622946 | 0.610026 | 0.485780 | 0.820239 |
| 1hop | T | 0.426675 [0.386236, 0.458584] | 0.541126 | 0.502280 | 0.564466 | 0.398415 | 0.970585 |
| 1hop | C+S | 0.565275 [0.530581, 0.589554] | 0.689070 | 0.623645 | 0.611073 | 0.485621 | 0.824497 |
| 1hop | C+T | 0.428746 [0.387317, 0.463571] | 0.542792 | 0.502925 | 0.564396 | 0.398747 | 0.967543 |
| 1hop | C+S+T | 0.565183 [0.528699, 0.590569] | 0.689334 | 0.619695 | 0.610057 | 0.482359 | 0.831589 |


### P2 CTCF

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.045564 [0.041385, 0.049744] | 0.525336 | 0.518503 | 0.074743 | 0.056478 | 0.181244 |
| strict | K | 0.061103 [0.046995, 0.077100] | 0.526016 | 0.533464 | 0.102181 | 0.127791 | 0.089112 |
| strict | S | 0.061809 [0.052703, 0.074871] | 0.544001 | 0.530627 | 0.082716 | 0.079638 | 0.127170 |
| strict | T | 0.061492 [0.051799, 0.076538] | 0.574290 | 0.542568 | 0.101481 | 0.098685 | 0.146960 |
| strict | C+S | 0.061559 [0.051995, 0.074838] | 0.544256 | 0.533824 | 0.095219 | 0.092807 | 0.142087 |
| strict | C+T | 0.060573 [0.051436, 0.074699] | 0.571396 | 0.541908 | 0.105117 | 0.102859 | 0.135194 |
| strict | C+S+T | 0.063033 [0.053059, 0.075599] | 0.544928 | 0.533997 | 0.091929 | 0.095486 | 0.136887 |
| 1hop | C | 0.045564 [0.041385, 0.049744] | 0.525336 | 0.518503 | 0.074743 | 0.056478 | 0.181244 |
| 1hop | K | 0.061103 [0.046995, 0.077100] | 0.526016 | 0.533464 | 0.102181 | 0.127791 | 0.089112 |
| 1hop | S | 0.061809 [0.052703, 0.074871] | 0.544001 | 0.530627 | 0.082716 | 0.079638 | 0.127170 |
| 1hop | T | 0.057397 [0.048307, 0.071447] | 0.565632 | 0.535701 | 0.087112 | 0.072700 | 0.166395 |
| 1hop | C+S | 0.061559 [0.051995, 0.074838] | 0.544256 | 0.533824 | 0.095219 | 0.092807 | 0.142087 |
| 1hop | C+T | 0.057763 [0.048198, 0.072198] | 0.564218 | 0.538150 | 0.088925 | 0.071358 | 0.183217 |
| 1hop | C+S+T | 0.063805 [0.054251, 0.077070] | 0.548724 | 0.535505 | 0.096217 | 0.098911 | 0.121754 |


### P2 H3K27ac

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.035798 [0.032219, 0.039969] | 0.542562 | 0.521356 | 0.057803 | 0.047802 | 0.139913 |
| strict | K | 0.042983 [0.032855, 0.054003] | 0.541780 | 0.532714 | 0.055056 | 0.060558 | 0.155067 |
| strict | S | 0.050027 [0.035695, 0.073720] | 0.580428 | 0.546223 | 0.064187 | 0.047928 | 0.241441 |
| strict | T | 0.042549 [0.036048, 0.048778] | 0.576128 | 0.539989 | 0.076769 | 0.061843 | 0.163326 |
| strict | C+S | 0.049680 [0.035702, 0.073066] | 0.580121 | 0.545792 | 0.086440 | 0.096914 | 0.211964 |
| strict | C+T | 0.041848 [0.035401, 0.048009] | 0.571604 | 0.541730 | 0.079706 | 0.069414 | 0.159028 |
| strict | C+S+T | 0.050373 [0.035782, 0.074393] | 0.584331 | 0.537460 | 0.072231 | 0.062395 | 0.173370 |
| 1hop | C | 0.035798 [0.032219, 0.039969] | 0.542562 | 0.521356 | 0.057803 | 0.047802 | 0.139913 |
| 1hop | K | 0.042983 [0.032855, 0.054003] | 0.541780 | 0.532714 | 0.055056 | 0.060558 | 0.155067 |
| 1hop | S | 0.050027 [0.035695, 0.073720] | 0.580428 | 0.546223 | 0.064187 | 0.047928 | 0.241441 |
| 1hop | T | 0.044561 [0.035403, 0.054847] | 0.580460 | 0.537274 | 0.070014 | 0.055962 | 0.160358 |
| 1hop | C+S | 0.049680 [0.035702, 0.073066] | 0.580121 | 0.545792 | 0.086440 | 0.096914 | 0.211964 |
| 1hop | C+T | 0.044803 [0.035381, 0.055373] | 0.582618 | 0.542044 | 0.075120 | 0.065479 | 0.157920 |
| 1hop | C+S+T | 0.053018 [0.037536, 0.075212] | 0.594022 | 0.547967 | 0.078928 | 0.070729 | 0.186088 |


### RNA ASE

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.024979 [0.017375, 0.035103] | 0.602551 | 0.530217 | 0.048201 | 0.035107 | 0.131914 |
| strict | K | 0.019028 [0.014479, 0.025077] | 0.558865 | 0.534202 | 0.032355 | 0.019112 | 0.234730 |
| strict | S | 0.023911 [0.017794, 0.030093] | 0.594222 | 0.531086 | 0.042527 | 0.032304 | 0.138125 |
| strict | T | 0.027295 [0.016513, 0.043835] | 0.602614 | 0.529146 | 0.051315 | 0.041398 | 0.099904 |
| strict | C+S | 0.025530 [0.018327, 0.033497] | 0.607324 | 0.526041 | 0.044282 | 0.044370 | 0.100106 |
| strict | C+T | 0.028255 [0.017008, 0.046173] | 0.605569 | 0.534627 | 0.055540 | 0.046248 | 0.120973 |
| strict | C+S+T | 0.027099 [0.018384, 0.037863] | 0.612091 | 0.530838 | 0.049265 | 0.039123 | 0.113306 |
| 1hop | C | 0.024979 [0.017375, 0.035103] | 0.602551 | 0.530217 | 0.048201 | 0.035107 | 0.131914 |
| 1hop | K | 0.019028 [0.014479, 0.025077] | 0.558865 | 0.534202 | 0.032355 | 0.019112 | 0.234730 |
| 1hop | S | 0.023911 [0.017794, 0.030093] | 0.594222 | 0.531086 | 0.042527 | 0.032304 | 0.138125 |
| 1hop | T | 0.026156 [0.017850, 0.039124] | 0.599844 | 0.537083 | 0.051409 | 0.034374 | 0.127341 |
| 1hop | C+S | 0.025530 [0.018327, 0.033497] | 0.607324 | 0.526041 | 0.044282 | 0.044370 | 0.100106 |
| 1hop | C+T | 0.026743 [0.017359, 0.041718] | 0.602098 | 0.534340 | 0.047209 | 0.031480 | 0.128317 |
| 1hop | C+S+T | 0.026996 [0.018707, 0.037301] | 0.607461 | 0.526433 | 0.046633 | 0.042509 | 0.098292 |


### atac

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.043793 [0.042432, 0.044566] | 0.511319 | 0.501286 | 0.076638 | 0.040638 | 0.746123 |
| strict | K | 0.046309 [0.043975, 0.048665] | 0.524944 | 0.514585 | 0.079135 | 0.047708 | 0.516484 |
| strict | S | 0.047449 [0.044566, 0.050458] | 0.528953 | 0.521722 | 0.080861 | 0.045838 | 0.407742 |
| strict | T | 0.044938 [0.043111, 0.047258] | 0.518794 | 0.511589 | 0.077876 | 0.042889 | 0.486079 |
| strict | C+S | 0.047280 [0.044413, 0.050357] | 0.528504 | 0.520074 | 0.080300 | 0.045349 | 0.445375 |
| strict | C+T | 0.044639 [0.042857, 0.046872] | 0.516891 | 0.508373 | 0.076259 | 0.042605 | 0.456838 |
| strict | C+S+T | 0.046971 [0.044144, 0.050215] | 0.527620 | 0.519521 | 0.079282 | 0.046227 | 0.374898 |
| 1hop | C | 0.043793 [0.042432, 0.044566] | 0.511319 | 0.501286 | 0.076638 | 0.040638 | 0.746123 |
| 1hop | K | 0.046309 [0.043975, 0.048665] | 0.524944 | 0.514585 | 0.079135 | 0.047708 | 0.516484 |
| 1hop | S | 0.047449 [0.044566, 0.050458] | 0.528953 | 0.521722 | 0.080861 | 0.045838 | 0.407742 |
| 1hop | T | 0.045496 [0.043197, 0.048363] | 0.524868 | 0.512531 | 0.075944 | 0.044606 | 0.373132 |
| 1hop | C+S | 0.047280 [0.044413, 0.050357] | 0.528504 | 0.520074 | 0.080300 | 0.045349 | 0.445375 |
| 1hop | C+T | 0.045471 [0.043162, 0.048389] | 0.524320 | 0.515389 | 0.077096 | 0.045583 | 0.365668 |
| 1hop | C+S+T | 0.047599 [0.044270, 0.051191] | 0.529634 | 0.518890 | 0.079128 | 0.047442 | 0.361078 |


### h3k4me3

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.072436 [0.061207, 0.085376] | 0.542681 | 0.524608 | 0.084465 | 0.071504 | 0.286798 |
| strict | K | 0.082727 [0.061685, 0.103769] | 0.551131 | 0.538887 | 0.103712 | 0.103814 | 0.347403 |
| strict | S | 0.080777 [0.058895, 0.104085] | 0.564003 | 0.536077 | 0.084019 | 0.073340 | 0.425912 |
| strict | T | 0.088726 [0.069303, 0.110765] | 0.592629 | 0.555272 | 0.129155 | 0.120864 | 0.210031 |
| strict | C+S | 0.080250 [0.058260, 0.103232] | 0.561953 | 0.529776 | 0.082214 | 0.073848 | 0.430694 |
| strict | C+T | 0.087555 [0.067215, 0.110721] | 0.587321 | 0.551391 | 0.122185 | 0.118006 | 0.197548 |
| strict | C+S+T | 0.084671 [0.060015, 0.109550] | 0.567680 | 0.547723 | 0.090989 | 0.076029 | 0.335721 |
| 1hop | C | 0.072436 [0.061207, 0.085376] | 0.542681 | 0.524608 | 0.084465 | 0.071504 | 0.286798 |
| 1hop | K | 0.082727 [0.061685, 0.103769] | 0.551131 | 0.538887 | 0.103712 | 0.103814 | 0.347403 |
| 1hop | S | 0.080777 [0.058895, 0.104085] | 0.564003 | 0.536077 | 0.084019 | 0.073340 | 0.425912 |
| 1hop | T | 0.092312 [0.073868, 0.110672] | 0.607028 | 0.551322 | 0.119404 | 0.113625 | 0.194677 |
| 1hop | C+S | 0.080250 [0.058260, 0.103232] | 0.561953 | 0.529776 | 0.082214 | 0.073848 | 0.430694 |
| 1hop | C+T | 0.092361 [0.072146, 0.112503] | 0.608667 | 0.554399 | 0.125026 | 0.118166 | 0.202198 |
| 1hop | C+S+T | 0.089251 [0.067337, 0.111149] | 0.586314 | 0.549084 | 0.091305 | 0.083057 | 0.320494 |


### h3k27me3

| Context | Features | AP [95% CI] | AUROC | Balanced accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| strict | C | 0.034126 [0.022460, 0.048292] | 0.533352 | 0.535405 | 0.068616 | 0.059583 | 0.168910 |
| strict | K | 0.087128 [0.062478, 0.110449] | 0.665292 | 0.584612 | 0.134589 | 0.123661 | 0.204085 |
| strict | S | 0.091857 [0.059858, 0.126318] | 0.682545 | 0.594978 | 0.181997 | 0.178115 | 0.211015 |
| strict | T | 0.073085 [0.051001, 0.094961] | 0.686102 | 0.572087 | 0.130257 | 0.122297 | 0.174589 |
| strict | C+S | 0.091497 [0.060137, 0.125212] | 0.682334 | 0.597311 | 0.184770 | 0.167007 | 0.216536 |
| strict | C+T | 0.076265 [0.053176, 0.098900] | 0.686314 | 0.576936 | 0.128167 | 0.120729 | 0.195755 |
| strict | C+S+T | 0.094883 [0.060278, 0.132577] | 0.683667 | 0.583302 | 0.157211 | 0.156568 | 0.188094 |
| 1hop | C | 0.034126 [0.022460, 0.048292] | 0.533352 | 0.535405 | 0.068616 | 0.059583 | 0.168910 |
| 1hop | K | 0.087128 [0.062478, 0.110449] | 0.665292 | 0.584612 | 0.134589 | 0.123661 | 0.204085 |
| 1hop | S | 0.091857 [0.059858, 0.126318] | 0.682545 | 0.594978 | 0.181997 | 0.178115 | 0.211015 |
| 1hop | T | 0.076399 [0.046105, 0.113908] | 0.671405 | 0.573967 | 0.131378 | 0.114546 | 0.176897 |
| 1hop | C+S | 0.091497 [0.060137, 0.125212] | 0.682334 | 0.597311 | 0.184770 | 0.167007 | 0.216536 |
| 1hop | C+T | 0.080940 [0.047879, 0.122034] | 0.675674 | 0.578262 | 0.130807 | 0.108897 | 0.188133 |
| 1hop | C+S+T | 0.098773 [0.059172, 0.144421] | 0.682379 | 0.592251 | 0.160443 | 0.153501 | 0.209140 |

## Appendix G. Reciprocal sequence contribution

| Task | Context | ΔS given C+T [95% CI] |
|---|---|---|
| P0 AS-prone cCRE | strict | +0.013598 [+0.010080, +0.017135] |
| P0 AS-prone cCRE | 1hop | +0.015084 [+0.012397, +0.017801] |
| P0 exposure matched | strict | -0.001813 [-0.005806, +0.002113] |
| P0 exposure matched | 1hop | -0.002037 [-0.004922, +0.000979] |
| P0 H3K27ac only | strict | -0.004012 [-0.007384, -0.000690] |
| P0 H3K27ac only | 1hop | -0.001821 [-0.005037, +0.001136] |
| P0 CTCF only | strict | +0.001436 [-0.000192, +0.003162] |
| P0 CTCF only | 1hop | +0.001644 [-0.000052, +0.003270] |
| P1 thyroid_gland | strict | +0.114654 [+0.106955, +0.121796] |
| P1 thyroid_gland | 1hop | +0.109651 [+0.103912, +0.116178] |
| P1 tibial_nerve | strict | +0.137270 [+0.129800, +0.145662] |
| P1 tibial_nerve | 1hop | +0.137141 [+0.127092, +0.146955] |
| P1 body_of_pancreas | strict | +0.098397 [+0.091220, +0.106454] |
| P1 body_of_pancreas | 1hop | +0.098423 [+0.092629, +0.103055] |
| P1 gastroesophageal_sphincter | strict | +0.100279 [+0.095431, +0.105483] |
| P1 gastroesophageal_sphincter | 1hop | +0.101960 [+0.094804, +0.108731] |
| P1 Peyers_patch | strict | +0.138803 [+0.129089, +0.153290] |
| P1 Peyers_patch | 1hop | +0.136437 [+0.125930, +0.147885] |
| P2 CTCF | strict | +0.002460 [-0.005298, +0.010048] |
| P2 CTCF | 1hop | +0.006042 [-0.001028, +0.013751] |
| P2 H3K27ac | strict | +0.008524 [-0.005499, +0.029043] |
| P2 H3K27ac | 1hop | +0.008215 [-0.003867, +0.026158] |
| RNA ASE | strict | -0.001156 [-0.008754, +0.004004] |
| RNA ASE | 1hop | +0.000253 [-0.005457, +0.004727] |
| atac | strict | +0.002332 [+0.000750, +0.003820] |
| atac | 1hop | +0.002128 [+0.000519, +0.003781] |
| h3k4me3 | strict | -0.002884 [-0.010719, +0.007825] |
| h3k4me3 | 1hop | -0.003110 [-0.017760, +0.010799] |
| h3k27me3 | strict | +0.018618 [-0.002522, +0.040524] |
| h3k27me3 | 1hop | +0.017833 [-0.010608, +0.047179] |

## Appendix H. Per-fold sizes and source distributions

Fold sizes below use C+S; the report verification checks equality with C+S+T. Counts are the units exported by each task (loci for P0/P1, original measurement rows for SNV tasks); do not sum across overlapping training folds.

### P0 AS-prone cCRE

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 124715 | 62566 | 63441 | 0.117842 |
| fold_b | 146551 | 41605 | 62566 | 0.113049 |
| fold_c | 168499 | 40618 | 41605 | 0.095662 |
| fold_d | 167612 | 42492 | 40618 | 0.121424 |
| fold_e | 144789 | 63441 | 42492 | 0.108985 |


### P0 exposure matched

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 24384 | 12848 | 13422 | 0.500000 |
| fold_b | 30602 | 7204 | 12848 | 0.500000 |
| fold_c | 34662 | 8788 | 7204 | 0.500000 |
| fold_d | 33474 | 8392 | 8788 | 0.500000 |
| fold_e | 28840 | 13422 | 8392 | 0.500000 |


### P0 H3K27ac only

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 71477 | 37144 | 38293 | 0.049043 |
| fold_b | 86802 | 22968 | 37144 | 0.048810 |
| fold_c | 100138 | 23808 | 22968 | 0.038706 |
| fold_d | 98405 | 24701 | 23808 | 0.050865 |
| fold_e | 83920 | 38293 | 24701 | 0.040889 |


### P0 CTCF only

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 58153 | 30176 | 31097 | 0.064476 |
| fold_b | 70675 | 18575 | 30176 | 0.064058 |
| fold_c | 81263 | 19588 | 18575 | 0.058358 |
| fold_d | 79848 | 19990 | 19588 | 0.071574 |
| fold_e | 68339 | 31097 | 19990 | 0.066633 |


### P1 thyroid_gland

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 132799 | 63864 | 64233 | 0.460168 |
| fold_b | 148072 | 48960 | 63864 | 0.444883 |
| fold_c | 172084 | 39852 | 48960 | 0.402431 |
| fold_d | 177057 | 43987 | 39852 | 0.451596 |
| fold_e | 152676 | 64233 | 43987 | 0.453338 |


### P1 tibial_nerve

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 118923 | 57787 | 56794 | 0.457179 |
| fold_b | 132877 | 42840 | 57787 | 0.447142 |
| fold_c | 154346 | 36318 | 42840 | 0.384500 |
| fold_d | 157421 | 39765 | 36318 | 0.425849 |
| fold_e | 136945 | 56794 | 39765 | 0.445014 |


### P1 body_of_pancreas

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 115604 | 55951 | 55994 | 0.456335 |
| fold_b | 129945 | 41653 | 55951 | 0.439563 |
| fold_c | 150220 | 35676 | 41653 | 0.413656 |
| fold_d | 153598 | 38275 | 35676 | 0.451704 |
| fold_e | 133280 | 55994 | 38275 | 0.432998 |


### P1 gastroesophageal_sphincter

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 104580 | 51128 | 50077 | 0.464565 |
| fold_b | 117406 | 37251 | 51128 | 0.470760 |
| fold_c | 136527 | 32007 | 37251 | 0.435102 |
| fold_d | 138456 | 35322 | 32007 | 0.460587 |
| fold_e | 120386 | 50077 | 35322 | 0.466055 |


### P1 Peyers_patch

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 118821 | 57727 | 58185 | 0.428375 |
| fold_b | 134470 | 42536 | 57727 | 0.413307 |
| fold_c | 155798 | 36399 | 42536 | 0.339994 |
| fold_d | 158448 | 39886 | 36399 | 0.405643 |
| fold_e | 136662 | 58185 | 39886 | 0.398887 |


### P2 CTCF

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 1140605 | 606910 | 643500 | 0.034098 |
| fold_b | 1446966 | 337139 | 606910 | 0.040291 |
| fold_c | 1641342 | 412534 | 337139 | 0.032346 |
| fold_d | 1587549 | 390932 | 412534 | 0.035609 |
| fold_e | 1356583 | 643500 | 390932 | 0.030494 |


### P2 H3K27ac

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 1512340 | 802822 | 853363 | 0.024003 |
| fold_b | 1916913 | 448790 | 802822 | 0.025605 |
| fold_c | 2176414 | 543321 | 448790 | 0.020921 |
| fold_d | 2104975 | 520229 | 543321 | 0.027389 |
| fold_e | 1794933 | 853363 | 520229 | 0.018307 |


### RNA ASE

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 805153 | 421401 | 414026 | 0.018624 |
| fold_b | 948223 | 270956 | 421401 | 0.014015 |
| fold_c | 1110880 | 258744 | 270956 | 0.009581 |
| fold_d | 1106383 | 275453 | 258744 | 0.012607 |
| fold_e | 951101 | 414026 | 275453 | 0.014779 |


### atac

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 1641835 | 820046 | 803274 | 0.040855 |
| fold_b | 1910179 | 534930 | 820046 | 0.042645 |
| fold_c | 2151299 | 578926 | 534930 | 0.038927 |
| fold_d | 2158250 | 527979 | 578926 | 0.041929 |
| fold_e | 1933902 | 803274 | 527979 | 0.038526 |


### h3k4me3

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 790252 | 413243 | 456253 | 0.042452 |
| fold_b | 1027901 | 218604 | 413243 | 0.047635 |
| fold_c | 1125571 | 315573 | 218604 | 0.040763 |
| fold_d | 1088100 | 256075 | 315573 | 0.048597 |
| fold_e | 947420 | 456253 | 256075 | 0.044792 |


### h3k27me3

| Fold | Train | Validation | Test | Test prevalence |
|---|---|---|---|---|
| fold_a | 623098 | 336978 | 331240 | 0.015397 |
| fold_b | 785348 | 168990 | 336978 | 0.023604 |
| fold_c | 890945 | 231381 | 168990 | 0.017569 |
| fold_d | 837208 | 222727 | 231381 | 0.017668 |
| fold_e | 737349 | 331240 | 222727 | 0.024945 |


### P0 source complexity counts

These are full-cohort counts; Appendix E reports means across test folds. They are not performance estimates.

| Stratum | Loci | AS-prone | Prevalence |
|---|---|---|---|
| low | 62388 | 5641 | 9.0418% |
| medium | 81260 | 8290 | 10.2018% |
| high | 107074 | 14161 | 13.2254% |

### P0 chromosome distribution (loci)

| Chromosome | Loci |
|---|---|
| chr1 | 22484 |
| chr2 | 20762 |
| chr3 | 16492 |
| chr4 | 12556 |
| chr5 | 13894 |
| chr6 | 15870 |
| chr7 | 13572 |
| chr8 | 12228 |
| chr9 | 11110 |
| chr10 | 13366 |
| chr11 | 12774 |
| chr12 | 12336 |
| chr13 | 6941 |
| chr14 | 8342 |
| chr15 | 8225 |
| chr16 | 8804 |
| chr17 | 10492 |
| chr18 | 5944 |
| chr19 | 8610 |
| chr20 | 7007 |
| chr21 | 3509 |
| chr22 | 5404 |

The preparation-era `p0_qc.json` supplies these counts only; its old “mapping pending” field is superseded by the successful mapping and completion receipts in Appendix A.4.

### SNV chromosome distribution (measurements)

| Chromosome | ctcf | h3k27ac | rna | atac | h3k4me3 | h3k27me3 |
|---|---|---|---|---|---|---|
| chr1 | 219465 | 309393 | 142316 | 264736 | 150712 | 110786 |
| chr2 | 171528 | 242912 | 130689 | 257071 | 111734 | 90772 |
| chr3 | 137566 | 190589 | 130648 | 204425 | 87669 | 53762 |
| chr4 | 99341 | 125058 | 82768 | 185321 | 70409 | 56715 |
| chr5 | 116341 | 148680 | 91179 | 172170 | 79459 | 61036 |
| chr6 | 162465 | 208077 | 109356 | 205132 | 132788 | 78366 |
| chr7 | 133887 | 151708 | 97893 | 198040 | 89290 | 79470 |
| chr8 | 104186 | 127279 | 67601 | 159859 | 64098 | 55851 |
| chr9 | 117785 | 156768 | 72027 | 173411 | 83970 | 76255 |
| chr10 | 120448 | 179136 | 91019 | 168307 | 76104 | 66749 |
| chr11 | 126197 | 162123 | 81241 | 162888 | 83450 | 57760 |
| chr12 | 120008 | 153363 | 91520 | 150227 | 88040 | 58073 |
| chr13 | 49804 | 70924 | 41095 | 90733 | 35129 | 34564 |
| chr14 | 79009 | 108466 | 47699 | 102157 | 52559 | 39165 |
| chr15 | 81768 | 110119 | 60531 | 96572 | 54104 | 41713 |
| chr16 | 91326 | 126748 | 58481 | 116979 | 63199 | 51289 |
| chr17 | 118624 | 171838 | 68921 | 137767 | 85397 | 64078 |
| chr18 | 45583 | 59998 | 31612 | 79913 | 31708 | 24813 |
| chr19 | 116399 | 153029 | 56250 | 118037 | 108635 | 59246 |
| chr20 | 72375 | 82294 | 32724 | 90930 | 46408 | 53229 |
| chr21 | 44047 | 47022 | 22632 | 53539 | 26104 | 33039 |
| chr22 | 62863 | 83001 | 32378 | 76941 | 38782 | 44585 |

### SNV donor distribution (measurements)

| Donor | ctcf | h3k27ac | rna | atac | h3k4me3 | h3k27me3 |
|---|---|---|---|---|---|---|
| ENC-001 | 550623 | 886183 | 605779 | 461572 | 434909 | 541128 |
| ENC-002 | 704695 | 729336 | 323658 | 937931 | 360577 | 392691 |
| ENC-003 | 550801 | 737589 | 400810 | 996352 | 488318 | 154656 |
| ENC-004 | 584896 | 815417 | 310333 | 869300 | 375944 | 202841 |

### SNV tissue distribution (measurements)

| Tissue | ctcf | h3k27ac | rna | atac | h3k4me3 | h3k27me3 |
|---|---|---|---|---|---|---|
| Peyers_patch | 106829 | 97733 | 50724 | 85099 | 78041 | 11918 |
| adrenal_gland | 145778 | 55690 | 73994 | 179007 | 27243 | 0 |
| ascending_aorta | 34124 | 57910 | 1023 | 0 | 38909 | 4367 |
| body_of_pancreas | 222315 | 113395 | 13929 | 164080 | 21372 | 9351 |
| breast_epithelium | 44621 | 97237 | 125268 | 375823 | 74279 | 30652 |
| coronary_artery | 18720 | 52517 | 0 | 36442 | 33877 | 0 |
| esophagus_muscularis_mucosa | 64511 | 177043 | 59630 | 16566 | 106067 | 30378 |
| esophagus_squamous_epithelium | 132925 | 165285 | 73697 | 13509 | 97402 | 75445 |
| gastrocnemius_medialis | 73595 | 310979 | 52650 | 238698 | 109062 | 160536 |
| gastroesophageal_sphincter | 109502 | 186053 | 38878 | 272052 | 69679 | 37426 |
| heart_left_ventricle | 54439 | 47393 | 19972 | 28723 | 59040 | 8119 |
| lower_leg_skin | 7287 | 0 | 44833 | 0 | 23752 | 0 |
| omental_fat_pad | 9254 | 0 | 99159 | 107142 | 0 | 0 |
| ovary | 37870 | 5611 | 37148 | 86170 | 17994 | 0 |
| prostate_gland | 59288 | 77750 | 22444 | 10058 | 43181 | 9107 |
| right_atrium_auricular_region | 33474 | 58713 | 24396 | 18111 | 38453 | 7092 |
| right_lobe_of_liver | 20224 | 61959 | 27394 | 57975 | 27263 | 0 |
| sigmoid_colon | 104612 | 203304 | 57538 | 490923 | 56542 | 52342 |
| spleen | 288974 | 351521 | 84237 | 101107 | 145083 | 441928 |
| stomach | 89066 | 126642 | 3096 | 200367 | 78313 | 67864 |
| subcutaneous_adipose_tissue | 0 | 0 | 70132 | 0 | 0 | 0 |
| suprapubic_skin | 19081 | 25189 | 55079 | 0 | 21194 | 31052 |
| testis | 35165 | 36496 | 197574 | 37361 | 33952 | 16067 |
| thoracic_aorta | 52426 | 60203 | 59617 | 10997 | 80623 | 1022 |
| thyroid_gland | 102237 | 111806 | 168507 | 179214 | 43809 | 16850 |
| tibial_artery | 29305 | 98700 | 0 | 111993 | 43435 | 45941 |
| tibial_nerve | 102807 | 100680 | 33149 | 34701 | 57764 | 21504 |
| transverse_colon | 143553 | 262909 | 80080 | 409037 | 78595 | 117970 |
| upper_lobe_of_left_lung | 159931 | 86134 | 58975 | 0 | 80614 | 74977 |
| uterus | 68438 | 90451 | 2387 | 0 | 39114 | 16841 |
| vagina | 20664 | 49222 | 5070 | 0 | 35096 | 2567 |

### All P1 tissue counts and exclusions

| Tissue | Selected | Loci | Active | Repressed | Conflict exclusions |
|---|---|---|---|---|---|
| Peyers_patch | True | 234737 | 93921 | 140816 | 26773 |
| adrenal_gland | False | 225108 | 139165 | 85943 | 30472 |
| ascending_aorta | False | 177722 | 81941 | 95781 | 7621 |
| body_of_pancreas | True | 227552 | 100064 | 127488 | 25486 |
| breast_epithelium | False | 187753 | 91458 | 96295 | 11900 |
| coronary_artery | False | 123352 | 83495 | 39857 | 2220 |
| esophagus_muscularis_mucosa | False | 199277 | 73184 | 126093 | 14805 |
| esophagus_squamous_epithelium | False | 191445 | 86095 | 105350 | 12070 |
| gastrocnemius_medialis | False | 224469 | 88824 | 135645 | 9210 |
| gastroesophageal_sphincter | True | 205788 | 94745 | 111043 | 14565 |
| heart_left_ventricle | False | 171650 | 81736 | 89914 | 7636 |
| ovary | False | 156098 | 90089 | 66009 | 6238 |
| prostate_gland | False | 153246 | 78037 | 75209 | 5353 |
| right_atrium_auricular_region | False | 178812 | 86810 | 92002 | 7546 |
| right_lobe_of_liver | False | 125292 | 68859 | 56433 | 6101 |
| sigmoid_colon | False | 205608 | 90532 | 115076 | 13550 |
| spleen | False | 184690 | 81376 | 103314 | 8855 |
| stomach | False | 187584 | 84962 | 102622 | 10863 |
| suprapubic_skin | False | 131185 | 62874 | 68311 | 3387 |
| testis | False | 165745 | 80603 | 85142 | 8168 |
| thoracic_aorta | False | 144005 | 86377 | 57628 | 6542 |
| thyroid_gland | True | 260898 | 115611 | 145287 | 32569 |
| tibial_artery | False | 206793 | 86356 | 120437 | 9943 |
| tibial_nerve | True | 233508 | 101438 | 132070 | 17605 |
| transverse_colon | False | 175162 | 86867 | 88295 | 7732 |
| upper_lobe_of_left_lung | False | 206688 | 88992 | 117696 | 16304 |
| uterus | False | 189582 | 84674 | 104908 | 9646 |
| vagina | False | 145246 | 75363 | 69883 | 6876 |

### Full accessible-source assay counts

| Assay | Measurements | Use in this report |
|---|---|---|
| HM-ChIP-seq_H3K27ac | 3168525 | Fitted full-source task |
| HM-ChIP-seq_H3K27me3 | 1291316 | Fitted full-source task |
| HM-ChIP-seq_H3K36me3 | 2078383 | No fitted endpoint reviewed |
| HM-ChIP-seq_H3K4me1 | 2233598 | No fitted endpoint reviewed |
| HM-ChIP-seq_H3K4me3 | 1659748 | Fitted full-source task |
| HM-ChIP-seq_H3K9me3 | 1769603 | No fitted endpoint reviewed |
| TF-ChIP-seq_CTCF | 2391015 | Fitted full-source task |
| TF-ChIP-seq_EP300 | 220043 | No fitted endpoint reviewed |
| TF-ChIP-seq_POLR2A | 857954 | No fitted endpoint reviewed |
| TF-ChIP-seq_POLR2AphosphoS5 | 843827 | No fitted endpoint reviewed |
| ATAC-seq | 3265155 | Fitted full-source task |
| RNA-seq | 2531272 | RNA high-confidence subset used instead |

## Appendix I. Graph-complexity contrasts across all tasks

Mean held-out size and prevalence, AP for each arm, and paired gains. Pointwise intervals; no correction across these many strata. These exploratory subsets do not establish a general complexity-dependent benefit. Normalized AP = (AP − prevalence)/(1 − prevalence); it adjusts the chance baseline but is not fully prevalence invariant.

### P0 AS-prone cCRE

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 12477.6 | 0.090691 | 0.117755 | 0.117306 | -0.000449 [-0.001383, +0.000453] | -0.000879 [-0.003325, +0.001509] | -0.000499 [-0.001528, +0.000497] |
| strict | medium | 16252.0 | 0.101949 | 0.129840 | 0.129344 | -0.000496 [-0.001436, +0.000664] | -0.001442 [-0.004379, +0.001293] | -0.000550 [-0.001594, +0.000742] |
| strict | high | 21414.8 | 0.130428 | 0.172560 | 0.173060 | +0.000500 [-0.000659, +0.001569] | +0.000806 [-0.001104, +0.002586] | +0.000579 [-0.000753, +0.001813] |
| 1hop | low | 12477.6 | 0.090691 | 0.117755 | 0.116619 | -0.001136 [-0.001836, -0.000457] | -0.001553 [-0.004014, +0.000475] | -0.001249 [-0.002014, -0.000504] |
| 1hop | medium | 16252.0 | 0.101949 | 0.129840 | 0.128429 | -0.001412 [-0.002348, -0.000620] | -0.001470 [-0.002552, -0.000287] | -0.001569 [-0.002604, -0.000691] |
| 1hop | high | 21414.8 | 0.130428 | 0.172560 | 0.172162 | -0.000398 [-0.002298, +0.001443] | -0.000501 [-0.003058, +0.002180] | -0.000430 [-0.002589, +0.001689] |

### P0 exposure matched

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 2159.8 | 0.478610 | 0.501372 | 0.501096 | -0.000276 [-0.002811, +0.002324] | -0.000437 [-0.003213, +0.002352] | -0.000526 [-0.005371, +0.004451] |
| strict | medium | 3095.2 | 0.487743 | 0.506341 | 0.507777 | +0.001436 [-0.001001, +0.004500] | +0.000071 [-0.002912, +0.002826] | +0.002780 [-0.001992, +0.008762] |
| strict | high | 4875.8 | 0.519433 | 0.560921 | 0.562494 | +0.001572 [-0.001076, +0.004244] | +0.001509 [-0.001502, +0.003907] | +0.003305 [-0.002222, +0.008934] |
| 1hop | low | 2159.8 | 0.478610 | 0.501372 | 0.500962 | -0.000410 [-0.002692, +0.001635] | -0.000194 [-0.002011, +0.001956] | -0.000781 [-0.005135, +0.003116] |
| 1hop | medium | 3095.2 | 0.487743 | 0.506341 | 0.509632 | +0.003291 [+0.001576, +0.004978] | +0.001927 [-0.000352, +0.003943] | +0.006466 [+0.003082, +0.009830] |
| 1hop | high | 4875.8 | 0.519433 | 0.560921 | 0.562656 | +0.001735 [-0.001058, +0.004303] | +0.001977 [-0.000515, +0.004524] | +0.003558 [-0.002233, +0.008864] |

### P0 H3K27ac only

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 6825.4 | 0.035229 | 0.049765 | 0.049058 | -0.000707 [-0.001999, +0.000308] | +0.001271 [-0.002212, +0.004143] | -0.000734 [-0.002074, +0.000319] |
| strict | medium | 9480.4 | 0.037488 | 0.047065 | 0.048202 | +0.001137 [-0.000941, +0.004804] | +0.002596 [-0.001288, +0.006443] | +0.001194 [-0.000975, +0.005027] |
| strict | high | 13077.0 | 0.057100 | 0.075407 | 0.075779 | +0.000372 [-0.000918, +0.001471] | +0.002845 [-0.001580, +0.006946] | +0.000396 [-0.000974, +0.001563] |
| 1hop | low | 6825.4 | 0.035229 | 0.049765 | 0.049175 | -0.000590 [-0.002110, +0.000937] | -0.001039 [-0.008209, +0.005556] | -0.000608 [-0.002182, +0.000975] |
| 1hop | medium | 9480.4 | 0.037488 | 0.047065 | 0.049437 | +0.002372 [-0.000197, +0.006187] | +0.000217 [-0.002744, +0.003185] | +0.002480 [-0.000204, +0.006472] |
| 1hop | high | 13077.0 | 0.057100 | 0.075407 | 0.075528 | +0.000121 [-0.001366, +0.001724] | +0.003921 [+0.000123, +0.007694] | +0.000133 [-0.001443, +0.001836] |

### P0 CTCF only

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 5533.4 | 0.059804 | 0.070388 | 0.069995 | -0.000393 [-0.002363, +0.001140] | +0.000113 [-0.005276, +0.003880] | -0.000422 [-0.002515, +0.001201] |
| strict | medium | 7590.4 | 0.061854 | 0.069331 | 0.069692 | +0.000362 [-0.000573, +0.001036] | -0.002314 [-0.005735, +0.001480] | +0.000385 [-0.000612, +0.001105] |
| strict | high | 10761.4 | 0.070505 | 0.089067 | 0.089417 | +0.000351 [-0.000919, +0.001747] | +0.000285 [-0.002896, +0.002603] | +0.000380 [-0.000985, +0.001882] |
| 1hop | low | 5533.4 | 0.059804 | 0.070388 | 0.069868 | -0.000520 [-0.001836, +0.000869] | -0.001868 [-0.004963, +0.001466] | -0.000561 [-0.001961, +0.000914] |
| 1hop | medium | 7590.4 | 0.061854 | 0.069331 | 0.070889 | +0.001559 [+0.000640, +0.002436] | +0.000161 [-0.002219, +0.002070] | +0.001662 [+0.000684, +0.002595] |
| 1hop | high | 10761.4 | 0.070505 | 0.089067 | 0.089862 | +0.000795 [-0.000310, +0.001768] | +0.002359 [-0.000323, +0.004972] | +0.000857 [-0.000332, +0.001903] |

### P1 thyroid_gland

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 15033.0 | 0.393148 | 0.509559 | 0.509525 | -0.000034 [-0.001939, +0.001475] | -0.000294 [-0.001324, +0.000984] | -0.000065 [-0.003282, +0.002486] |
| strict | medium | 17587.8 | 0.440225 | 0.571858 | 0.573123 | +0.001264 [-0.000871, +0.003701] | +0.000079 [-0.000619, +0.000865] | +0.002205 [-0.001665, +0.006540] |
| strict | high | 19558.4 | 0.480809 | 0.596997 | 0.599553 | +0.002556 [+0.000572, +0.004851] | +0.000952 [-0.000327, +0.002240] | +0.004789 [+0.001049, +0.009101] |
| 1hop | low | 15033.0 | 0.393148 | 0.509559 | 0.509162 | -0.000397 [-0.002122, +0.001639] | +0.000447 [-0.000714, +0.001612] | -0.000677 [-0.003512, +0.002599] |
| 1hop | medium | 17587.8 | 0.440225 | 0.571858 | 0.573565 | +0.001706 [-0.000471, +0.004137] | +0.000620 [-0.001056, +0.002268] | +0.002956 [-0.000943, +0.007240] |
| 1hop | high | 19558.4 | 0.480809 | 0.596997 | 0.599189 | +0.002192 [-0.002769, +0.006284] | +0.001418 [-0.001008, +0.003141] | +0.004660 [-0.004461, +0.012382] |

### P1 tibial_nerve

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 12888.2 | 0.396576 | 0.532875 | 0.532482 | -0.000393 [-0.003038, +0.002051] | -0.000650 [-0.003010, +0.001844] | -0.000713 [-0.005053, +0.003252] |
| strict | medium | 15233.4 | 0.429575 | 0.592047 | 0.592170 | +0.000123 [-0.003393, +0.003139] | -0.000373 [-0.002985, +0.002414] | +0.000134 [-0.006058, +0.005437] |
| strict | high | 18579.2 | 0.455435 | 0.597869 | 0.603849 | +0.005979 [+0.003899, +0.008420] | +0.003336 [+0.000232, +0.006476] | +0.011068 [+0.007143, +0.015126] |
| 1hop | low | 12888.2 | 0.396576 | 0.532875 | 0.533275 | +0.000400 [-0.002207, +0.002926] | +0.000485 [-0.001577, +0.002866] | +0.000780 [-0.003530, +0.004985] |
| 1hop | medium | 15233.4 | 0.429575 | 0.592047 | 0.591610 | -0.000437 [-0.004213, +0.003727] | -0.000677 [-0.002877, +0.001841] | -0.000945 [-0.007423, +0.006188] |
| 1hop | high | 18579.2 | 0.455435 | 0.597869 | 0.601393 | +0.003524 [-0.003295, +0.008847] | +0.000970 [-0.005352, +0.005163] | +0.007143 [-0.004829, +0.016810] |

### P1 body_of_pancreas

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 12941.2 | 0.403391 | 0.507969 | 0.507346 | -0.000624 [-0.002686, +0.001014] | -0.000681 [-0.002107, +0.000630] | -0.001054 [-0.004478, +0.001686] |
| strict | medium | 14977.8 | 0.432331 | 0.552046 | 0.552610 | +0.000564 [-0.002162, +0.004140] | +0.000208 [-0.001271, +0.001669] | +0.000937 [-0.003863, +0.007203] |
| strict | high | 17590.8 | 0.467269 | 0.574359 | 0.576532 | +0.002173 [+0.000289, +0.004182] | +0.001560 [-0.000021, +0.003074] | +0.003911 [+0.000504, +0.007447] |
| 1hop | low | 12941.2 | 0.403391 | 0.507969 | 0.508943 | +0.000974 [-0.000969, +0.002614] | +0.000790 [-0.000548, +0.001992] | +0.001658 [-0.001607, +0.004413] |
| 1hop | medium | 14977.8 | 0.432331 | 0.552046 | 0.550981 | -0.001066 [-0.002905, +0.001135] | -0.000930 [-0.002412, +0.000765] | -0.001914 [-0.005166, +0.001947] |
| 1hop | high | 17590.8 | 0.467269 | 0.574359 | 0.576168 | +0.001809 [-0.000240, +0.004058] | +0.001774 [-0.000043, +0.003591] | +0.003393 [-0.000493, +0.007622] |

### P1 gastroesophageal_sphincter

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 11674.8 | 0.443967 | 0.543423 | 0.541884 | -0.001539 [-0.004772, +0.001590] | -0.001746 [-0.003977, +0.000348] | -0.002675 [-0.008442, +0.002900] |
| strict | medium | 13520.8 | 0.465960 | 0.584399 | 0.582589 | -0.001811 [-0.004888, +0.000731] | -0.000653 [-0.002950, +0.001581] | -0.003407 [-0.009211, +0.001355] |
| strict | high | 15961.4 | 0.463326 | 0.576288 | 0.579567 | +0.003279 [-0.000029, +0.006614] | +0.001597 [-0.000887, +0.004411] | +0.006027 [-0.000163, +0.012308] |
| 1hop | low | 11674.8 | 0.443967 | 0.543423 | 0.541447 | -0.001977 [-0.004914, +0.000911] | -0.001846 [-0.004057, -0.000230] | -0.003426 [-0.008698, +0.001794] |
| 1hop | medium | 13520.8 | 0.465960 | 0.584399 | 0.583058 | -0.001341 [-0.005920, +0.002512] | -0.001738 [-0.005537, +0.001731] | -0.002406 [-0.010850, +0.004776] |
| 1hop | high | 15961.4 | 0.463326 | 0.576288 | 0.578748 | +0.002460 [-0.007222, +0.010112] | +0.000244 [-0.007914, +0.006149] | +0.005218 [-0.011934, +0.018969] |

### P1 Peyers_patch

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 12752.8 | 0.336238 | 0.496841 | 0.495992 | -0.000850 [-0.003921, +0.002195] | -0.000889 [-0.002570, +0.001017] | -0.001224 [-0.005817, +0.003371] |
| strict | medium | 15409.8 | 0.390869 | 0.563088 | 0.561624 | -0.001464 [-0.003914, +0.001025] | -0.000586 [-0.001825, +0.000657] | -0.002443 [-0.006445, +0.001640] |
| strict | high | 18784.0 | 0.440757 | 0.585884 | 0.585356 | -0.000528 [-0.001804, +0.000797] | -0.000304 [-0.001236, +0.000643] | -0.000910 [-0.003321, +0.001535] |
| 1hop | low | 12752.8 | 0.336238 | 0.496841 | 0.497660 | +0.000819 [-0.001660, +0.003067] | +0.000899 [-0.000884, +0.002620] | +0.001317 [-0.002462, +0.004817] |
| 1hop | medium | 15409.8 | 0.390869 | 0.563088 | 0.561074 | -0.002014 [-0.004162, -0.000209] | -0.000744 [-0.001607, +0.000010] | -0.003279 [-0.006748, -0.000345] |
| 1hop | high | 18784.0 | 0.440757 | 0.585884 | 0.586076 | +0.000191 [-0.002004, +0.001945] | +0.000407 [-0.000857, +0.001697] | +0.000582 [-0.003014, +0.003592] |

### CTCF SNVs

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 104046.2 | 0.026057 | 0.043303 | 0.042031 | -0.001272 [-0.003082, +0.000378] | -0.003259 [-0.010094, +0.003269] | -0.001303 [-0.003162, +0.000389] |
| strict | medium | 145751.2 | 0.032184 | 0.068291 | 0.069488 | +0.001197 [-0.001247, +0.004157] | -0.000924 [-0.006816, +0.004278] | +0.001224 [-0.001296, +0.004271] |
| strict | high | 228390.6 | 0.040389 | 0.066625 | 0.068983 | +0.002358 [-0.000226, +0.006485] | +0.003094 [-0.001722, +0.007411] | +0.002454 [-0.000236, +0.006751] |
| 1hop | low | 104046.2 | 0.026057 | 0.043303 | 0.041446 | -0.001857 [-0.004881, +0.000292] | -0.000590 [-0.005090, +0.004449] | -0.001901 [-0.004993, +0.000301] |
| 1hop | medium | 145751.2 | 0.032184 | 0.068291 | 0.069132 | +0.000842 [-0.000864, +0.003920] | +0.000435 [-0.004518, +0.005009] | +0.000868 [-0.000888, +0.004029] |
| 1hop | high | 228390.6 | 0.040389 | 0.066625 | 0.070886 | +0.004261 [+0.002256, +0.006560] | +0.009585 [-0.001444, +0.017763] | +0.004438 [+0.002351, +0.006834] |

### H3K27ac SNVs

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 133325.4 | 0.015872 | 0.032043 | 0.030600 | -0.001443 [-0.003613, +0.000429] | -0.000213 [-0.009662, +0.007493] | -0.001466 [-0.003671, +0.000436] |
| strict | medium | 193175.0 | 0.020483 | 0.063856 | 0.068494 | +0.004638 [-0.000817, +0.015690] | +0.004488 [-0.004623, +0.016438] | +0.004795 [-0.000828, +0.016229] |
| strict | high | 307196.0 | 0.028381 | 0.054470 | 0.054052 | -0.000418 [-0.002290, +0.001694] | -0.001733 [-0.023027, +0.016746] | -0.000424 [-0.002353, +0.001758] |
| 1hop | low | 133325.4 | 0.015872 | 0.032043 | 0.032425 | +0.000381 [-0.000745, +0.001524] | +0.001600 [-0.011369, +0.016300] | +0.000388 [-0.000757, +0.001550] |
| 1hop | medium | 193175.0 | 0.020483 | 0.063856 | 0.065769 | +0.001914 [-0.000547, +0.005962] | +0.001596 [-0.008909, +0.015190] | +0.001980 [-0.000557, +0.006163] |
| 1hop | high | 307196.0 | 0.028381 | 0.054470 | 0.058925 | +0.004454 [-0.000563, +0.010572] | +0.019627 [+0.002493, +0.041445] | +0.004603 [-0.000576, +0.010951] |

### RNA ASE

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 82648.2 | 0.007935 | 0.012115 | 0.013475 | +0.001360 [-0.000137, +0.003498] | +0.003705 [-0.006053, +0.015081] | +0.001373 [-0.000138, +0.003532] |
| strict | medium | 104384.8 | 0.010448 | 0.015753 | 0.016140 | +0.000387 [-0.000351, +0.001303] | +0.000503 [-0.006915, +0.009496] | +0.000392 [-0.000355, +0.001318] |
| strict | high | 141083.0 | 0.019833 | 0.035318 | 0.036856 | +0.001538 [-0.000971, +0.005358] | +0.005216 [-0.005414, +0.018390] | +0.001590 [-0.000988, +0.005526] |
| 1hop | low | 82648.2 | 0.007935 | 0.012115 | 0.012698 | +0.000582 [-0.000252, +0.001507] | -0.004773 [-0.018760, +0.007706] | +0.000588 [-0.000254, +0.001520] |
| 1hop | medium | 104384.8 | 0.010448 | 0.015753 | 0.016254 | +0.000501 [-0.000391, +0.001783] | +0.003661 [+0.000096, +0.008240] | +0.000507 [-0.000396, +0.001803] |
| 1hop | high | 141083.0 | 0.019833 | 0.035318 | 0.037065 | +0.001746 [-0.001211, +0.005751] | -0.001391 [-0.013706, +0.011455] | +0.001792 [-0.001235, +0.005911] |

### ATAC SNVs

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 151786.8 | 0.035306 | 0.039557 | 0.039791 | +0.000234 [-0.000119, +0.000522] | +0.001201 [-0.000966, +0.003235] | +0.000242 [-0.000123, +0.000541] |
| strict | medium | 197092.0 | 0.038747 | 0.047527 | 0.047274 | -0.000253 [-0.000698, +0.000186] | -0.000118 [-0.002545, +0.002247] | -0.000263 [-0.000727, +0.000195] |
| strict | high | 304145.8 | 0.044554 | 0.051178 | 0.050407 | -0.000771 [-0.001721, -0.000071] | -0.003394 [-0.007516, -0.000004] | -0.000808 [-0.001804, -0.000074] |
| 1hop | low | 151786.8 | 0.035306 | 0.039557 | 0.039144 | -0.000414 [-0.000969, +0.000112] | -0.001253 [-0.003551, +0.001139] | -0.000429 [-0.001004, +0.000116] |
| 1hop | medium | 197092.0 | 0.038747 | 0.047527 | 0.047266 | -0.000261 [-0.000953, +0.000518] | +0.000831 [-0.003291, +0.005331] | -0.000270 [-0.000994, +0.000547] |
| 1hop | high | 304145.8 | 0.044554 | 0.051178 | 0.052234 | +0.001057 [-0.000427, +0.003488] | +0.002087 [-0.003096, +0.008503] | +0.001110 [-0.000447, +0.003663] |

### H3K4me3 SNVs

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 67791.2 | 0.031344 | 0.058596 | 0.060036 | +0.001440 [-0.002480, +0.005740] | +0.013902 [+0.002313, +0.028254] | +0.001496 [-0.002550, +0.005946] |
| strict | medium | 95878.8 | 0.037274 | 0.073767 | 0.077485 | +0.003718 [-0.000139, +0.008862] | -0.004825 [-0.012796, +0.003320] | +0.003884 [-0.000141, +0.009311] |
| strict | high | 168255.4 | 0.055736 | 0.092310 | 0.096300 | +0.003991 [-0.003223, +0.012818] | +0.002987 [-0.019872, +0.022597] | +0.004278 [-0.003417, +0.013703] |
| 1hop | low | 67791.2 | 0.031344 | 0.058596 | 0.059005 | +0.000409 [-0.006326, +0.007792] | +0.018657 [-0.010098, +0.056476] | +0.000441 [-0.006550, +0.008120] |
| 1hop | medium | 95878.8 | 0.037274 | 0.073767 | 0.077835 | +0.004068 [-0.000637, +0.009975] | +0.003946 [-0.007477, +0.015462] | +0.004235 [-0.000691, +0.010431] |
| 1hop | high | 168255.4 | 0.055736 | 0.092310 | 0.103551 | +0.011242 [-0.000363, +0.024533] | +0.032125 [+0.002782, +0.076312] | +0.011973 [-0.000333, +0.026053] |

### H3K27me3 SNVs

| Context | Complexity | Mean test n | Prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| strict | low | 49496.4 | 0.014749 | 0.080123 | 0.076979 | -0.003144 [-0.012788, +0.002586] | -0.003199 [-0.009147, +0.002383] | -0.003195 [-0.013019, +0.002629] |
| strict | medium | 67573.6 | 0.014740 | 0.105350 | 0.109267 | +0.003917 [-0.009200, +0.017032] | -0.005194 [-0.016397, +0.006832] | +0.003986 [-0.009348, +0.017308] |
| strict | high | 141167.0 | 0.023759 | 0.087460 | 0.091972 | +0.004512 [+0.000911, +0.011472] | +0.003006 [-0.001479, +0.008223] | +0.004654 [+0.000931, +0.011880] |
| 1hop | low | 49496.4 | 0.014749 | 0.080123 | 0.081431 | +0.001308 [-0.002880, +0.007586] | -0.013556 [-0.026929, -0.001622] | +0.001339 [-0.002917, +0.007725] |
| 1hop | medium | 67573.6 | 0.014740 | 0.105350 | 0.102207 | -0.003142 [-0.019578, +0.006305] | -0.007437 [-0.015281, +0.002141] | -0.003182 [-0.019891, +0.006416] |
| 1hop | high | 141167.0 | 0.023759 | 0.087460 | 0.096804 | +0.009344 [-0.000460, +0.026486] | +0.004076 [-0.002158, +0.010637] | +0.009671 [-0.000468, +0.027452] |

## Appendix J. Limitations, unfinished evidence and reproducibility

**Supported:** frozen representations can be reused across these measurement-derived tasks, with small, assay-dependent incremental AP; all loci remain chromosome-held-out and the completed programme includes null results. **Unresolved:** learned-versus-random attribution for EN-TEx, robust external/donor-held-out transfer, improved allele-specific prediction from actual allele sequences, and gains beyond stronger full-context sequence baselines.

Key limitations:

1. Four biological donors, many correlated experiments; millions of measurements are not millions of independent individuals.
2. Static locus/segment features cannot distinguish donors, tissues or two alleles at the same locus; distinct nearby loci may share the same feature vector.
3. “Any AS” and measurement-weighted targets depend on detection opportunity and source ascertainment.
4. P1 conflicting states are excluded; state labels may summarize differing donor/assay contexts rather than a universal tissue truth.
5. End-sampled NT segments are not an allele-centred local sequence baseline.
6. Original pretraining contains a masking shortcut and incompletely controlled RNG states. Unmasked extraction does not undo what pretrained weights learned.
7. Existing H/R controls concern cCRE/SV tasks; do not transfer their attribution conclusions quantitatively to EN-TEx.
8. Repeated inspection of the same chromosome folds makes subsequent design selection exploratory.
9. Raw-library QC is inherited from EN-TEx, not independently reproduced here.
10. Historical QC documents marked mapping or fitting “pending” at preparation time. Later mapping/completion/per-run outputs supersede those fields; the old receipts remain as an audit trail.

### Not yet demonstrated

- Full EN-TEx E-versus-random-versus-H comparison, strict donor/tissue-held-out fits and genotype/haplotype-specific AS direction prediction.
- Full long-sequence/local-allele baseline and a completed length-weighted P0 pooling sensitivity.
- Additional default-source assays H3K36me3, H3K4me1, H3K9me3, EP300, POLR2A and POLR2AphosphoS5; their source availability is not a fitted result.
- Independent FASTQ-level QC. None is claimed as completed here.

### Reproducibility record

- Core execution receipt: commit `3475d979943ea800371e27e633ec9f4928adc02d`.
- Protocol identifiers: `entex_v1`, `entex_meeting_20260929`, `entex_extension_20260927`; raw-source and graph hashes retained in Appendix A.
- Order of operations: validate supplied calls → prepare compact task tables → map to pinned HPRC R2 segments → common C/K/S/T coverage → chromosome partitions → validation-only calibration/threshold selection → frozen-feature probes → paired summaries → saved-prediction sensitivities.
- Original large graph, checkpoints, embeddings and full predictions retained on the lab server. This briefing reviews exported metrics/QC; it does not re-run training or independently re-bootstrap intervals.
- Source reference: Rozowsky and colleagues, “The EN-TEx resource of multi-tissue personal epigenomes & variant-impact models”, Cell 186(7), 1493–1511.e40, 2023. Resource descriptions do not independently validate PangenomeFM performance.

### Review checks

- [x] All 450 configurations and 3,150 feature-specific metric records complete.
- [x] Paired AP arithmetic and seven-metric summary means agree within 1e-12.
- [x] Measurement/locus/donor units, registry correction and feature-coverage limits explicit.
- [x] All detailed scientific tables retained unchanged; overview labels rewritten and notation explained.
- [x] Null and negative outcomes retained alongside positive pointwise intervals.
- [x] No discussion questions, hyperlinks or external-document navigation.
- [x] Main briefing organized into seven timed blocks totalling 15 minutes.
- [ ] Learned-versus-random attribution and independent biological confirmation remain scientific work.
