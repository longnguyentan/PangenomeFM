# EN-TEx: complete results, data quality and interpretation for the PangenomeFM team

Takeaway: EN-TEx shows small assay-dependent graph gains; regulatory-locus and enhancer results remain inconclusive.

**Evidence reviewed: 29 September 2026 (America/New_York).** Prepared for team discussion. This report supersedes the incomplete extension status in the earlier meeting brief.

We evaluated whether frozen pangenome representations improve biological prediction beyond genomic coordinates and nucleotide embeddings. Saved outputs contain 450 complete fold–run–context configurations across 15 datasets, each with seven feature combinations. AS-prone cCREs, the enhancer-state tissue macro and RNA allele-specific expression remain inconclusive globally. CTCF, H3K27ac and H3K4me3 show small positive estimates in some settings, but weighting and multiplicity analyses narrow the claims. The representations describe loci, not donor-specific alleles, and these EN-TEx evaluations do not establish that learned weights outperform matched untrained encoders. The next useful step is a controlled representation comparison on the existing endpoints, with the repeated-measurement and allele-resolution limitations explicitly tested.

## 1. Scientific context and question

EN-TEx combines personal genomes and functional assays across tissues from four individuals. Its published resource contains 1,635 datasets; our experiments use released, processed accessible/testable AS calls and active/repressed cCRE annotations, not all raw experiments. [Original EN-TEx paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10074325/) · [official portal](https://entex.encodeproject.org/main.html).

The primary question is whether adding frozen topology T improves average precision (AP, reported as AUPRC in the code) beyond C+S on identical examples: `ΔT = AP(C+S+T) − AP(C+S)`. The reciprocal contrast is `ΔS = AP(C+S+T) − AP(C+T)`. A positive topology increment is neither proof of causal biological information nor proof that pretraining was necessary.

### Experimental units and labels

| Task | Unit and positive | Negative / exclusions |
|---|---|---|
| P0 | One cCRE locus; any supplied significant AS call among informative measurements | Measured/testable with no significant call; absent/unmeasured elements never become negative |
| P0b | Existing P0 predictions within predefined graph-complexity strata | No re-binning or selection using performance |
| P0 sensitivities | Exact exposure matching, H3K27ac-only, CTCF-only | Definitions fixed before P0 biological fitting; matching after common feature coverage |
| P1 | A distal enhancer-like cCRE in one tissue, explicitly active | Explicitly repressed; require V2 dELS and supplied distal state; exclude conflicting locus/tissue labels |
| P2 / extensions | A measured heterozygous SNV/experiment, supplied AS significance=1 | Accessible/informative SNV measurement with significance=0; no absence-based negatives |

SNV features and predictions are constant for every occurrence of a locus within a run. Training aggregates identical locus/label examples with counts while retaining original measurement class weights and scaler occurrence weights. A locus may have both AS and non-AS measurements across experiments. Thus the task measures propensity for imbalance within the sampled measurements; it does not predict the favoured allele, its direction of effect, a causal variant, or a donor/tissue-specific response.

## 2. Lifecycle and completeness

| Programme | Configurations | Status |
|---|---|---|
| P0 + 3 sensitivities | 120 | Complete |
| P1, 5 tissues | 150 | Complete |
| CTCF/H3K27ac SNVs | 60 | Complete |
| RNA ASE | 30 | Complete |
| ATAC/H3K4me3/H3K27me3 SNVs | 90 | Complete |
| Total | 450 | 3,150 feature-specific classifier metric records |

Here “configuration” means one task × fold × run identifier × graph context, with seven separately fitted feature-set probes. It is not 450 donors or biological replicates. We checked all 3,150 per-run metric rows, five folds, three run identifiers, two contexts, and all seven features; recomputed summary means and paired ΔT from the saved metrics agree within 1e-12. CIs are the saved hierarchical-bootstrap estimates, not newly bootstrapped predictions.

Completed follow-ups re-evaluate 150 existing prediction configurations (60 CTCF/H3K27ac and 90 extension runs); they are not additional training runs. Full prediction arrays and large graph/cache files remain on the lab server. The present review checks locally saved result tables, QC and audit receipts, and does not claim a new server rerun. These are custom repository runs, not registered NGS Workbench jobs.

## 3. Data quality and provenance

### 3.1 Source files and schemas

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

### 3.2 Assay-specific SNV data

| Assay/source | Measurements | Unique loci | AS+ measurements | AS− measurements | Prevalence | Donors | Tissues |
|---|---|---|---|---|---|---|---|
| ctcf | 2391015 | 596650 | 83911 | 2307104 | 3.5094% | 4 | 30 |
| h3k27ac | 3168525 | 713418 | 74833 | 3093692 | 2.3618% | 4 | 28 |
| rna | 1640580 | 466867 | 23546 | 1617034 | 1.4352% | 4 | 29 |
| atac | 3265155 | 1498771 | 133227 | 3131928 | 4.0803% | 4 | 24 |
| h3k4me3 | 1659748 | 265099 | 74771 | 1584977 | 4.5050% | 4 | 29 |
| h3k27me3 | 1291316 | 702509 | 25667 | 1265649 | 1.9877% | 4 | 24 |

All six prepared SNV datasets record zero identical-duplicate exclusions. CTCF and H3K27ac come from the full accessible source; the initially supplied high-confidence file contains RNA-seq only. The full source contains 22,310,439 measurements across 12 assays and was parsed in chunks, once into compact caches. RNA here uses the high-confidence subset (1,640,580 measurements), not all 2,531,272 RNA measurements in the default file. Source selection therefore differs by endpoint.

### 3.3 cCREs, tissue selection and registry mismatch

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

### 3.4 Mapping and feature coverage

All tasks use the same checksum-pinned HPRC R2 SV graph as the manuscript. The existing half-open interval mapper was reused. P0 maps 250,722/250,722 loci: 247,382 overlap one segment and 3,340 overlap multiple segments (1.3322%). The default mean pools segment features; length-weighted pooling is implemented, but a completed length-weighted sensitivity is not established by the reviewed outputs. SNVs use only the containing segment; no neighbourhood pooling extension was added. All six SNV datasets map 100% and have no multi-segment SNVs. The original P1 completion record reports 100% mapping; its per-run feature-coverage receipts are available locally, but the separate full mapping receipt was not re-exported for this review.

Joint C/K/S/T coverage is 100% across the 180 P0/sensitivity/CTCF/H3K27ac configurations. Across 150 P1 configurations, common coverage is at least 99.9982869966%; two to four examples per run are excluded under the existing topology-extraction rules. They are removed from every feature arm, not just T. Do not sum these repeated fold/context exclusions into a count of unique loci. RNA/extension preparation, smoke and completed summary/follow-up audits are available; this review does not have a uniform 120-run raw feature-audit export for those later fits, so their every-run feature coverage is not independently re-certified here.

### 3.5 Resource identities

Graph segment SHA256: `e0d832a820969403797662f9af267440599898f068ba9df4069b114d669a3347`. NT: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, revision `81b29e5786726d891dbf929404ef20adca5b36f1`. Registry SHA256: `16fe76cbbc1f24e38a5476ce44fa4b81a9612f61a23517860d6e61f5619615ec`. No release or checkpoint substitution is allowed.

| Source | SHA256 |
|---|---|
| cCREs_default_AS.tsv | 679d916cf78008b0f6c279a30ef7131d8e6a5645c7bb5e6051a7906c0500ba28 |
| hetSNVs_high-confidence_AS.tsv | 01b6b7d27446010e214a8cdf1569312cb253c6bedc064a942554b9e28f0927e1 |
| active.combined_set.txt.zip | 676e62e3e2fbeae71d1ff39d09121a5b574ccd4079b9609c27d1dcc8afde9637 |
| repressed.combined_set.txt.zip | 0096f4c5d0fce555bf67d3e8dddf52413e471be4a38ab86c8ede937dbdcc3770 |
| hetSNVs_default_AS.tsv | e59a83a1595cbe12ba56714af79c297ac9f31b593e13ee44966c732d30d4daa8 |

## 4. Model, splits and statistics

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

## 5. Complete primary and sensitivity results

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

## 6. Measurement exposure and donor/tissue sensitivity

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

## 7. Graph complexity: within-stratum evidence

Use the existing label-free native complexity v2 categories and locus-start regional assignment, fixed independently of EN-TEx performance. AS prevalence increasing with complexity does not show that adding T helps. The following P0 comparison uses the same examples within each stratum; normalized AP and AUROC appear alongside AP.

| Context | Stratum | Mean test n | Mean prevalence | AP C+S | AP C+S+T | ΔAP [95% CI] | ΔAUROC [95% CI] | Δnormalized AP [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 1hop | high | 21414.8 | 0.130428 | 0.172560 | 0.172162 | -0.000398 [-0.002298, +0.001443] | -0.000501 [-0.003058, +0.002180] | -0.000430 [-0.002589, +0.001689] |
| 1hop | low | 12477.6 | 0.090691 | 0.117755 | 0.116619 | -0.001136 [-0.001836, -0.000457] | -0.001553 [-0.004014, +0.000475] | -0.001249 [-0.002014, -0.000504] |
| 1hop | medium | 16252.0 | 0.101949 | 0.129840 | 0.128429 | -0.001412 [-0.002348, -0.000620] | -0.001470 [-0.002552, -0.000287] | -0.001569 [-0.002604, -0.000691] |
| strict | high | 21414.8 | 0.130428 | 0.172560 | 0.173060 | +0.000500 [-0.000659, +0.001569] | +0.000806 [-0.001104, +0.002586] | +0.000579 [-0.000753, +0.001813] |
| strict | low | 12477.6 | 0.090691 | 0.117755 | 0.117306 | -0.000449 [-0.001383, +0.000453] | -0.000879 [-0.003325, +0.001509] | -0.000499 [-0.001528, +0.000497] |
| strict | medium | 16252.0 | 0.101949 | 0.129840 | 0.129344 | -0.000496 [-0.001436, +0.000664] | -0.001442 [-0.004379, +0.001293] | -0.000550 [-0.001594, +0.000742] |

P0 does not establish a positive high-complexity topology gain. In one-hop context, low- and medium-complexity intervals are negative. Prevalence-normalization does not turn this into evidence of beneficial topology. Every task’s full complexity table is linked in Section 10; no bin was chosen for a favorable outcome.

## 8. What these results do and do not establish

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

## 9. Suggested discussion at the group meeting

1. Is the next target **locus susceptibility** or **individual allele response**? These require different representations and labels.
2. Should H3K27ac/H3K4me3 become fixed development endpoints, with CTCF/ATAC/RNA/null tasks retained as checks rather than discarded?
3. Compare trained and matched untrained encoders plus H on the same EN-TEx examples before attributing small gains to pretraining.
4. Fit an equal-locus-weight sensitivity, separate from the already completed evaluation-only reweighting; keep chromosome splits unchanged.
5. Compare an allele-centred frozen sequence representation against the segment representation on identical loci, reporting actual variant-base coverage.
6. Reserve untouched external/donor-aware evidence before selecting the model on repeatedly viewed folds.
7. Do not add assays solely to obtain a positive interval; prioritize correcting representation and attribution limitations.

### Work still not demonstrated

- Full EN-TEx E-versus-random-versus-H comparison, strict donor/tissue-held-out fits and genotype/haplotype-specific AS direction prediction.
- Full long-sequence/local-allele baseline and a completed length-weighted P0 pooling sensitivity.
- Additional default-source assays H3K36me3, H3K4me1, H3K9me3, EP300, POLR2A and POLR2AphosphoS5; their source availability is not a fitted result.
- Independent FASTQ-level QC. None is claimed as completed here.

## 10. Verified artifacts and reproduction

### Results and QC

- **P0 AS-prone cCRE:** [per-run metrics](../results/entex/v1/p0_analysis/per_run.csv) · [all features/metrics](../results/entex/v1/p0_analysis/summary.csv) · [paired AP contrasts](../results/entex/v1/p0_analysis/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p0_analysis/complexity_metric_summary.csv)
- **P0 exposure matched:** [per-run metrics](../results/entex/v1/p0_exposure_matched_analysis/per_run.csv) · [all features/metrics](../results/entex/v1/p0_exposure_matched_analysis/summary.csv) · [paired AP contrasts](../results/entex/v1/p0_exposure_matched_analysis/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p0_exposure_matched_analysis/complexity_metric_summary.csv)
- **P0 H3K27ac only:** [per-run metrics](../results/entex/v1/p0_h3k27ac_analysis/per_run.csv) · [all features/metrics](../results/entex/v1/p0_h3k27ac_analysis/summary.csv) · [paired AP contrasts](../results/entex/v1/p0_h3k27ac_analysis/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p0_h3k27ac_analysis/complexity_metric_summary.csv)
- **P0 CTCF only:** [per-run metrics](../results/entex/v1/p0_ctcf_analysis/per_run.csv) · [all features/metrics](../results/entex/v1/p0_ctcf_analysis/summary.csv) · [paired AP contrasts](../results/entex/v1/p0_ctcf_analysis/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p0_ctcf_analysis/complexity_metric_summary.csv)
- **P1 thyroid_gland:** [per-run metrics](../results/entex/v1/p1_analysis/thyroid_gland/per_run.csv) · [all features/metrics](../results/entex/v1/p1_analysis/thyroid_gland/summary.csv) · [paired AP contrasts](../results/entex/v1/p1_analysis/thyroid_gland/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p1_analysis/thyroid_gland/complexity_metric_summary.csv)
- **P1 tibial_nerve:** [per-run metrics](../results/entex/v1/p1_analysis/tibial_nerve/per_run.csv) · [all features/metrics](../results/entex/v1/p1_analysis/tibial_nerve/summary.csv) · [paired AP contrasts](../results/entex/v1/p1_analysis/tibial_nerve/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p1_analysis/tibial_nerve/complexity_metric_summary.csv)
- **P1 body_of_pancreas:** [per-run metrics](../results/entex/v1/p1_analysis/body_of_pancreas/per_run.csv) · [all features/metrics](../results/entex/v1/p1_analysis/body_of_pancreas/summary.csv) · [paired AP contrasts](../results/entex/v1/p1_analysis/body_of_pancreas/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p1_analysis/body_of_pancreas/complexity_metric_summary.csv)
- **P1 gastroesophageal_sphincter:** [per-run metrics](../results/entex/v1/p1_analysis/gastroesophageal_sphincter/per_run.csv) · [all features/metrics](../results/entex/v1/p1_analysis/gastroesophageal_sphincter/summary.csv) · [paired AP contrasts](../results/entex/v1/p1_analysis/gastroesophageal_sphincter/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p1_analysis/gastroesophageal_sphincter/complexity_metric_summary.csv)
- **P1 Peyers_patch:** [per-run metrics](../results/entex/v1/p1_analysis/Peyers_patch/per_run.csv) · [all features/metrics](../results/entex/v1/p1_analysis/Peyers_patch/summary.csv) · [paired AP contrasts](../results/entex/v1/p1_analysis/Peyers_patch/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p1_analysis/Peyers_patch/complexity_metric_summary.csv)
- **P2 CTCF:** [per-run metrics](../results/entex/v1/p2_analysis/ctcf/per_run.csv) · [all features/metrics](../results/entex/v1/p2_analysis/ctcf/summary.csv) · [paired AP contrasts](../results/entex/v1/p2_analysis/ctcf/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p2_analysis/ctcf/complexity_metric_summary.csv)
- **P2 H3K27ac:** [per-run metrics](../results/entex/v1/p2_analysis/h3k27ac/per_run.csv) · [all features/metrics](../results/entex/v1/p2_analysis/h3k27ac/summary.csv) · [paired AP contrasts](../results/entex/v1/p2_analysis/h3k27ac/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/v1/p2_analysis/h3k27ac/complexity_metric_summary.csv)
- **RNA ASE:** [per-run metrics](../results/entex/meeting_20260929/rna_analysis/per_run.csv) · [all features/metrics](../results/entex/meeting_20260929/rna_analysis/summary.csv) · [paired AP contrasts](../results/entex/meeting_20260929/rna_analysis/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/meeting_20260929/rna_analysis/complexity_metric_summary.csv)
- **atac:** [per-run metrics](../results/entex/extension_20260927/analysis/atac/per_run.csv) · [all features/metrics](../results/entex/extension_20260927/analysis/atac/summary.csv) · [paired AP contrasts](../results/entex/extension_20260927/analysis/atac/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/extension_20260927/analysis/atac/complexity_metric_summary.csv)
- **h3k4me3:** [per-run metrics](../results/entex/extension_20260927/analysis/h3k4me3/per_run.csv) · [all features/metrics](../results/entex/extension_20260927/analysis/h3k4me3/summary.csv) · [paired AP contrasts](../results/entex/extension_20260927/analysis/h3k4me3/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/extension_20260927/analysis/h3k4me3/complexity_metric_summary.csv)
- **h3k27me3:** [per-run metrics](../results/entex/extension_20260927/analysis/h3k27me3/per_run.csv) · [all features/metrics](../results/entex/extension_20260927/analysis/h3k27me3/summary.csv) · [paired AP contrasts](../results/entex/extension_20260927/analysis/h3k27me3/paired_gains.csv) · [complexity AP/AUROC/normalized AP](../results/entex/extension_20260927/analysis/h3k27me3/complexity_metric_summary.csv)

QC sources: [cCREs_default_AS.tsv.qc.json](../results/entex/v1/qc/cCREs_default_AS.tsv.qc.json) · [hetSNVs_high-confidence_AS.tsv.qc.json](../results/entex/v1/qc/hetSNVs_high-confidence_AS.tsv.qc.json) · [active.combined_set.txt.zip.qc.json](../results/entex/v1/qc/active.combined_set.txt.zip.qc.json) · [repressed.combined_set.txt.zip.qc.json](../results/entex/v1/qc/repressed.combined_set.txt.zip.qc.json) · [preparation.json](../results/entex/v1/qc/p2/preparation.json) · [final_coverage.json](../results/entex/v1/qc/completed/final_coverage.json) · [rna_preparation.json](../results/entex/meeting_20260929/qc/rna_preparation.json) · [audit.json](../results/entex/extension_20260927/qc/audit.json).

### Visual reports

[P0 all features](../results/entex/v1/p0_analysis/auprc_strict.png) · [P0 complexity](../results/entex/v1/p0_analysis/complexity_gain_strict.png) · [CTCF/H3K27ac weighting](../results/entex/meeting_20260929/report/weighting_auprc.pdf) · [Three-assay weighting](../results/entex/extension_20260927/report/weighting_auprc.pdf) · [Extension donor strata](../results/entex/extension_20260927/report/donor_gains.pdf) · [Extension tissue strata](../results/entex/extension_20260927/report/tissue_gains.pdf)

### Provenance and commands

Core configuration: [entex_v1.json](../configs/entex_v1.json). Follow-up protocols: [entex_meeting_20260929.json](../configs/entex_meeting_20260929.json) and [entex_extension_20260927.json](../configs/entex_extension_20260927.json). Core execution receipt pins commit `3475d979943ea800371e27e633ec9f4928adc02d`; per-stage audits record their own code/config identities. Reuse exact graph/checkpoint/cache versions.

```bash
# From the repository root in the original server environment
export PYTHONPATH=src:.
# Core preparation, mapping, probe and analysis commands:
# docs/ENTEX_EXPERIMENTS.md (historical status paragraphs are superseded here)
bash scripts/server/run_entex_extension_20260927.sh prepare
bash scripts/server/run_entex_extension_20260927.sh smoke
bash scripts/server/run_entex_extension_20260927.sh matrix
```

These commands reproduce existing work; they were not rerun in this review. Large graph and embedding inputs are required on the server. The checked compact outputs are sufficient for this report.

## Appendix A. All seven feature combinations

AP is mean [pointwise 95% CI]; other metrics below are means. Their SDs and CIs, precision/recall, calibrated thresholds and per-run sizes are retained in the linked summary/per-run files.

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


## Appendix B. Reciprocal sequence contribution

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

## Appendix C. Per-fold sizes and source distributions

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

These are full-cohort counts; Section 7 reports means across test folds. They are not performance estimates.

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

The preparation-era `p0_qc.json` supplies these counts only; its old “mapping pending” field is superseded by the successful mapping and completion receipts in Section 3.4.

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

## Review checks

[Machine-readable validation receipt](ENTEX_REPORT_AUDIT_20260929.json) records the checked tables, their hashes, and the scope of the arithmetic checks.


- [x] All 450 configuration records and 3,150 feature-specific rows complete.
- [x] Paired AP arithmetic and seven-metric summary means reproduce within 1e-12.
- [x] Source measurement/locus units and registry-resolution history distinguished.
- [x] Null outcomes and all predefined assays/tissues retained.
- [x] Historical pending states distinguished from later completion evidence.
- [x] No new encoder/probe fitting, no new data download, no changed significance thresholds.
- [ ] Learned-versus-random attribution and independent biological confirmation for EN-TEx remain scientific work, not editorial completion.
