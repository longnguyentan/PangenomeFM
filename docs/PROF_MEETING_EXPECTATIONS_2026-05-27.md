# Prof Meeting Expectations and Next Steps

Meeting date: 2026-05-27  
Project: GraphGenome-FM / PSB 2026 draft

## Executive Takeaway

Prof is positive about the current HPRC and HGSVC transfer results, but the
paper needs a clearer story and stronger comparison frame. The main manuscript
should be built around HPRC graph pretraining, HGSVC replication/transfer, and
carefully framed cCRE downstream prediction. The Chinese pangenome graph should
not be a main result for now.

## Prof's Main Expectations

### 1. Revise the abstract

Current issue:
- The abstract has too many exact numbers and experiment details.
- It reads like a results log rather than a high-level paper summary.

Required change:
- Remove most per-run statistics from the abstract.
- Keep only the top-level claims:
  - graph-native pretraining learns pangenome topology;
  - HPRC chromosome-held-out generalization is strong;
  - HGSVC transfer supports external replication;
  - cCRE is a downstream application/stress test still being improved.
- Put detailed AUROC/AUPRC/chromosome numbers in the Results section.

### 2. Do not focus on the Chinese pangenome graph

Prof's reasoning:
- The Chinese pangenome graph appears unusual and not organized like the HPRC
  or HGSVC pangenome graphs.
- It may hide or omit important path/walk information.
- It is risky as a central replication dataset.

Decision:
- Do not use it as a main PSB result.
- Mention it only in Discussion/Future Work as an external resource to inspect
  later if needed.

Main external replication set:
- HPRC as the primary training/internal validation graph.
- HGSVC/HGSVC3 as the primary independent external graph.

### 3. Add a new HGSVC imputation/enhancement application

Prof suggested a separate application section:
- Train or use the HPRC graph foundation model.
- Apply it to older HGSVC pangenome samples/graphs.
- Use the model to impute or propose missing graph nodes/links.
- Validate against a newer pangenome build if Tomoya can run the newer HPRC-like
  pangenome pipeline on one or two HGSVC samples.

Paper framing:
- This can be a new Results subsection if there is time.
- Otherwise include it as a proposed application / pilot study.

Suggested manuscript section:
- `4.4 HGSVC Pangenome Imputation and Graph Enhancement`

Minimum pilot deliverable:
- Pick 1-2 HGSVC samples or chromosomes.
- Use HPRC-trained model to score candidate missing links or graph extensions.
- Compare with a newer pangenome construction, if available.
- Report whether model-imputed links/nodes recover new graph structure.

Open people/questions:
- Ask on Monday call who generated/maintains the HGSVC pangenome graph.
- Possible contacts mentioned: Peter Ebert, Tobias, HPRC pangenome group, Aaron.

### 4. cCRE needs external comparative methods

Current issue:
- The cCRE section compares mostly our methods and our internal baselines.
- Prof wants comparison against methods that represent alternative modeling
  assumptions.

Required comparisons:
- Linear-reference / no-graph baseline.
- DeepGene or a similar linearized-pangenome/linearized-graph model, if feasible.
- Existing logistic regression baseline can stay, but it is not enough by itself.

Minimum acceptable cCRE comparison table:
- Linear-reference-only model.
- Simple graph/coordinate feature logistic model.
- Scratch GAT.
- Frozen pretrained GraphGenome-FM encoder.
- Fine-tuned GraphGenome-FM encoder.
- DeepGene/linearized graph baseline if runnable or reproducible enough.

Important caveat:
- Current GAT cCRE evaluation uses benchmark-window nodes, while logistic uses
  all labeled GRCh38 nodes. This comparison is not fully fair yet.
- Align evaluation before making strong downstream claims.

### 5. Reorganize cCRE around categories and confidence/meta information

Prof and Tomoya discussed that ENCODE cCRE labels may not have a single simple
confidence score, but the underlying SCREEN/ENCODE tracks may contain Z scores
or signal scores.

Action items:
- Inspect ENCODE cCRE/SCREEN metadata.
- Check for confidence, Z score, or signal columns tied to:
  - DNase accessibility;
  - CTCF;
  - H3K4me3;
  - H3K27ac;
  - promoter/enhancer classes.
- If available, stratify cCRE results by confidence/signal level.
- If no direct confidence score exists, reorganize by cCRE category/type.

Scientific hypothesis to test:
- Graph models may help more in complex or structurally variable regulatory
  contexts.
- Small/simple cCRE classes or regions with little sequence diversity may show
  little graph-model advantage.

Reduced-category options:
- Binary: cCRE vs background.
- Reduced multiclass:
  - promoter-like: PLS, CA-H3K4me3;
  - enhancer-like: pELS, dELS;
  - CTCF/TF-associated: CA-CTCF, CA-TF, TF;
  - open chromatin/CA;
  - background.
- Full 9-class task should remain as a harder stress test, not the only result.

### 6. Add future/potential application: haplotype-specific functional annotation

Prof wants this idea captured so it is not lost.

Concept:
- Current ENCODE cCRE labels are built largely from functional genomics reads
  aligned to a linear reference and then summarized/merged.
- A graph foundation model could instead help map or infer functional elements
  on haplotype-specific graph paths.

Possible inputs:
- ENCODE functional genomics data.
- HGSVC/HPRC/4DN functional data.
- ATAC/DNase, methylation, histone modification, CTCF, Hi-C/3D genome tracks.

Potential output:
- Haplotype-specific functional element annotation.
- Functional quantification along pangenome paths.
- Allele/haplotype-specific regulatory differences.

Paper framing:
- Put this in Discussion/Future Work unless a pilot can be run quickly.
- Suggested manuscript section:
  - `4.5 Haplotype-Specific Functional Annotation`

### 7. Add figures and tables, even as drafts

Prof explicitly asked for draft figures/tables to show the ideas.

Required figure/table concepts:
- Model schematic: graph slice, strict closure, 1-hop closure, dual-stream GAT.
- HPRC held-out performance table/plot.
- Negative sampling audit: random vs distance-matched negatives.
- HGSVC external transfer per chromosome.
- cCRE binary ROC/PR curves.
- cCRE multiclass confusion matrix or class-wise F1.
- cCRE category breakdown table.
- Comparative study table: linear/no graph vs DeepGene/linearized graph vs our graph model.
- HGSVC imputation application diagram/table.
- Optional embedding visualization: PCA/UMAP colored by chromosome, graph closure,
  cCRE class, or graph topology.

### 8. Separate ablation studies from comparative studies

Prof requested clearer organization in Methods/Results.

Suggested Results structure:

1. HPRC chromosome-held-out graph pretraining.
2. Ablation and robustness studies.
   - random vs distance-matched negatives;
   - strict vs 1-hop closure;
   - maybe model component ablations if available.
3. External replication on HGSVC.
4. Comparative downstream cCRE study.
   - linear/no graph;
   - DeepGene/linearized graph if possible;
   - scratch GAT;
   - pretrained/fine-tuned graph model.
5. Application pilot: HGSVC imputation/enhancement.
6. Future application: haplotype-specific functional annotation.

## Concrete Next Steps

### Immediate manuscript edits

- Shorten abstract and remove detailed numbers.
- Move all AUROC/AUPRC/chromosome details into Results tables.
- Add subsections for:
  - ablation/robustness studies;
  - comparative cCRE studies;
  - HGSVC imputation;
  - haplotype-specific functional annotation.
- Add placeholder figure/table captions.
- Keep cCRE GAT claims cautious.

### Immediate experiments

1. Keep HPRC + HGSVC as the main paper datasets.
2. Do not spend major time on the Chinese pangenome graph.
3. Finish/fix fair binary cCRE GAT:
   - align GAT and logistic evaluation nodes;
   - report per-chromosome AUROC/AUPRC/macro-F1/balanced accuracy.
4. Build cCRE category/reduced-class labels.
5. Inspect ENCODE/SCREEN metadata for confidence/signal/Z-score fields.
6. Identify a runnable linear-reference baseline.
7. Investigate DeepGene/linearized graph comparison feasibility.
8. Sketch HGSVC imputation pilot:
   - candidate HGSVC sample/chromosome;
   - candidate missing edge scoring;
   - validation target from newer pipeline if Tomoya can generate it.

### Questions for Prof/Tomoya

- Which author list should exactly match the application?
- Should the Chinese pangenome be completely omitted or mentioned briefly in
  Discussion/Future Work?
- Who is the best HGSVC pangenome graph contact?
- Can Tomoya generate a newer pangenome build for 1-2 HGSVC samples?
- Which DeepGene setup/checkpoint should be used for comparison?
- Which cCRE grouping is most biologically meaningful for the first reduced
  class experiment?
- If Pittsburgh clusters are used, what exact acknowledgment wording is needed?

## Priority Order

### P0: Paper story and safe claims

- Short abstract.
- HPRC and HGSVC as core evidence.
- cCRE framed as promising but unfinished.
- No overclaiming about multiclass cCRE GAT.

### P1: Experiments required for a stronger PSB draft

- Fair binary cCRE GAT vs all-node/logistic baseline.
- cCRE category/reduced-class task.
- Linear/no-graph baseline.
- DeepGene/linearized graph baseline if feasible.
- Publication-ready figures/tables.

### P2: New application/pilot

- HGSVC imputation/enhancement pilot.
- Haplotype-specific functional annotation as future work or pilot.

### P3: Later/future

- Chinese pangenome graph inspection.
- Full haplotype-specific functional data mapping.
- Multi-dataset co-training and leave-one-dataset-out experiments.
