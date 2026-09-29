# Bibliography verification, 29 September 2026

## Outcome

The external `references.bib` resolves all 52 active citation keys: 41 main references and 11 Supplementary references, representing 45 distinct works. Seven `supp__` aliases intentionally repeat main-list works so that independently numbered reference lists have distinct PDF anchors. No bibliography entries remain inside `main.tex`.

## Google Scholar and citation keys

Google Scholar could not be retrieved during this check. **Exact agreement with Scholar-exported BibTeX keys is therefore not certified.** Citation keys are local labels, not globally assigned identifiers or evidence that a citation is correct. We adopted the author-supplied `macias2026benchmarking` and updated active citations consistently; historical comments retain their original text. Other existing keys remain stable. Each work below was instead checked against its DOI/publisher, conference proceedings, arXiv version or database record.

The supplied `references (1).bib` had only 19 entries, whereas the manuscript requires 45 distinct works. Missing records were restored and verified; keys were not pointed at unrelated references simply to remove a question mark. The JSON audit pins the resulting bibliography hash. `check_bibliography.py` tests key coverage and alias identity; it cannot automatically validate scientific claims.

## Substantive corrections

- Umap/Bismap: corrected journal, volume, issue, article number and DOI.
- GWAS Catalog: corrected five given names.
- HyenaDNA: corrected author order and included all conference authors.
- Nucleotide Transformer: corrected the compound family name Lopez Carranza.
- Minigraph-Cactus and ENCODE: restored credited consortia.
- DeepGene: corrected the journal title and added DOI/issue; HGSVC3 and HeaRT also receive verified identifiers.
- DNABERT-2: corrected the title to “Genomes”; Tang et al. receives article number 203.
- EN-TEx: exact title punctuation restored; all duplicated Supplementary records match their main counterparts.

## Record-by-record review

| Citation key | Work / year | Source | Review |
|---|---|---|---|
| `schneider2017grch38` | Evaluation of GRCh38 and de novo haploid genome assemblies demonstrates the enduring quality of the reference assembly (2017) | [Primary record](https://doi.org/10.1101/gr.213611.116) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `nurk2022complete` | The complete sequence of a human genome (2022) | [Primary record](https://doi.org/10.1126/science.abj6987) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `paten2017genome` | Genome graphs and the evolution of genome inference (2017) | [Primary record](https://doi.org/10.1101/gr.214155.116) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `eizenga2020pangenome` | Pangenome graphs (2020) | [Primary record](https://doi.org/10.1146/annurev-genom-120219-080406) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `garrison2018variation` | Variation graph toolkit improves read mapping by representing genetic variation in the reference (2018) | [Primary record](https://doi.org/10.1038/nbt.4227) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `li2020minigraph` | The design and construction of reference pangenome graphs with minigraph (2020) | [Primary record](https://doi.org/10.1186/s13059-020-02168-z) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `hickey2024pangenome` | Pangenome graph construction from genome alignments with Minigraph-Cactus (2024) | [Primary record](https://doi.org/10.1038/s41587-023-01793-w); [cross-check](https://www.nature.com/articles/s41587-023-01793-w) | Restored HPRC consortium in the publisher author order. The cited issue is 2024; the article first appeared online in 2023. |
| `hprc2023draft` | A draft human pangenome reference (2023) | [Primary record](https://doi.org/10.1038/s41586-023-05896-x) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `hgsvc2021` | Haplotype-resolved diverse human genomes and integrated analysis of structural variation (2021) | [Primary record](https://doi.org/10.1126/science.abf7117) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `hgsvc32025` | Complex genetic variation in nearly complete human genomes (2025) | [Primary record](https://doi.org/10.1038/s41586-025-09140-6) | Added DOI, issue 8076 and named first three authors; not a different HPRC release. |
| `siren2021giraffe` | Pangenomics enables genotyping of known structural variants in 5202 diverse genomes (2021) | [Primary record](https://doi.org/10.1126/science.abg8871) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `ebler2022pangenie` | Pangenome-based genome inference allows efficient and accurate genotyping across a wide spectrum of variant classes (2022) | [Primary record](https://doi.org/10.1038/s41588-022-01043-w) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `macias2026benchmarking` | Benchmarking genome choice in functional genomics analyses (2026) | [Primary record](https://doi.org/10.1038/s41467-026-73663-3) | Renamed the active local key from maciasvelasco2026benchmarking to the user-supplied macias2026benchmarking. DOI, title, authors, 2026, volume 17 and article 6837 match the record. |
| `ji2021dnabert` | DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome (2021) | [Primary record](https://doi.org/10.1093/bioinformatics/btab083) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `zhou2024dnabert2` | DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes (2024) | [Primary record](https://openreview.net/pdf?id=oMLQB4EZE1) | Corrected title ending to multi-species Genomes (plural); uses ICLR 2024, not preprint year 2023. |
| `nguyen2023hyenadna` | HyenaDNA: Long-range genomic sequence modeling at single nucleotide resolution (2023) | [Primary record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract.html) | Corrected conference author order and completed all 13 names; preserved NeurIPS 36 (2023). |
| `dalla2025nucleotide` | Nucleotide Transformer: building and evaluating robust foundation models for human genomics (2025) | [Primary record](https://doi.org/10.1038/s41592-024-02523-z) | Corrected Nicolas Lopez Carranza family name. Uses 2025 issue, not 2024 online publication year. |
| `benegas2023gpn` | DNA language models are powerful predictors of genome-wide variant effects (2023) | [Primary record](https://doi.org/10.1073/pnas.2311219120) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `avsec2021enformer` | Effective gene expression prediction from sequence by integrating long-range interactions (2021) | [Primary record](https://doi.org/10.1038/s41592-021-01252-x) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `tang2025evaluating` | Evaluating the representational power of pre-trained DNA language models for regulatory genomics (2025) | [Primary record](https://doi.org/10.1186/s13059-025-03674-8) | Added article number 203. |
| `zhang2025deepgene` | DeepGene: An efficient foundation model for genomics based on pan-genome graph transformer (2025) | [Primary record](https://doi.org/10.1109/TCBBIO.2025.3614354) | Added DOI and issue; corrected journal to IEEE Transactions on Computational Biology and Bioinformatics (the 2025 title does not include ACM). |
| `xue2025pangenomex` | PangenomeX: a graph convolutional network-based pangenome framework for unbiased population-scale genomic variation analysis (2025) | [Primary record](https://doi.org/10.1093/bib/bbaf550) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `velickovic2018gat` | Graph attention networks (2018) | [Primary record](https://openreview.net/references/pdf?id=HkkE_0ELM) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `kipf2016vgae` | Variational graph auto-encoders (2016) | [Primary record](https://arxiv.org/abs/1611.07308) | Retained 2016 arXiv workshop/preprint record; not presented as a journal article. |
| `hu2020strategies` | Strategies for pre-training graph neural networks (2020) | [Primary record](https://arxiv.org/abs/1905.12265) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `hou2022graphmae` | GraphMAE: Self-Supervised Masked Graph Autoencoders (2022) | [Primary record](https://arxiv.org/abs/2205.10803) | Retained the 2022 arXiv citation; repository record notes acceptance at KDD 2022. |
| `li2023heart` | Evaluating graph neural networks for link prediction: current pitfalls and new benchmarking (2023) | [Primary record](https://doi.org/10.52202/075280-0169); [cross-check](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html) | Added DOI and proceedings pages 3853–3866; cross-checked the NeurIPS conference record. |
| `dong2022fakeedge` | FakeEdge: alleviate dataset shift in link prediction (2022) | [Primary record](https://proceedings.mlr.press/v198/dong22a.html) | PMLR source confirms pages 56:1–56:19, not 56:1–56:23. |
| `geirhos2020shortcut` | Shortcut learning in deep neural networks (2020) | [Primary record](https://doi.org/10.1038/s42256-020-00257-z) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `rozowsky2023entex` | The EN-TEx resource of multi-tissue personal epigenomes & variant-impact models (2023) | [Primary record](https://doi.org/10.1016/j.cell.2023.02.018) | Title uses an ampersand, matching the publication; full journal pagination ends at 1511.e40. |
| `encode2020registry` | Expanded encyclopaedias of DNA elements in the human and mouse genomes (2020) | [Primary record](https://doi.org/10.1038/s41586-020-2493-4); [cross-check](https://pubmed.ncbi.nlm.nih.gov/32728249/) | Restored The ENCODE Project Consortium as first credited author; confirmed named author order with publisher/PubMed because Crossref expands collaborators. |
| `hewitt2019control` | Designing and interpreting probes with control tasks (2019) | [Primary record](https://doi.org/10.18653/v1/D19-1275) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `barabasi1999emergence` | Emergence of scaling in random networks (1999) | [Primary record](https://doi.org/10.1126/science.286.5439.509) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `libennowell2007link` | The link-prediction problem for social networks (2007) | [Primary record](https://doi.org/10.1002/asi.20591) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `benegas2025traitgym` | Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics (2025) | [Primary record](https://doi.org/10.1101/2025.02.11.637758) | Explicitly retained the cited 2025 bioRxiv work; not relabelled as a peer-reviewed publication. |
| `bolognini2026cosigt` | COSIGT: population-scalable genotyping of complex loci from low-coverage sequencing data using pangenome graphs (2026) | [Primary record](https://doi.org/10.1186/s13059-026-04242-4) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `siren2022gbz` | GBZ file format for pangenome graphs (2022) | [Primary record](https://doi.org/10.1093/bioinformatics/btac656) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `su2021roformer` | RoFormer: Enhanced Transformer with Rotary Position Embedding (2021) | [Primary record](https://arxiv.org/abs/2104.09864) | Retained the six-author 2021 arXiv record actually cited, rather than mixing later journal metadata. |
| `lin2017focal` | Focal Loss for Dense Object Detection (2017) | [Primary record](https://doi.org/10.1109/ICCV.2017.324); [cross-check](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) | Retained CVF open-access proceedings pages 2980–2988. Crossref/IEEE pagination is 2999–3007; this source-version difference is explicit, not silently mixed. Added the CVF source link. |
| `loshchilov2019adamw` | Decoupled Weight Decay Regularization (2019) | [Primary record](https://openreview.net/pdf/5963886abef941684ffc0cf670297e47fb1e5155.pdf) | Listed author order, title, publication year, venue and available volume/pages agree with DOI metadata; differences in title capitalization follow bibliography style. |
| `ncbi_hg002` | NIST HG002 NA24385: BioSample SAMN03283347 (2026) | [Primary record](https://www.ncbi.nlm.nih.gov/biosample/SAMN03283347/) | Database record, not a journal article. Year 2026 is the stated access year; supports specimen identity only. |
| `supp__karimzadeh2018umap` | Umap and Bismap: quantifying genome and methylome mappability (2018) | [Primary record](https://doi.org/10.1093/nar/gky677); [cross-check](https://academic.oup.com/nar/article/46/20/e120/5086676) | Corrected erroneous Bioinformatics 34(13), i716–i725 and DOI bty617 to Nucleic Acids Research 46(20), e120, DOI gky677; publisher verifies all fields. |
| `supp__frankish2023gencode` | GENCODE: reference annotation for the human and mouse genomes in 2023 (2023) | [Primary record](https://doi.org/10.1093/nar/gkac1071) | 2023 issue date retained (online 2022). General GENCODE reference, not evidence for the particular v50 file release. |
| `supp__lappalainen2013transcriptome` | Transcriptome and genome sequencing uncovers functional variation in humans (2013) | [Primary record](https://doi.org/10.1038/nature12531) | Named leading authors and bibliographic fields match the article; others abbreviates the remaining authors. |
| `supp__sollis2023gwas` | The NHGRI-EBI GWAS Catalog: knowledgebase and deposition resource (2023) | [Primary record](https://doi.org/10.1093/nar/gkac1010) | Corrected given names: Elliot Sollis, Abayomi Mosaku, Ala Abid, Laurent Gil and Osman Güneş. Title, DOI, year and pages verified. |

## Interpretation safeguards

The references distinguish graph construction/genotyping from representation learning. DeepGene and PangenomeX are discussed as different ways of representing pangenome information, without a first-model claim. EN-TEx is cited for its resource and assay definitions, not as independent confirmation of PangenomeFM performance. GraphMAE, RoFormer and TraitGym retain the explicit versions above. No citation is used to convert a single-fold development result into replicated evidence.

## Recheck

```bash
python check_bibliography.py
python check_editorial_comments.py
python build_manuscript.py
```

The final full-project build is required to check typesetting and resolved citation labels; a metadata audit alone is not a compilation test.
