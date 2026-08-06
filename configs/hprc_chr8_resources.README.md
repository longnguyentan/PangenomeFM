# HPRC chr8 methylation cohort manifest

`hprc_chr8_resources.tsv` is the frozen acquisition manifest for the eight
additional-donor experiment. `hprc_chr8_cohort_metadata.tsv` adds the two
completed pilot donors for a ten-donor modeling cohort.

## Cohort

| Donor | 1KG population | Super-population |
|---|---|---|
| HG00097 | GBR | EUR |
| HG01167 | PUR | AMR |
| HG02040 | KHV | EAS |
| HG02922 | ESN | AFR |
| HG03130 | ESN | AFR |
| HG03742 | ITU | SAS |
| NA18565 | CHB | EAS |
| NA20850 | GIH | SAS |

The full modeling cohort also contains HG00438 (CHS/EAS) and HG002
(GIAB Ashkenazi trio, labeled ASH rather than forcing it into a 1KG
super-population). HG002 has no RepeatMasker/HMMFlagger objects in the public
epigenome sample prefix, so its exact-track annotation fields remain missing
while its graph-derived proxies are retained.

The population labels come from the official 1000 Genomes
`integrated_call_samples_v3.20130502.ALL.panel` table. Donors were intersected
with both the HPRC epigenome sample listing and HPRC v2 GBZ path metadata.

## Exact chromosome-path resolution

`scripts/prepare_hprc_chr_paths.py` selects every donor/haplotype path inside
the `chr8` block delimited by the GBZ metadata's `REFERENCE` records. This is
necessary because a haplotype chromosome can be represented by one `CM`
accession, several WGS contigs, or a mixture of both. Accession-number order is
not a chromosome mapping and is never used. The frozen resolution audit is
`data/public/hprc/v2.0/path_batches/chr8.path_manifest.tsv`.

`scripts/extract_hprc_chr_paths.sh` then verifies complete interval coverage
for the current path list and records its SHA-256 digest in each completion
marker, preventing a stale extraction from being silently reused after a
manifest correction.

## Resource provenance

- Epigenome objects:
  `https://hprc-epigenome.s3.us-east-2.amazonaws.com/samples/<DONOR>/`
- HPRC v2 path metadata:
  `data/public/hprc/v2.0/paths.metadata.tsv`
- Population panel:
  `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel`

Every object has an expected byte size obtained from the public S3 listing.
`scripts/download_public_resources.sh` resumes partial downloads, verifies the
final byte size, and enforces a configurable free-space reserve.

RepeatMasker bigBeds are exact donor/haplotype repeat annotations.
HMMFlagger intervals are assembly-confidence/QC annotations; they are not
presented as direct experimental mappability measurements. Graph-derived
mappability, SV, and copy-number columns are explicitly named as proxies.
