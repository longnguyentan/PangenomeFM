# Running The New HPRC Workflow

The new repo convention is intentionally simple. A cleaned dataset is a folder
with graph tables:

```text
data/
  hprc/
    full_segments.csv
    full_links.csv
  encode/
    GRCh38-human-cCREs.bed
```

The HPRC files must have these columns:

- `full_segments.csv`: `id`, `name`, `seq`, `LN`, `SN`, `SO`, `SR`
- `full_links.csv`: `from_seg`, `from_orient`, `to_seg`, `to_orient`, `overlap`

## 1. Install

```bash
pip install -e ".[training,test]"
```

If PyTorch is already managed separately on a server, `pip install -e ".[test]"`
is enough for packaging and lightweight checks.

```bash
pip install -e ".[test]"
```

## 2. Validate HPRC Data

```bash
python -m graphgenomefm check-data --data-dir data/hprc
```

This checks that `full_segments.csv` and `full_links.csv` exist and have the
expected columns.

## 3. Build Benchmark Windows

Small smoke run:

```bash
python -m graphgenomefm make-benchmark \
  --data-dir data/hprc \
  --out-dir data/hprc/benchmark_smoke \
  --targets chr21 chr22 \
  --n-windows 1 \
  --no-network-analysis \
  --no-viz
```

Main HPRC benchmark:

```bash
python -m graphgenomefm make-benchmark \
  --data-dir data/hprc \
  --out-dir data/hprc/benchmark \
  --targets all \
  --n-windows 10 \
  --window-bp 50000 \
  --no-network-analysis \
  --no-viz
```

The CLI accepts short chromosomes such as `chr22` and expands them to HPRC SN
values such as `GRCh38#0#chr22` when needed.

## 4. Run Shared Pretraining

Smoke run:

```bash
python -m graphgenomefm pretrain \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark_smoke \
  --out-dir results/hprc/pretrain_smoke \
  --test-chrs chr22 \
  --epochs 2 \
  --patience 1 \
  --device cpu
```

Main run:

```bash
python -m graphgenomefm pretrain \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark \
  --out-dir results/hprc/pretrain \
  --val-chrs chr16 \
  --test-chrs chr1 chr8 chr19 chrY \
  --epochs 100 \
  --patience 20 \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on a CUDA server.

## 5. Map ENCODE cCRE Labels

```bash
python -m graphgenomefm label-ccre \
  --data-dir data/hprc \
  --encode-bed data/encode/GRCh38-human-cCREs.bed \
  --out-dir data/hprc/ccre
```

This creates a versioned run directory such as
`data/hprc/ccre/run_001/node_labels.csv.gz`.

## 6. Run cCRE Baseline

Binary cCRE/non-cCRE baseline for Task v1:

```bash
python -m graphgenomefm ccre-binary-baseline \
  --data-dir data/hprc \
  --test-chrs chr8 chr19 chr22 \
  --val-chr chr16 \
  --out-dir results/hprc/ccre_binary_baseline
```

Multiclass cCRE baseline:

```bash
python -m graphgenomefm ccre-baseline \
  --data-dir data/hprc \
  --test-chrs chr8 chr19 chr22 \
  --val-chr chr16 \
  --out-dir results/hprc/ccre_baseline
```

The new wrapper defaults to disjoint validation and test chromosomes.

## 7. Run cCRE GAT

```bash
python -m graphgenomefm ccre-gat \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --out-dir results/hprc/ccre_gat \
  --epochs 60 \
  --patience 15 \
  --device cpu
```

Frozen pretrained encoder pilot:

```bash
python -m graphgenomefm ccre-gat \
  --data-dir data/hprc \
  --benchmark-dir data/hprc/benchmark \
  --test-chrs chr8 chr19 chr22 \
  --val-chrs chr16 \
  --out-dir results/hprc/ccre_gat_pretrained_frozen \
  --pretrained-checkpoint results/hprc/pretrain/run_001/ckpt_strict_...pt \
  --freeze-backbone \
  --keep-is-grch38 \
  --epochs 60 \
  --patience 15 \
  --device cpu
```

Fine-tuned pretrained encoder is the same command without `--freeze-backbone`.

## 8. Add A New Dataset Later

After manual cleaning, make the same folder shape:

```text
data/hgsvc3/
  full_segments.csv.gz
  full_links.csv.gz
```

Then run the same commands with `--data-dir data/hgsvc3`.

For raw GFA input, first create those files:

```bash
python -m graphgenomefm parse-gfa \
  --gfa /path/to/external_graph.gfa.gz \
  --out-dir data/hgsvc3
```

For a first HGSVC3 chr22 pilot from a large GFA, parse only the target reference
walk plus direct neighbors:

```bash
python -m graphgenomefm parse-gfa \
  --gfa data/hgsvc3/hgsvc3-2024-02-23-mc-chm13.sv.gfa.gz \
  --out-dir data/hgsvc3 \
  --target-sns chr22 \
  --target-prefix 'id=CHM13|' \
  --include-link-neighbors

python -m graphgenomefm make-benchmark \
  --data-dir data/hgsvc3 \
  --out-dir data/hgsvc3/benchmark_chr22 \
  --targets chr22 \
  --target-prefix 'id=CHM13|' \
  --n-windows 10 \
  --window-bp 50000 \
  --no-network-analysis \
  --no-viz

python -m graphgenomefm eval-external \
  --data-dir data/hgsvc3 \
  --benchmark-dir data/hgsvc3/benchmark_chr22 \
  --checkpoint results/hprc/pretrain/run_001/ckpt_strict_...pt \
  --out-dir results/hgsvc3/external_eval_chr22_strict \
  --split all \
  --device cpu
```
