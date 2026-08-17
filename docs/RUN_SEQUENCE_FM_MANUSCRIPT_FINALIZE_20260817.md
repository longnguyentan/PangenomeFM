# Final sequence-FM manuscript evidence: one-command server run

This run does **not** retrain PangenomeFM, rerun the 60 GPU probes, or rebuild
the frozen sequence cache. It performs the remaining evidence-finalization work:

1. focused code tests;
2. concurrent cCRE and SV aggregation with 10,000 fold-then-seed bootstrap draws;
3. the direct topology-given-coordinate-and-frozen-sequence-FM contrast;
4. paired 1-hop/strict downstream context interactions;
5. SV stratum power and identifiability safeguards;
6. a strengthened named-contrast decision audit;
7. manuscript CSV, LaTeX, Markdown, PDF/PNG, and notebook outputs;
8. provenance capture, checksums, and a compact `.tar.zst` package.

The run is restart-safe. A completed aggregate is skipped. An incomplete output
directory is moved to a timestamped preservation path before retrying.

## Expected time and resources

| Stage | Estimate | Resource | Can overlap? |
|---|---:|---|---|
| Focused tests | 1--3 min | CPU | first |
| cCRE aggregation | 1--5 min | CPU | yes, with SV |
| SV aggregation and strata | 2--10 min | CPU | yes, with cCRE |
| Decision audit | 1--3 min | CPU/RAM | after both aggregates |
| Tables, figure, notebook | 1--3 min | CPU | after audit |
| Provenance and package | 2--10 min | CPU/I/O | last |

Expected total wall time is approximately **10--30 minutes**. No GPU is
required. The audit briefly loads the existing 303,425-node sequence cache, so
several GiB of available RAM are desirable; the server has ample capacity.

## The one command

Run this once from the server. It fetches the exact reviewed implementation,
requires the server checkout to be a clean fast-forward ancestor of it, and
then launches the restart-safe workflow in `tmux`:

```bash
cd ~/PangenomeFM &&
test -z "$(git status --porcelain)" &&
git fetch origin codex/canonical-complexity-v2-20260815 &&
git merge --ff-only FETCH_HEAD &&
bash scripts/server/launch_sequence_fm_manuscript_finalize.sh
```

If the clean-worktree test stops the command, inspect `git status --short` and
preserve the server changes before continuing. Do not bypass that guard by
discarding files.

The launcher starts this session:

```text
pangenomefm-sequence-fm-finalize-20260817
```

It immediately prints the exact attach, log, and status commands.

## Monitor without attaching

```bash
tail -f \
  ~/PangenomeFM/server_workspace/results/manuscript_sequence_fm_finalize_20260817/console.log
```

Or attach:

```bash
tmux attach -t pangenomefm-sequence-fm-finalize-20260817
```

Detach with `Ctrl-b`, then `d`.

The run is complete only when the log contains:

```text
STRENGTHENED_MANUSCRIPT_DECISION_GATE_PASSED
SEQUENCE_FM_MANUSCRIPT_FINALIZATION_COMPLETE
FINALIZE_EXIT_CODE=0
```

## Final status

```bash
cat \
  ~/PangenomeFM/server_workspace/results/manuscript_sequence_fm_finalize_20260817/final_status.json
```

Expected:

```json
{
  "status": "complete"
}
```

The file also records the exact git commit, aggregate directories, package
path, byte size, and package SHA-256.

## Scientific output locations

Versioned cCRE aggregation:

```text
server_workspace/results/ccre_sequence_fm_factorial_20260815/
  paper_source_data_v2_20260817/
```

Versioned SV aggregation:

```text
server_workspace/results/hgsvc3_sv_sequence_fm_factorial_20260815/
  paper_source_data_v2_20260817/
```

Strengthened final audit:

```text
server_workspace/results/
  manuscript_sequence_fm_decision_audit_v2_20260817.json
```

Group-discussion and manuscript artifacts:

```text
server_workspace/results/manuscript_sequence_fm_outputs_20260817/
  README.md
  audit.json
  manuscript_sequence_fm_analysis.ipynb
  manuscript_sequence_fm_key_contrasts_v2.csv
  manuscript_sequence_fm_primary_auprc.csv
  manuscript_sequence_fm_context_interactions.csv
  manuscript_sequence_fm_absolute_performance.csv
  manuscript_sequence_fm_primary_table.tex
  figure_sequence_topology_survival.png
  figure_sequence_topology_survival.pdf
```

Standalone compact package:

```text
server_workspace/results/
  manuscript_sequence_fm_evidence_v2_20260817.tar.zst
  manuscript_sequence_fm_evidence_v2_20260817.tar.zst.sha256
```

The package excludes NPZ caches and model checkpoints. The authoritative caches
and raw probe outputs remain on the server; the package contains the evidence
tables, audits, figures, notebook, and provenance needed for manuscript work.

## Download to the local Mac

The working route is the same single-jump `rsync` pattern previously used for
`cis-chen`; there is no need to SSH separately to the internal Tailscale address.

```bash
LOCAL_DIR="/Users/longnguyentan/Downloads/Temple Uni/PangenomeFM/server_imports/sequence_fm_v2_20260817"
mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR"

rsync -aH --partial --inplace --progress \
  -e 'ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o ProxyCommand="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -W %h:%p tuv43532@cis-linux2.temple.edu"' \
  tuv43532@cis-chen:/home/tuv43532/PangenomeFM/server_workspace/results/manuscript_sequence_fm_evidence_v2_20260817.tar.zst \
  .

rsync -aH --partial --inplace --progress \
  -e 'ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o ProxyCommand="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -W %h:%p tuv43532@cis-linux2.temple.edu"' \
  tuv43532@cis-chen:/home/tuv43532/PangenomeFM/server_workspace/results/manuscript_sequence_fm_evidence_v2_20260817.tar.zst.sha256 \
  .
```

Validate and extract:

```bash
shasum -a 256 -c \
  manuscript_sequence_fm_evidence_v2_20260817.tar.zst.sha256

zstd -t manuscript_sequence_fm_evidence_v2_20260817.tar.zst

mkdir -p extracted
zstd -dc manuscript_sequence_fm_evidence_v2_20260817.tar.zst | \
  tar -xf - -C extracted
```

## Restart after interruption

Run the same one command again:

```bash
cd ~/PangenomeFM &&
bash scripts/server/launch_sequence_fm_manuscript_finalize.sh
```

If the tmux session still exists but the job has ended, inspect it first:

```bash
tmux capture-pane \
  -p \
  -t pangenomefm-sequence-fm-finalize-20260817 \
  -S -200 | tail -n 200
```

If the pane shows `FINALIZE_EXIT_CODE` and a shell prompt, remove only the tmux
session—not any result directory—and relaunch:

```bash
tmux kill-session -t pangenomefm-sequence-fm-finalize-20260817
cd ~/PangenomeFM && \
bash scripts/server/launch_sequence_fm_manuscript_finalize.sh
```

## Interpretation boundary

The primary result is the paired AUPRC gain for
`topology_given_coordinate_and_frozen_sequence_fm`. Positive values support
complementary topology information beyond coordinate and the pinned frozen
sequence model. This remains chromosome-held-out evidence. It is not
donor-held-out representation pretraining, causality, colocalization,
fine-mapping, or proof of a general-purpose biological foundation model.
