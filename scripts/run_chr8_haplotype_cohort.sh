#!/usr/bin/env bash
set -euo pipefail

# Resumable end-to-end runner after public downloads and GBZ extraction finish.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GRAPHGENOMEFM_PYTHON:-/opt/anaconda3/envs/PangenomeFM/bin/python}"
data_root="${GRAPHGENOMEFM_PUBLIC_DATA_ROOT:-${project_root}/data/public}"
result_root="${GRAPHGENOMEFM_RESULT_ROOT:-${project_root}/results/hprc/haplotype_methylation_chr8_cohort_v2}"
cohort="${project_root}/configs/hprc_chr8_cohort_metadata.tsv"
checkpoint="${GRAPHGENOMEFM_CHECKPOINT:-${project_root}/results/hprc/pretrain_masked_train_graph/run_001/ckpt_strict__shared_dual_graph_mscale3_orient_adpwk32a4_focal2.0_dedge0.1_maskedq_heldout_chr1_chr8_chr19_chrY_val_chr16_ep100_pat20.pt}"
export PYTHONPATH="${project_root}/src"
export HF_MODULES_CACHE="${project_root}/tmp/hf_modules"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
donor_filter=",${GRAPHGENOMEFM_DONORS:-},"
donor_stages_only="${GRAPHGENOMEFM_DONOR_STAGES_ONLY:-0}"

mkdir -p "${result_root}"
mapfile_cmd=()
while IFS=$'\t' read -r donor _; do
  [[ "${donor}" == "donor_id" ]] && continue
  if [[ "${donor_filter}" != ",," && "${donor_filter}" != *",${donor},"* ]]; then
    continue
  fi
  mapfile_cmd+=("${donor}")
done < "${cohort}"

pair_paths=()
graph_embedding_paths=()
sequence_embedding_paths=()
for donor in "${mapfile_cmd[@]}"; do
  donor_out="${result_root}/donors/${donor}"
  pair_path="${donor_out}/paired_windows.csv.gz"
  pair_paths+=("${pair_path}")
  if [[ ! -s "${pair_path}" ]]; then
    gfa_files=("${data_root}/hprc/v2.0/${donor}.chr8.chunks2m/"chunk*.gfa.gz)
    "${python_bin}" -m tasks.haplotype.methylation build \
      --gfa "${gfa_files[@]}" \
      --methylation \
        "${data_root}/hprc_epigenome/${donor}/methylation.PacBio.hap1.methylc.gz" \
        "${data_root}/hprc_epigenome/${donor}/methylation.PacBio.hap2.methylc.gz" \
      --out-dir "${donor_out}" \
      --sample "${donor}" \
      --path-list "${data_root}/hprc/v2.0/path_batches/${donor}.chr8.paths.txt"
  fi
  graph_embedding="${donor_out}/graphgenomefm_embeddings.npz"
  graph_embedding_paths+=("${graph_embedding}")
  if [[ ! -s "${graph_embedding}" ]]; then
    "${python_bin}" -m tasks.haplotype.embeddings graph \
      --pairs "${pair_path}" \
      --chunks-dir "${data_root}/hprc/v2.0/${donor}.chr8.chunks2m" \
      --checkpoint "${checkpoint}" \
      --path-list "${data_root}/hprc/v2.0/path_batches/${donor}.chr8.paths.txt" \
      --out "${graph_embedding}"
  fi
  sequence_embedding="${donor_out}/nucleotide_transformer_embeddings.npz"
  sequence_embedding_paths+=("${sequence_embedding}")
  if [[ ! -s "${sequence_embedding}" ]]; then
    "${python_bin}" -m tasks.haplotype.embeddings sequence \
      --pairs "${pair_path}" \
      --out "${sequence_embedding}" \
      --batch-size 16
  fi
done

if [[ "${donor_stages_only}" == "1" ]]; then
  echo "[complete] donor-level pairs and embeddings: ${mapfile_cmd[*]}"
  exit 0
fi

"${python_bin}" "${project_root}/scripts/audit_hprc_chr8_cohort.py" \
  --cohort "${cohort}" \
  --path-manifest "${data_root}/hprc/v2.0/path_batches/chr8.path_manifest.tsv" \
  --path-batch-dir "${data_root}/hprc/v2.0/path_batches" \
  --chunks-root "${data_root}/hprc/v2.0" \
  --result-root "${result_root}" \
  --out-tsv "${result_root}/donor_artifact_audit.tsv" \
  --out-json "${result_root}/donor_artifact_audit.json" \
  --require-shared-node-anchors

combined="${result_root}/paired_windows_annotated.csv.gz"
if [[ ! -s "${combined}" ]]; then
  "${python_bin}" "${project_root}/scripts/assemble_hprc_chr8_cohort.py" \
    --pairs "${pair_paths[@]}" \
    --cohort "${cohort}" \
    --epigenome-root "${data_root}/hprc_epigenome" \
    --out "${combined}" \
    --allow-missing-annotations
fi

sequence_embeddings="${result_root}/nucleotide_transformer_embeddings.npz"
if [[ ! -s "${sequence_embeddings}" ]]; then
  "${python_bin}" -m tasks.haplotype.embeddings merge \
    --inputs "${sequence_embedding_paths[@]}" \
    --out "${sequence_embeddings}"
fi

merged_graph_embeddings="${result_root}/graphgenomefm_embeddings.npz"
if [[ ! -s "${merged_graph_embeddings}" ]]; then
  "${python_bin}" -m tasks.haplotype.embeddings merge \
    --inputs "${graph_embedding_paths[@]}" \
    --out "${merged_graph_embeddings}"
fi

for split in donor population; do
  out="${result_root}/hierarchical_${split}"
  if [[ ! -s "${out}/summary.json" ]]; then
    "${python_bin}" -m tasks.haplotype.hierarchical_residual \
      --pairs "${combined}" \
      --sequence-embeddings "${sequence_embeddings}" \
      --graph-embeddings "${merged_graph_embeddings}" \
      --cohort "${cohort}" \
      --out-dir "${out}" \
      --split-mode "${split}"
  fi
done

"${python_bin}" "${project_root}/scripts/evaluate_functional_modality_gate.py" \
  --summary "${result_root}/hierarchical_donor/summary.json" \
  --out "${result_root}/functional_modality_gate.json"
