#!/usr/bin/env bash
set -euo pipefail

# Resumable batch extraction of exact donor chromosome paths from an HPRC GBZ.
# Run prepare_hprc_chr_paths.py first. One output directory is produced per
# donor; a nonempty .complete marker prevents accidental re-extraction.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${GRAPHGENOMEFM_PUBLIC_DATA_ROOT:-${project_root}/data/public}"
gbz="${GRAPHGENOMEFM_HPRC_GBZ:-${data_root}/hprc/v2.0/hprc-v2.0-mc-grch38.gbz}"
path_dir="${GRAPHGENOMEFM_PATH_DIR:-${data_root}/hprc/v2.0/path_batches}"
chromosome="${GRAPHGENOMEFM_CHROMOSOME:-chr8}"
vg_image="${GRAPHGENOMEFM_VG_IMAGE:-quay.io/vgteam/vg:v1.71.0}"
chunk_size_bp="${GRAPHGENOMEFM_CHUNK_SIZE_BP:-2000000}"
chunk_dir_suffix="${GRAPHGENOMEFM_CHUNK_DIR_SUFFIX:-chunks2m}"
donor_filter=",${GRAPHGENOMEFM_DONORS:-},"

if [[ ! -s "${gbz}" ]]; then
  echo "Missing GBZ: ${gbz}" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then rerun this script." >&2
  exit 1
fi

relative_gbz="${gbz#"${project_root}/"}"
while IFS= read -r paths_file; do
  donor="$(basename "${paths_file}" ".${chromosome}.paths.txt")"
  if [[ "${donor_filter}" != ",," && "${donor_filter}" != *",${donor},"* ]]; then
    continue
  fi
  output_dir="${data_root}/hprc/v2.0/${donor}.${chromosome}.${chunk_dir_suffix}"
  marker="${output_dir}/.complete"
  paths_sha256="$(shasum -a 256 "${paths_file}" | awk '{print $1}')"
  if [[ -s "${marker}" ]]; then
    if grep -q "^chunk_size_bp=${chunk_size_bp}$" "${marker}"; then
      # Reverify against the current path list instead of trusting a stale
      # marker. This catches corrected manifests and changed cohort selections.
      PYTHONPATH="${project_root}/src" \
        /opt/anaconda3/envs/PangenomeFM/bin/python \
        "${project_root}/scripts/verify_hprc_chr_chunks.py" \
          --path-list "${paths_file}" \
          --chunks-dir "${output_dir}" \
          --out "${output_dir}/path_coverage.json" \
          >/dev/null
      count="$(find "${output_dir}" -type f -name '*.gfa.gz' | wc -l | tr -d ' ')"
      expected="$(wc -l < "${paths_file}" | tr -d ' ')"
      printf 'chunk_size_bp=%s\nchunks=%s\npaths=%s\npaths_sha256=%s\n' \
        "${chunk_size_bp}" "${count}" "${expected}" "${paths_sha256}" > "${marker}"
      echo "[present] ${donor} ${chromosome} (${chunk_size_bp} bp chunks)"
      continue
    fi
    echo "Marker ${marker} was made with a different chunk size." >&2
    exit 1
  fi
  mkdir -p "${output_dir}"
  relative_paths="${paths_file#"${project_root}/"}"
  relative_output="${output_dir#"${project_root}/"}"
  echo "[extract] ${donor} ${chromosome}"
  docker run --rm \
    --volume "${project_root}:/work" \
    "${vg_image}" \
    vg chunk \
      --xg-name "/work/${relative_gbz}" \
      --path-list "/work/${relative_paths}" \
      --prefix "/work/${relative_output}/chunk" \
      --chunk-size "${chunk_size_bp}" \
      --output-fmt gfa \
      --context-steps 1 \
      --threads 1
  while IFS= read -r -d '' uncompressed; do
    gzip -f "${uncompressed}"
  done < <(find "${output_dir}" -type f -name '*.gfa' -print0)
  count="$(find "${output_dir}" -type f -name '*.gfa.gz' | wc -l | tr -d ' ')"
  expected="$(wc -l < "${paths_file}" | tr -d ' ')"
  if [[ "${count}" -lt "${expected}" ]]; then
    echo "Extraction incomplete for ${donor}: ${count} chunks, expected >= ${expected}." >&2
    exit 1
  fi
  PYTHONPATH="${project_root}/src" \
    /opt/anaconda3/envs/PangenomeFM/bin/python \
    "${project_root}/scripts/verify_hprc_chr_chunks.py" \
      --path-list "${paths_file}" \
      --chunks-dir "${output_dir}" \
      --out "${output_dir}/path_coverage.json" \
      >/dev/null
  printf 'chunk_size_bp=%s\nchunks=%s\npaths=%s\npaths_sha256=%s\n' \
    "${chunk_size_bp}" "${count}" "${expected}" "${paths_sha256}" > "${marker}"
  echo "[verified] ${donor}: ${count} chunks"
done < <(find "${path_dir}" -maxdepth 1 -type f -name "*.${chromosome}.paths.txt" | sort)
