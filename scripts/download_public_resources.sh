#!/usr/bin/env bash
set -euo pipefail

# Resumable, size-checked acquisition for the public-data phases of the PSB
# project. Large raw assay collections are deliberately opt-in. The default
# "pilot" profile is the smallest set that supports a real donor-matched
# haplotype methylation experiment.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${GRAPHGENOMEFM_PUBLIC_DATA_ROOT:-${project_root}/data/public}"
reserve_gib="${GRAPHGENOMEFM_DISK_RESERVE_GIB:-15}"
profile="${1:-pilot}"
resource_manifest="${GRAPHGENOMEFM_HPRC_RESOURCE_MANIFEST:-${project_root}/configs/hprc_chr8_resources.tsv}"

mkdir -p "${data_root}"

file_size() {
  if stat -f%z "$1" >/dev/null 2>&1; then
    stat -f%z "$1"
  else
    stat -c%s "$1"
  fi
}

available_bytes() {
  df -Pk "${data_root}" | awk 'NR==2 {print $4 * 1024}'
}

require_space() {
  local expected="$1"
  local available reserve
  available="$(available_bytes)"
  reserve=$((reserve_gib * 1024 * 1024 * 1024))
  if (( available - expected < reserve )); then
    echo "Refusing download: ${expected} bytes would leave less than ${reserve_gib} GiB free." >&2
    return 1
  fi
}

download_one() {
  local url="$1"
  local expected="$2"
  local destination="$3"
  local part="${destination}.part"
  local current=0

  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]]; then
    current="$(file_size "${destination}")"
    if [[ "${current}" == "${expected}" ]]; then
      echo "[present] ${destination} (${current} bytes)"
      return 0
    fi
    echo "Existing final file has wrong size: ${destination} (${current}, expected ${expected})" >&2
    return 1
  fi
  if [[ -f "${part}" ]]; then
    current="$(file_size "${part}")"
  fi
  require_space $((expected - current))
  echo "[download] ${url}"
  local attempt
  for attempt in {1..20}; do
    if curl \
      --fail \
      --location \
      --silent \
      --show-error \
      --retry 5 \
      --retry-all-errors \
      --retry-delay 3 \
      --continue-at - \
      --output "${part}" \
      "${url}"; then
      break
    fi
    if [[ "${attempt}" == "20" ]]; then
      echo "Download failed after ${attempt} resumable attempts: ${url}" >&2
      return 1
    fi
    echo "[resume] ${destination} after transient transfer failure (${attempt}/20)"
    sleep 3
  done
  current="$(file_size "${part}")"
  if [[ "${current}" != "${expected}" ]]; then
    echo "Size check failed for ${part}: ${current} bytes, expected ${expected}" >&2
    return 1
  fi
  mv "${part}" "${destination}"
  echo "[verified] ${destination} (${current} bytes)"
}

download_hprc_v1_graph() {
  download_one \
    "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.1-mc-grch38/hprc-v1.1-mc-grch38.gbz" \
    3289083600 \
    "${data_root}/hprc/v1.1/hprc-v1.1-mc-grch38.gbz"
}

download_hprc_v2_graph() {
  download_one \
    "https://human-pangenomics.s3.amazonaws.com/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gbz" \
    5419804360 \
    "${data_root}/hprc/v2.0/hprc-v2.0-mc-grch38.gbz"
}

download_hg002_methylation() {
  local base="https://hprc-epigenome.s3.us-east-2.amazonaws.com/samples/HG002"
  local out="${data_root}/hprc_epigenome/HG002"
  download_one \
    "${base}/methylation.PacBio.hap1.methylc.gz" \
    283129634 \
    "${out}/methylation.PacBio.hap1.methylc.gz"
  download_one \
    "${base}/methylation.PacBio.hap1.methylc.gz.tbi" \
    1729184 \
    "${out}/methylation.PacBio.hap1.methylc.gz.tbi"
  download_one \
    "${base}/methylation.PacBio.hap2.methylc.gz" \
    286434094 \
    "${out}/methylation.PacBio.hap2.methylc.gz"
  download_one \
    "${base}/methylation.PacBio.hap2.methylc.gz.tbi" \
    1785806 \
    "${out}/methylation.PacBio.hap2.methylc.gz.tbi"
}

download_hg00438_methylation() {
  local base="https://hprc-epigenome.s3.us-east-2.amazonaws.com/samples/HG00438"
  local out="${data_root}/hprc_epigenome/HG00438"
  download_one \
    "${base}/methylation.PacBio.hap1.methylc.gz" \
    287536517 \
    "${out}/methylation.PacBio.hap1.methylc.gz"
  download_one \
    "${base}/methylation.PacBio.hap1.methylc.gz.tbi" \
    1804804 \
    "${out}/methylation.PacBio.hap1.methylc.gz.tbi"
  download_one \
    "${base}/methylation.PacBio.hap2.methylc.gz" \
    285081146 \
    "${out}/methylation.PacBio.hap2.methylc.gz"
  download_one \
    "${base}/methylation.PacBio.hap2.methylc.gz.tbi" \
    1800473 \
    "${out}/methylation.PacBio.hap2.methylc.gz.tbi"
}

download_hg00438_expression() {
  local base="https://hprc-epigenome.s3.us-east-2.amazonaws.com/samples/HG00438"
  local out="${data_root}/hprc_epigenome/HG00438"
  download_one \
    "${base}/expression.minus.hap1.specific.bw" \
    9233789 \
    "${out}/expression.minus.hap1.specific.bw"
  download_one \
    "${base}/expression.plus.hap1.specific.bw" \
    9517625 \
    "${out}/expression.plus.hap1.specific.bw"
  download_one \
    "${base}/expression.minus.hap2.specific.bw" \
    1975238 \
    "${out}/expression.minus.hap2.specific.bw"
  download_one \
    "${base}/expression.plus.hap2.specific.bw" \
    2088691 \
    "${out}/expression.plus.hap2.specific.bw"
}

download_hprc_cohort_resources() {
  local resource_filter="${1:-all}"
  if [[ ! -s "${resource_manifest}" ]]; then
    echo "Missing HPRC resource manifest: ${resource_manifest}" >&2
    return 1
  fi
  while IFS=$'\t' read -r donor population super_population resource remote_name expected; do
    if [[ "${donor}" == "donor_id" ]]; then
      continue
    fi
    if [[ "${resource_filter}" != "all" && "${resource}" != "${resource_filter}" ]]; then
      continue
    fi
    download_one \
      "https://hprc-epigenome.s3.us-east-2.amazonaws.com/samples/${donor}/${remote_name}" \
      "${expected}" \
      "${data_root}/hprc_epigenome/${donor}/${remote_name}"
  done < "${resource_manifest}"
}

case "${profile}" in
  graph)
    download_hprc_v2_graph
    ;;
  graph-v1.1)
    download_hprc_v1_graph
    ;;
  hg002-methylation)
    download_hg002_methylation
    ;;
  hg00438-methylation)
    download_hg00438_methylation
    ;;
  hg00438-expression)
    download_hg00438_expression
    ;;
  methylation-cohort)
    download_hprc_cohort_resources methylation
    ;;
  annotation-cohort)
    download_hprc_cohort_resources repeat
    download_hprc_cohort_resources mappability
    ;;
  chr8-cohort)
    download_hprc_cohort_resources all
    ;;
  pilot)
    download_hprc_v2_graph
    download_hg002_methylation
    download_hg00438_methylation
    download_hg00438_expression
    ;;
  *)
    echo "Usage: $0 {pilot|graph|graph-v1.1|hg002-methylation|hg00438-methylation|hg00438-expression|methylation-cohort|annotation-cohort|chr8-cohort}" >&2
    exit 2
    ;;
esac

echo "[done] profile=${profile}; data_root=${data_root}"
