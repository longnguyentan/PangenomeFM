#!/usr/bin/env bash
set -euo pipefail

# CUDA wheels are installed separately so the user can match them to the
# server driver. Override, for example:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash ...
ENV_NAME="${PANGENOMEFM_ENV_NAME:-pangenomefm-server}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if command -v micromamba >/dev/null 2>&1; then
  ENV_TOOL=micromamba
elif command -v mamba >/dev/null 2>&1; then
  ENV_TOOL=mamba
elif command -v conda >/dev/null 2>&1; then
  ENV_TOOL=conda
else
  echo "ERROR: install micromamba, mamba, or conda first." >&2
  exit 2
fi

if ! "${ENV_TOOL}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${ENV_TOOL}" env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environment-server.yml"
else
  "${ENV_TOOL}" env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environment-server.yml" --prune
fi

"${ENV_TOOL}" run -n "${ENV_NAME}" python -m pip install \
  --index-url "${TORCH_INDEX_URL}" torch
"${ENV_TOOL}" run -n "${ENV_NAME}" python -m pip install \
  -e "${REPO_ROOT}[training,analysis,test]"

"${ENV_TOOL}" run -n "${ENV_NAME}" python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, round(props.total_memory / 2**30, 2), "GiB")
PY

echo
echo "Environment ready. Activate with:"
echo "  ${ENV_TOOL} activate ${ENV_NAME}"
echo "If activation is unavailable in a batch shell, prefix commands with:"
echo "  ${ENV_TOOL} run -n ${ENV_NAME}"
