#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/delta-Mem" >&2
  exit 2
fi

SOURCE_REPO="$1"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${PATCH_DIR}/delta_mem_rwkv_ms.patch"
EXPECTED_BASE_REV="5cd5d9153c7f408764728d953565201e198c39e2"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi

if [[ ! -d "${SOURCE_REPO}/.git" ]]; then
  echo "not a git checkout: ${SOURCE_REPO}" >&2
  exit 2
fi

SOURCE_REV="$(git -C "${SOURCE_REPO}" rev-parse HEAD)"
if [[ "${SOURCE_REV}" != "${EXPECTED_BASE_REV}" ]]; then
  echo "delta-Mem revision mismatch: expected ${EXPECTED_BASE_REV}, got ${SOURCE_REV}" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

WORKTREE="${TMP_ROOT}/delta-Mem"
git clone --quiet "${SOURCE_REPO}" "${WORKTREE}"

(
  cd "${WORKTREE}"
  git apply --check "${PATCH_FILE}"
  git apply "${PATCH_FILE}"
  "${PYTHON_BIN}" -m py_compile \
    deltamem/chat_templates.py \
    deltamem/core/hrm_rwkv7.py \
    deltamem/core/delta_impl.py \
    deltamem/core/delta.py \
    deltamem/model_loading.py \
    deltamem/runtime/session.py \
    deltamem/train/delta_sft_experimental.py
  bash -n scripts/run_rwkv_ms_delta_rule_comparison.sh
  cmp deltamem/core/hrm_rwkv7.py "${PATCH_DIR}/hrm_rwkv7.py"
  cmp \
    scripts/run_rwkv_ms_delta_rule_comparison.sh \
    "${PATCH_DIR}/run_rwkv_ms_delta_rule_comparison.sh"
  "${PYTHON_BIN}" -m pytest -q deltamem/tests/test_delta_mem_regressions.py \
    -k 'rwkv_ms or qwen3_5 or gemma4'
  "${PYTHON_BIN}" -m pytest -q deltamem/tests/test_qwen36_runtime_regressions.py
  "${PYTHON_BIN}" -m pytest -q "${PATCH_DIR}/gguf/test_rwkv_ms_math_fixture.py"
  git diff --check
)

echo "delta-Mem RWKV-MS patch applies and focused regressions pass"
