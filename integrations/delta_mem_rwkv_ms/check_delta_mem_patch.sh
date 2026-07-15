#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [/path/to/clean/delta-Mem]" >&2
  exit 2
fi

SOURCE_REPO="${1:-}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PATCH_DIR}/../.." && pwd)"
PATCH_FILE="${PATCH_DIR}/delta_mem_rwkv_ms.patch"
EXPECTED_BASE_REV="5cd5d9153c7f408764728d953565201e198c39e2"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m py_compile \
  "${REPO_ROOT}/deltamem/chat_templates.py" \
  "${REPO_ROOT}/deltamem/core/hrm_rwkv7.py" \
  "${REPO_ROOT}/deltamem/core/delta_impl.py" \
  "${REPO_ROOT}/deltamem/core/delta.py" \
  "${REPO_ROOT}/deltamem/model_loading.py" \
  "${REPO_ROOT}/deltamem/runtime/session.py" \
  "${REPO_ROOT}/deltamem/train/delta_sft_experimental.py"
bash -n "${PATCH_DIR}/run_rwkv_ms_delta_rule_comparison.sh"
cmp "${REPO_ROOT}/deltamem/core/hrm_rwkv7.py" "${PATCH_DIR}/hrm_rwkv7.py"
"${PYTHON_BIN}" -m pytest -q "${REPO_ROOT}/deltamem/tests/test_delta_mem_regressions.py" \
  -k 'rwkv_ms or qwen3_5 or gemma4'
"${PYTHON_BIN}" -m pytest -q "${REPO_ROOT}/deltamem/tests/test_qwen36_runtime_regressions.py"
"${PYTHON_BIN}" -m pytest -q "${PATCH_DIR}/gguf/test_rwkv_ms_math_fixture.py"

if [[ -n "${SOURCE_REPO}" ]]; then
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
    diff -qr \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      deltamem \
      "${REPO_ROOT}/deltamem"
    cmp \
      scripts/run_rwkv_ms_delta_rule_comparison.sh \
      "${PATCH_DIR}/run_rwkv_ms_delta_rule_comparison.sh"
    git diff --check
  )
  echo "optional upstream patch reproduces the bundled runtime"
fi

git -C "${REPO_ROOT}" diff --check
echo "bundled Qwen/Gemma RWKV-MS runtime and regressions pass"
