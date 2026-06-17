#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/delta-Mem" >&2
  exit 2
fi

SOURCE_REPO="$1"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${PATCH_DIR}/delta_mem_rwkv_ms.patch"

if [[ ! -d "${SOURCE_REPO}/.git" ]]; then
  echo "not a git checkout: ${SOURCE_REPO}" >&2
  exit 2
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

WORKTREE="${TMP_ROOT}/delta-Mem"
git clone --quiet "${SOURCE_REPO}" "${WORKTREE}"

(
  cd "${WORKTREE}"
  git apply --unidiff-zero --whitespace=nowarn "${PATCH_FILE}"
  python -m py_compile \
    deltamem/core/hrm_rwkv7.py \
    deltamem/core/delta_impl.py \
    deltamem/core/delta.py \
    deltamem/train/delta_sft_experimental.py
  bash -n scripts/run_rwkv_ms_delta_rule_comparison.sh
  git diff --check
)

echo "delta-Mem RWKV-MS patch applies and syntax checks pass"
