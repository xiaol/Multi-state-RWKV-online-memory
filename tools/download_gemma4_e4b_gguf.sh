#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-ggml-org/gemma-4-E4B-it-GGUF}
LOCAL_DIR=${LOCAL_DIR:-/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it}
MODEL_FILE=${MODEL_FILE:-gemma-4-E4B-it-Q8_0.gguf}
MMPROJ_FILE=${MMPROJ_FILE:-mmproj-gemma-4-E4B-it-Q8_0.gguf}

mkdir -p "${LOCAL_DIR}"

hf download "${REPO_ID}" \
  "${MODEL_FILE}" \
  "${MMPROJ_FILE}" \
  --local-dir "${LOCAL_DIR}" \
  --max-workers "${HF_MAX_WORKERS:-4}"

hf cache verify "${REPO_ID}" --local-dir "${LOCAL_DIR}" --format json

sha256sum "${LOCAL_DIR}/${MODEL_FILE}" "${LOCAL_DIR}/${MMPROJ_FILE}"
