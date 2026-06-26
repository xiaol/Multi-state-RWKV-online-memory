#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN:-llama-server}
GGUF_MODEL_PATH=${GGUF_MODEL_PATH:-/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf}
GGUF_MMPROJ_PATH=${GGUF_MMPROJ_PATH:-/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/mmproj-gemma-4-E4B-it-Q8_0.gguf}
DEFAULT_RWKV_MS_SIDECAR_PATH=/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/gemma-4-E4B-it-rwkv-ms-memory.gguf
GGUF_RWKV_MS_SIDECAR_PATH=${GGUF_RWKV_MS_SIDECAR_PATH:-}
LLAMA_RWKV_MS=${LLAMA_RWKV_MS:-0}
LLAMA_HOST=${LLAMA_HOST:-127.0.0.1}
LLAMA_PORT=${LLAMA_PORT:-8080}
LLAMA_CTX_SIZE=${LLAMA_CTX_SIZE:-8192}
LLAMA_GPU_LAYERS=${LLAMA_GPU_LAYERS:-999}
LLAMA_MODEL_ALIAS=${LLAMA_MODEL_ALIAS:-}
LLAMA_REASONING=${LLAMA_REASONING:-off}
LLAMA_SLOT_SAVE_PATH=${LLAMA_SLOT_SAVE_PATH:-}

if [[ "${LLAMA_RWKV_MS}" == "1" && -z "${GGUF_RWKV_MS_SIDECAR_PATH}" ]]; then
  GGUF_RWKV_MS_SIDECAR_PATH=${DEFAULT_RWKV_MS_SIDECAR_PATH}
fi

if [[ -n "${GGUF_RWKV_MS_SIDECAR_PATH}" ]]; then
  LLAMA_BATCH_SIZE=${LLAMA_BATCH_SIZE:-2}
  LLAMA_UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-1}
  LLAMA_PARALLEL=${LLAMA_PARALLEL:-1}
  LLAMA_CONT_BATCHING=${LLAMA_CONT_BATCHING:-0}
  LLAMA_CONTEXT_SHIFT=${LLAMA_CONTEXT_SHIFT:-0}
  LLAMA_CACHE_PROMPT=${LLAMA_CACHE_PROMPT:-0}
  LLAMA_CACHE_IDLE_SLOTS=${LLAMA_CACHE_IDLE_SLOTS:-0}
  LLAMA_CACHE_RAM=${LLAMA_CACHE_RAM:-0}
  LLAMA_CTX_CHECKPOINTS=${LLAMA_CTX_CHECKPOINTS:-0}
  LLAMA_TEXT_ONLY=${LLAMA_TEXT_ONLY:-1}
  LLAMA_SLOT_SAVE_PATH=${LLAMA_SLOT_SAVE_PATH:-.openresearch/artifacts/gguf_ui/slots}
else
  LLAMA_BATCH_SIZE=${LLAMA_BATCH_SIZE:-2048}
  LLAMA_UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-512}
  LLAMA_PARALLEL=${LLAMA_PARALLEL:--1}
  LLAMA_CONT_BATCHING=${LLAMA_CONT_BATCHING:-}
  LLAMA_CONTEXT_SHIFT=${LLAMA_CONTEXT_SHIFT:-}
  LLAMA_CACHE_PROMPT=${LLAMA_CACHE_PROMPT:-}
  LLAMA_CACHE_IDLE_SLOTS=${LLAMA_CACHE_IDLE_SLOTS:-}
  LLAMA_CACHE_RAM=${LLAMA_CACHE_RAM:-}
  LLAMA_CTX_CHECKPOINTS=${LLAMA_CTX_CHECKPOINTS:-}
  LLAMA_TEXT_ONLY=${LLAMA_TEXT_ONLY:-0}
fi

if [[ -z "${LLAMA_MODEL_ALIAS}" ]]; then
  if [[ -n "${GGUF_RWKV_MS_SIDECAR_PATH}" ]]; then
    LLAMA_MODEL_ALIAS=gemma-4-e4b-it-rwkv-ms-q8
  else
    LLAMA_MODEL_ALIAS=gemma-4-e4b-it-q8
  fi
fi

if [[ ! -f "${GGUF_MODEL_PATH}" ]]; then
  echo "Missing GGUF model: ${GGUF_MODEL_PATH}" >&2
  exit 1
fi

if [[ -n "${GGUF_RWKV_MS_SIDECAR_PATH}" && ! -f "${GGUF_RWKV_MS_SIDECAR_PATH}" ]]; then
  echo "Missing RWKV-MS GGUF sidecar: ${GGUF_RWKV_MS_SIDECAR_PATH}" >&2
  exit 1
fi

if [[ -n "${GGUF_RWKV_MS_SIDECAR_PATH}" ]]; then
  if [[ "${LLAMA_PARALLEL}" != "1" ]]; then
    echo "RWKV-MS sidecar mode requires LLAMA_PARALLEL=1; got ${LLAMA_PARALLEL}" >&2
    exit 1
  fi
  if [[ "${LLAMA_UBATCH_SIZE}" != "1" ]]; then
    echo "RWKV-MS sidecar mode scans tokens serially; setting LLAMA_UBATCH_SIZE=1 instead of ${LLAMA_UBATCH_SIZE}" >&2
    LLAMA_UBATCH_SIZE=1
  fi
  if (( LLAMA_BATCH_SIZE < 2 )); then
    echo "RWKV-MS sidecar mode requires LLAMA_BATCH_SIZE>=2 for llama.cpp preflight checks; got ${LLAMA_BATCH_SIZE}" >&2
    exit 1
  fi
  if [[ "${LLAMA_CONT_BATCHING}" != "0" || "${LLAMA_CONTEXT_SHIFT}" != "0" || "${LLAMA_CACHE_PROMPT}" != "0" || "${LLAMA_CACHE_IDLE_SLOTS}" != "0" ]]; then
    echo "RWKV-MS sidecar mode requires disabled continuous batching, context shift, prompt cache, and idle-slot cache" >&2
    exit 1
  fi
  if [[ "${LLAMA_CACHE_RAM}" != "0" || "${LLAMA_CTX_CHECKPOINTS}" != "0" ]]; then
    echo "RWKV-MS sidecar mode requires LLAMA_CACHE_RAM=0 and LLAMA_CTX_CHECKPOINTS=0" >&2
    exit 1
  fi
  if [[ "${LLAMA_TEXT_ONLY}" != "1" ]]; then
    echo "RWKV-MS sidecar mode is currently text-only; set LLAMA_TEXT_ONLY=1" >&2
    exit 1
  fi
  for extra_arg in "$@"; do
    case "${extra_arg}" in
      -b|--batch-size|-b=*|--batch-size=*|\
      -ub|--ubatch-size|-ub=*|--ubatch-size=*|\
      -np|--parallel|-np=*|--parallel=*|\
      -ctxcp|--ctx-checkpoints|--swa-checkpoints|-ctxcp=*|--ctx-checkpoints=*|--swa-checkpoints=*|\
      -cram|--cache-ram|-cram=*|--cache-ram=*|\
      --cache-reuse|--cache-reuse=*|\
      --spec-*|--model-draft|--model-draft=*|--mtp|--spec-default|\
      -mm|--mmproj|-mm=*|--mmproj=*|\
      -mmu|--mmproj-url|-mmu=*|--mmproj-url=*|\
      --rwkv-ms-sidecar|--rwkv-ms-sidecar=*|\
      --cont-batching|--no-cont-batching|\
      --context-shift|--no-context-shift|\
      --cache-prompt|--no-cache-prompt|\
      --cache-idle-slots|--no-cache-idle-slots|\
      --mmproj-auto|--no-mmproj|--no-mmproj-auto|\
      --slot-save-path|--slot-save-path=*)
        echo "RWKV-MS sidecar mode does not allow overriding ${extra_arg}; use the launcher environment variables" >&2
        exit 1
        ;;
    esac
  done
fi

if [[ -n "${LLAMA_SLOT_SAVE_PATH}" ]]; then
  mkdir -p "${LLAMA_SLOT_SAVE_PATH}"
fi

args=(
  -m "${GGUF_MODEL_PATH}"
  --alias "${LLAMA_MODEL_ALIAS}"
  --host "${LLAMA_HOST}"
  --port "${LLAMA_PORT}"
  --ctx-size "${LLAMA_CTX_SIZE}"
  --batch-size "${LLAMA_BATCH_SIZE}"
  --ubatch-size "${LLAMA_UBATCH_SIZE}"
  --n-gpu-layers "${LLAMA_GPU_LAYERS}"
  --parallel "${LLAMA_PARALLEL}"
  --jinja
  --reasoning "${LLAMA_REASONING}"
)

if [[ -n "${GGUF_RWKV_MS_SIDECAR_PATH}" ]]; then
  args+=(--rwkv-ms-sidecar "${GGUF_RWKV_MS_SIDECAR_PATH}")
fi

if [[ "${LLAMA_CONT_BATCHING}" == "0" ]]; then
  args+=(--no-cont-batching)
elif [[ "${LLAMA_CONT_BATCHING}" == "1" ]]; then
  args+=(--cont-batching)
fi

if [[ "${LLAMA_CONTEXT_SHIFT}" == "0" ]]; then
  args+=(--no-context-shift)
elif [[ "${LLAMA_CONTEXT_SHIFT}" == "1" ]]; then
  args+=(--context-shift)
fi

if [[ "${LLAMA_CACHE_PROMPT}" == "0" ]]; then
  args+=(--no-cache-prompt)
elif [[ "${LLAMA_CACHE_PROMPT}" == "1" ]]; then
  args+=(--cache-prompt)
fi

if [[ "${LLAMA_CACHE_IDLE_SLOTS}" == "0" ]]; then
  args+=(--no-cache-idle-slots)
elif [[ "${LLAMA_CACHE_IDLE_SLOTS}" == "1" ]]; then
  args+=(--cache-idle-slots)
fi

if [[ -n "${LLAMA_CACHE_RAM}" ]]; then
  args+=(--cache-ram "${LLAMA_CACHE_RAM}")
fi

if [[ -n "${LLAMA_CTX_CHECKPOINTS}" ]]; then
  args+=(--ctx-checkpoints "${LLAMA_CTX_CHECKPOINTS}")
fi

if [[ -n "${LLAMA_REASONING_BUDGET:-}" ]]; then
  args+=(--reasoning-budget "${LLAMA_REASONING_BUDGET}")
fi

if [[ -n "${LLAMA_SLOT_SAVE_PATH}" ]]; then
  args+=(--slot-save-path "${LLAMA_SLOT_SAVE_PATH}")
fi

if [[ "${LLAMA_TEXT_ONLY:-0}" != "1" && -f "${GGUF_MMPROJ_PATH}" ]]; then
  args+=(--mmproj "${GGUF_MMPROJ_PATH}")
fi

exec "${LLAMA_SERVER_BIN}" "${args[@]}" "$@"
