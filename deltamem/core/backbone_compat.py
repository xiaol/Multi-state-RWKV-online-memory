from __future__ import annotations

from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Attention,
    apply_rotary_pos_emb as smollm3_apply_rotary_pos_emb,
    eager_attention_forward as smollm3_eager_attention_forward,
)

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb,
        eager_attention_forward as qwen3_5_eager_attention_forward,
    )
except Exception:  # pragma: no cover - optional Transformers version support
    Qwen3_5Attention = None
    qwen3_5_apply_rotary_pos_emb = None
    qwen3_5_eager_attention_forward = None

try:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextAttention,
        apply_rotary_pos_emb as gemma4_apply_rotary_pos_emb,
        eager_attention_forward as gemma4_eager_attention_forward,
    )
except Exception:  # pragma: no cover - optional Transformers version support
    Gemma4TextAttention = None
    gemma4_apply_rotary_pos_emb = None
    gemma4_eager_attention_forward = None

HAS_SMOLLM3 = True


def ensure_attention_compat_views(module):
    return module
