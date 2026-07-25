from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


GGUF_DIR = Path(__file__).resolve().parent
if str(GGUF_DIR) not in sys.path:
    sys.path.insert(0, str(GGUF_DIR))

from export_memory_gguf import export_sidecar  # noqa: E402


def test_export_rejects_content_gated_fusion_before_writing(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "delta_mem_config.json").write_text(
        json.dumps({"memory_fusion_mode": "content_gated_add"}),
        encoding="utf-8",
    )
    output = tmp_path / "adapter.gguf"
    args = SimpleNamespace(
        memory_dir=memory_dir,
        output=output,
        gguf_py_root=tmp_path / "missing-gguf-py",
    )

    with pytest.raises(ValueError, match="content_gated_add.*not implemented"):
        export_sidecar(args)

    assert not output.exists()


@pytest.mark.parametrize(
    "fusion_placement",
    ["post_attention_norm", "normalized_residual_correction"],
)
def test_export_rejects_norm_hook_fusion_before_writing(
    tmp_path: Path,
    fusion_placement: str,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "delta_mem_config.json").write_text(
        json.dumps(
            {
                "memory_fusion_mode": "add",
                "memory_fusion_placement": fusion_placement,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "adapter.gguf"
    args = SimpleNamespace(
        memory_dir=memory_dir,
        output=output,
        gguf_py_root=tmp_path / "missing-gguf-py",
    )

    with pytest.raises(ValueError, match=rf"{fusion_placement}.*not implemented"):
        export_sidecar(args)

    assert not output.exists()
