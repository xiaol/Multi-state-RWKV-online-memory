#!/usr/bin/env python3
"""Run the final evaluator against a newly sealed split and adapter."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as routed_common
from experiments.rethinking_rwkv_ms_gemma import run_recurrent_routed_final as final


SPLIT_ROOT = Path(__file__).resolve().parent / "local_artifacts/recurrent_routed_query_value_distill_blend50_fresh_split_v1"
FINAL_ROOT = Path(__file__).resolve().parent / "local_artifacts/recurrent_routed_query_value_distill_blend50_fresh_final_v1"
DEVELOPMENT_RESULT = Path(__file__).resolve().parent / "local_artifacts/recurrent_routed_query_value_distill_blend50_development_v2/result.json"
CANDIDATE_ADAPTER = Path(__file__).resolve().parent / "local_artifacts/recurrent_routed_query_value_distill_blend50_v2/adapter"


def main() -> int:
    final.FINAL_ROOT = FINAL_ROOT
    final.FINAL_OPENING_RECEIPT = "b86f1a558891147cef990a6c5de4125a265526a0479f6e08facca33c9bec0889"
    final.SPLIT_MANIFEST_RECEIPT = "ca0c5a6653dd17b2ba861e3d54718a11840f490974220d4834be8b807ffe5861"
    final.FINAL_COMMITMENT_RECEIPT = "4789c253bbe5ae6ed133ddc2763aa391113937bd12bc06b0b92149cfb51860c9"
    final.OPEN_SPLIT_RECEIPT = "bb92eeb62e2bedce2fe5e695b73d9697fd913ee7ac0205ce98d859121a51ef7c"
    final.PROTOCOL_FILE = Path(__file__).resolve().parent / "natural_memory_native_recurrent_routed_adapter_blend_protocol_v1.json"
    final.PROTOCOL_RECEIPT = "e88e756e051719229289ed73f4d494b36885f04882bbb325a3695f7287faf556"
    final.TRAINING_ROOT = Path(__file__).resolve().parent / "local_artifacts/recurrent_routed_query_value_distill_blend50_v2"
    final.TRAINING_RESULT_RECEIPT = "75fc307f2c5db320c0cbca4f7bc92c0829db5a842cf0f052a6b536d3b0016206"
    final.DEVELOPMENT_RESULT = DEVELOPMENT_RESULT
    final.DEVELOPMENT_RESULT_RECEIPT = "224e7e86ecdf9d3673542e182e6420600a7cbec3daed1c4308fc6dcca33af015"
    final.CANDIDATE_ADAPTER = CANDIDATE_ADAPTER
    routed_common.SPLIT_ROOT = SPLIT_ROOT
    return final.main()


if __name__ == "__main__":
    raise SystemExit(main())
