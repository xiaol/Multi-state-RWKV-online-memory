#!/usr/bin/env python3
"""Run a narrow, state-isolating Novel Agent scene-boundary evaluation.

The general Novel Agent evaluator feeds the complete prompt to both ordinary
attention and RWKV-MS. This focused evaluator additionally separates the two:
it can prime RWKV-MS from ``[system, user]``, discard the ordinary KV cache,
then generate from a system-only prompt while preserving only online state.

An explicit row selection is required so this command cannot accidentally run
the complete benchmark.
"""

from __future__ import annotations

import argparse
import copy
import gc
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import (  # noqa: E402
    collect_rwkv_trace,
    iter_delta_modules,
    load_model_and_tokenizer,
    logits_to_keep_kwargs,
    memory_condition,
    reset_delta_state,
)
from deltamem.chat_templates import apply_chat_template  # noqa: E402
from analyze_novel_agent_eval import recover_scene, strict_gold_boundaries  # noqa: E402
from run_novel_agent_eval import (  # noqa: E402
    NORMAL_FUSION_PROFILES,
    SCENE_V6_IDENTITY_OBJECTIVE_VERSION,
    append_record,
    apply_normal_fusion_profile,
    extract_json,
    git_revision,
    memory_architecture_contract,
    normal_fusion_fingerprint_fields,
    scene_v6_training_lineage,
    score_prediction,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_atomic,
)


TASK_NAME = "scene-v4-current"
DEFAULT_MAX_NEW_TOKENS = 128
MAX_SELECTED_ROWS = 32
SHA256_RE = re.compile(r"[0-9a-f]{64}")
HISTORICAL_V6_HARD32_CONTRACT = "scene_historical_v6_hard32_three_condition"
HISTORICAL_V6_HARD32_CONDITIONS = (
    "base_full",
    "no_write_full",
    "normal_full",
)
HISTORICAL_V6_HARD32_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_historical_v6_hard32_screen.v1"
)
HISTORICAL_V6_HARD32_AUTHORIZATION_SCOPE = (
    "diagnostic_fixed_hard32_only_no_full170_no_test"
)
SCENE_V8_HARD32_CONTRACT = "scene_v8_authorized_hard32"
SCENE_V8_TRAIN32_AUTHORIZATION_KIND = "scene_memory_v8_train32_gate_receipt"
SCENE_V8_HARD32_AUTHORIZATION_SCOPE = (
    "fixed_scene_v4_current_hard32_only_no_full170_no_test_no_other_benchmarks"
)
SCENE_V14_HARD32_CONTRACT = "scene_v14_authorized_hard32"
SCENE_V14_VALUE14_DIAGNOSTIC_RECEIPT_KIND = (
    "scene_memory_v14_value14_gate_receipt"
)
# Historical import name; this receipt is diagnostic under the candidate-lock policy.
SCENE_V14_VALUE14_AUTHORIZATION_KIND = (
    SCENE_V14_VALUE14_DIAGNOSTIC_RECEIPT_KIND
)
SCENE_V14_CANDIDATE_LOCK_SCHEMA = (
    "rwkv_ms_scene_memory_v14_hard32_candidate_lock.v1"
)
SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND = (
    "scene_memory_v14_benchmark_first_candidate_lock"
)
SCENE_V14_CANDIDATE_SELECTION_POLICY = (
    "benchmark_first_single_production_checkpoint3_v1"
)
SCENE_V14_EVALUATOR_BRIDGE_REASON = (
    "legacy_value14_14_of_14_gate_failed_but_checkpoint3_was_preselected_as_"
    "the_single_benchmark_first_candidate_v1"
)
SCENE_V14_CANDIDATE_LOCK_PATH = (
    SCRIPT_DIR / "scene_memory_v14_checkpoint3_hard32_candidate_lock.json"
)
# The lock binds historical evidence; live evaluator code binds this exact lock.
SCENE_V14_CANDIDATE_LOCK_FILE_SHA256 = (
    "83f1eebf5f9e615f10c0d41a6d29745326e3f4f054107897c060c6ebf6290a24"
)
SCENE_V14_HARD32_AUTHORIZATION_SCOPE = SCENE_V8_HARD32_AUTHORIZATION_SCOPE
SCENE_V15_HARD32_CONTRACT = "scene_v15_authorized_hard32"
SCENE_V15_CANDIDATE_LOCK_SCHEMA = (
    "rwkv_ms_scene_memory_v15_hard32_candidate_lock.v1"
)
SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND = (
    "scene_memory_v15_train32_selected_checkpoint_lock"
)
SCENE_V15_CANDIDATE_SELECTION_POLICY = (
    "correct_state_exact_then_own_paired_identity_then_semantic_nll_v1"
)
SCENE_V15_HARD32_AUTHORIZATION_SCOPE = SCENE_V8_HARD32_AUTHORIZATION_SCOPE
SCENE_V15_HARD32_CONSUMPTION_MARKER_SCHEMA = (
    "rwkv_ms_scene_memory_v15_hard32_authorization_consumed.v1"
)
SCENE_V15_HARD32_CONSUMPTION_MARKER_FILENAME = (
    "hard32_authorization_consumed.json"
)
SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT = (
    "scene_hard_failure_curriculum_train32_overfit_v1"
)
SCENE_HARD_FAILURE_HARD32_CONTRACT = (
    "scene_hard_failure_curriculum_hard32_v1"
)
# Stable import names used by the focused gate and experiment launchers.
SCENE_FOCUSED_TRAIN_OVERFIT_CONTRACT = SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT
SCENE_FOCUSED_HARD32_CONTRACT = SCENE_HARD_FAILURE_HARD32_CONTRACT
SCENE_HARD_FAILURE_TRAIN_SELECTION_SCHEMA = (
    "rwkv_ms_scene_hard_failure_train_overfit_selection.v2"
)
SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA = (
    "rwkv_ms_scene_hard_failure_hard32_authorization_consumed.v1"
)
SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_FILENAME = (
    "hard32_authorization_consumed.json"
)
SCENE_HARD_FAILURE_SOURCE_SCHEMA = "rwkv_ms_scene_memory_v7_source.v1"
SCENE_HARD_FAILURE_PAIR_SCHEMA = "rwkv_ms_scene_memory_v7_pairing.v1"
SCENE_HARD_FAILURE_ROW_SCHEMA = "rwkv_ms_scene_memory_v7_row.v1"
SCENE_HARD_FAILURE_SOURCE_PURPOSE = (
    "train_only_frozen_base_failure_benchmark_boost_v1"
)
SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256 = (
    "a3b1e0a255f2e7440971e81337d9648a1dbf96da8a1211aa3e103f2d01f052d8"
)
SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256 = (
    "8326a1e6c79552f72ccfe372614ee80375673d1956f498b64a254f82fd953caf"
)
SCENE_HARD_FAILURE_TRAIN_FILE_SHA256 = (
    "254e4fe5c2c8e8e9f107ef72a5e08abb48c1c86a446f75e3b265034de1ada8df"
)
SCENE_HARD_FAILURE_ROW_MANIFEST_FILE_SHA256 = (
    "d12dff5cc2d8d038b8a04aef002cb0dc3b8dc047f726d63eb063f9aba66200df"
)
SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256 = (
    "61a4597ef1e014d20b0aece500117d4068e39e36fa320868f7758c506f30f78f"
)
SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256 = (
    "af568b19542541944250e663d73fdd1b5794d2e214fc5ff4bf26b26ea147e35e"
)
SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256 = (
    "8e43ae9b5902f8d6b8b277bf61611d3f05de871852b288a197a79ed6ab627d33"
)
SCENE_HARD_FAILURE_ROWS = 32
SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE = (
    "single_selected_checkpoint_fixed_scene_v4_current_hard32_only_no_full170_"
    "no_test_no_other_benchmarks"
)
HISTORICAL_V6_LINEAGE_LIMITATION = (
    "This historical training run predates commit-bound source locks and atomic "
    "checkpoint receipts. Exact adapter, config, and trainer-state hashes prove "
    "the screened artifact identity, but cannot retrospectively prove its training "
    "source commit or a receipt-bound data lineage."
)
HISTORICAL_V6_CHECKPOINT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_outputs/"
    "novel_rwkv_ms_memory/"
    "v6_semantics2_all_l0_23_r4_n32_ceonly_gain02_lr1e3_e21_cont640/"
    "trainer/checkpoint-672"
)
HISTORICAL_V6_CHECKPOINT_ARTIFACT_SHA256 = {
    "delta_mem_adapter.pt": (
        "79bbe9c300be0eb894bd10808e9f1f08783b4f96a29713fa9d262b4a89786de6"
    ),
    "delta_mem_config.json": (
        "da66ada98c4fc0f83ad36c0e749f5a9b52f0f88f2b516d004027f58052182288"
    ),
    "trainer_state.json": (
        "de158d601787d7913fe89a8d7e46cf73a7d33e433d7dfca3791a3d9268c93d53"
    ),
}
HISTORICAL_V6_EXPECTED_GLOBAL_STEP = 672
HISTORICAL_V6_EXPECTED_LAYER_COUNT = 24
HISTORICAL_V6_BASE_MODEL = Path(
    "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"
)
HISTORICAL_V6_HARD32_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "pairs_candidate64_failure32_holdout32_v1"
)
HISTORICAL_V6_HARD32_HOLDOUT = HISTORICAL_V6_HARD32_ROOT / "holdout.jsonl"
HISTORICAL_V6_HARD32_SELECTION = (
    HISTORICAL_V6_HARD32_ROOT / "holdout_source_indices.json"
)
HISTORICAL_V6_OFFICIAL_VAL = Path(
    "/run/media/xiaol/B214449214445C0B/datasets/novel-agent-sft-dataset/"
    "training/v4-scene-boundary-detection/val.jsonl"
)
DONOR_RULE_CYCLIC = "next_selected_row_cyclic"
DONOR_RULE_LENGTH_MATCHED = "length_matched_label_distinct_symmetric_pair_v1"
DONOR_RULE_LENGTH_MATCHED_LEGACY = "write_token_length_matched_label_distinct_pairs_v1"
DONOR_RULES = (
    DONOR_RULE_CYCLIC,
    DONOR_RULE_LENGTH_MATCHED,
    DONOR_RULE_LENGTH_MATCHED_LEGACY,
)
HARD32_FROZEN_DONOR_PAIRS = (
    (3, 112),
    (6, 33),
    (16, 141),
    (21, 88),
    (24, 47),
    (30, 102),
    (50, 56),
    (59, 70),
    (63, 67),
    (64, 71),
    (66, 74),
    (75, 79),
    (87, 128),
    (113, 166),
    (132, 151),
    (144, 159),
)
HARD32_FROZEN_DONOR_PAIRS_SHA256 = (
    "e772e8c77210537234df4b584b7bf5f762a228362d56eb644baffd33d16c9aea"
)
HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256 = (
    "a531552ef876479a7462fe290dc61f50168fb01926be47727d177337ad13b0cf"
)
HARD32_FROZEN_DONOR_MAX_WRITE_TOKEN_DIFFERENCE = 85
HARD32_FROZEN_DONOR_TOTAL_WRITE_TOKEN_DIFFERENCE = 329
HARD32_FROZEN_DONOR_STRATUM_ROWS = {
    "empty_vs_nonempty": 18,
    "nonempty_same_cardinality": 10,
    "nonempty_different_cardinality": 4,
}
EVALUATION_CONTRACTS = (
    "generic",
    SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
    SCENE_HARD_FAILURE_HARD32_CONTRACT,
    "scene_v6_identity_hard32",
    SCENE_V8_HARD32_CONTRACT,
    SCENE_V14_HARD32_CONTRACT,
    SCENE_V15_HARD32_CONTRACT,
    "scene_v6_matched_donor_validation",
    HISTORICAL_V6_HARD32_CONTRACT,
)
OFFICIAL_SCENE_V4_DATASET_REVISION = "5d3040d21f51b3ce90b9396b058e552c47f43cd5"
OFFICIAL_SCENE_V4_VAL_SHA256 = (
    "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513"
)
HARD32_SELECTION_SHA256 = (
    "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
)
HARD32_HOLDOUT_SHA256 = (
    "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
)
HARD32_PAIR_MANIFEST_SHA256 = (
    "2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008"
)
HARD32_SELECTION_ROWS = (
    (3, "5de02e1f6aa05d091486b282f9ae85fab5301c9c87459f3dbc0b780bccb96c8b"),
    (6, "d2e1d7d1d571a506c95413b733ea5eb7e9a4efd0f94351bf0c961935da32623a"),
    (16, "d0cbbb052b1aaed870e992d64d8663a5c0003ecdf15dd6ada7dfa3aa09de4b2d"),
    (21, "45d7d6fad2d731390e94f7cef2fd07cf00e91743fdbd81418546e94ba77e1d5c"),
    (24, "a20e204d020d1e77c8f6efbeb578e55e8d826159044aa1db77cda7d0aea97edb"),
    (30, "af0aff9cfdbc5a6ddc78b4796ad4051539084f80572140f0aab7adea7cb0d52b"),
    (33, "57e71c02ef3e87a2699f702cb20187cf2d75334d083b57fd6c04f77a2ce23c38"),
    (47, "e5a99a9efee3571a51ca6d0141ae11fa5678a1e28c631ac239a6d4f27382aa1a"),
    (50, "5c19227cfe9d1c5d08ebf029c692b3edc20897dcfeea432a8edcaf088d16aa25"),
    (56, "8c856ddd10d3f2172a7d4b87a8ab653c4b31d78ff3f29b91b299450e110a1375"),
    (59, "b7c66d441b9e099457db1ef234d5bf5c2143872ab9766d01047d131ac9c27607"),
    (63, "f577abdeda7a398bb0043d6245c6ed34ef068570b7a62593a343a7b02f61ff2f"),
    (64, "56c6e98eba35df777f49b0c0d9cb76955f11befc49994743c3a06bd47912e762"),
    (66, "d285b2d837229ab4aba0d0795075f44eaf37d702587d493a868ec532504e94ad"),
    (67, "e020382cd06c8a62b2b2a105a5bddecc40fe43d11af9b5c71c25ad4fc646b083"),
    (70, "a760a7ab7fb02e16f7b6e293dde68aeb94c61e43ecb852f4c4596e70c11b90dd"),
    (71, "32da8986e58527c502be761152f038923951b13f02c63fd4f2fc7d246a67722e"),
    (74, "377ea83402cf31097fd5ff316aab118e09cf49a09fab77f75acff2a30716450c"),
    (75, "18e0b168182eb8e5efb1147e8b423e88dea46edac810227e298710bb36ef5ff6"),
    (79, "ff141c61249985a15e5425a0e6bc658be1cf41be1839c719aa0692e7e9391ef0"),
    (87, "02e641cffdbc06e1b983c06cada6b53ac0ca27f800f0ddadeed0815505bdc435"),
    (88, "814cf6252344ff9c8dce0bdcf11b1f327f2bc4ec872b4aad69a9aba5d816e73d"),
    (102, "3a1fc080ba16f3b6b324973ba46c62aed71d6db992b97b981a2df7be6528002b"),
    (112, "099103e40eb72fa971cad85b94abe2b67e6af5d561a3cd5e8e59549f21e1108e"),
    (113, "172605e20c3e53eb31be94a43d9c384cbd1ccaadbc15db6f05055874703bba75"),
    (128, "8478233a8ccd1dc48909a00f4c6508582a5b15ad773bc74bf32241c590c2324d"),
    (132, "adbf744b7aad1be317dafd5c9c4304520daf68d3ddd1ae9e8752079587e660b9"),
    (141, "1cf24fad6e1cbfffa8106e530e600db0537e3ae128d60e388b2409837b6ffc49"),
    (144, "7ff2c5ed3cd0d4411e49676ac362136e00b9760167b71ef13a8b116c5c747c68"),
    (151, "8fd285743493bbae329d247045d22e33f048eeb6c0a91c1d67762e2bf81909c5"),
    (159, "3d92d3c96e2b59e59751e7716eb2e381abb63327c2feb49ff87df999fa4de847"),
    (166, "19a8a96479602d60943702d2c97c13e2975b1b1b85523e175a2eba58b62c6f8d"),
)
HARD32_ROW_INDICES = tuple(index for index, _ in HARD32_SELECTION_ROWS)
HARD32_ROW_HASHES = dict(HARD32_SELECTION_ROWS)
SEMANTIC_DECISION_MASK_MODE = "top_level_boundaries_json_decision_tokens_v1"
SEMANTIC_DECISION_NLL_NORMALIZATION = "per_row_selected_target_token_mean_v1"
PAIR_TARGET_DECISION_MASK_MODE = (
    "first_pair_distinguishing_boundaries_semantic_token_v1"
)
PAIR_TARGET_DECISION_NLL_NORMALIZATION = "single_selected_target_token_v1"
BENCHMARK_SCENE_METRIC_NAME = "benchmark_strict_boundaries_micro_f1"
HARD32_RECEIPT_SCHEMA = "scene_v6_identity_hard32_receipt.v2"
CHECKPOINT_RECEIPT_SCHEMA = "rwkv_ms_scene_v6_identity_checkpoint.v1"
SCENE_V7_TRAIN32_AUTHORIZATION_KIND = (
    "scene_v7_train32_strict_generation_receipt"
)
SCENE_V6_IDENTITY_OBJECTIVE_INTERPRETATION = {
    "objective_version": SCENE_V6_IDENTITY_OBJECTIVE_VERSION,
    "donor_training_role": "live_symmetric_per_row_hinge",
    "zero_training_role": "diagnostic_only_no_objective_gradient",
    "zero_hard32_role": "required_correct_state_causal_control",
}
SCENE_V7_HARD32_OBJECTIVE_INTERPRETATION = {
    "objective_version": "scene_state_generation_ce_v1",
    "training_objective": (
        "weighted_generation_plus_first_error_pair_identity_zero_causal"
    ),
    "donor_training_role": "two_token_source_donor_identity_classification",
    "zero_training_role": "detached_decision_margin_hinge",
    "hard32_role": "fixed_confirmatory_evaluation_only",
}
SCENE_V8_HARD32_OBJECTIVE_INTERPRETATION = {
    "objective_version": (
        "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
    ),
    "training_objective": (
        "weighted_generation_plus_pair_identity_zero_causal_plus_"
        "edit_aligned_generated_prefix_unlikelihood"
    ),
    "train32_gate": (
        "value14_canonical_generation_and_first_pair_distinguishing_token_identity"
    ),
    "hard32_role": "fixed_confirmatory_evaluation_only",
    "full170_authorized": False,
    "test_authorized": False,
    "other_benchmarks_authorized": False,
}
SCENE_V14_HARD32_OBJECTIVE_INTERPRETATION = {
    "objective_version": (
        "scene_state_generation_ce_symmetric_cached_prefix_boundary_v14"
    ),
    "training_objective": (
        "symmetric_cached_actual_prefix_boundary_repair_and_cached_gold_prefix_"
        "decision_retention"
    ),
    "value14_diagnostic": (
        "exact_correct_and_reciprocal_donor_semantic_generation_with_"
        "row_invariant_zero_state_control"
    ),
    "candidate_selection": SCENE_V14_CANDIDATE_SELECTION_POLICY,
    "candidate_selection_role": (
        "one_predeclared_checkpoint_for_frozen_hard32_benchmark_first_test"
    ),
    "raw_token_exact_role": "telemetry_only",
    "hard32_role": "fixed_confirmatory_evaluation_only",
    "full170_authorized": False,
    "test_authorized": False,
    "other_benchmarks_authorized": False,
}
SCENE_V15_HARD32_OBJECTIVE_INTERPRETATION = {
    "objective_version": (
        "scene_state_generation_ce_symmetric_cached_prefix_identity_v15"
    ),
    "training_objective": (
        "cached_prefix_semantic_boundary_repair_plus_pair_identity_margin_"
        "with_full_answer_ce_weight_zero"
    ),
    "checkpoint_selection": SCENE_V15_CANDIDATE_SELECTION_POLICY,
    "checkpoint_selection_role": (
        "post_save_train32_only_lexicographic_selection_before_frozen_hard32"
    ),
    "raw_token_exact_role": "telemetry_only",
    "hard32_role": "fixed_confirmatory_evaluation_only",
    "full170_authorized": False,
    "test_authorized": False,
    "other_benchmarks_authorized": False,
}
HARD32_GATE_REQUIREMENTS = {
    "semantic_advantage_positive_rows": 20,
    "same_cardinality_nonempty_rows": 10,
    "same_cardinality_nonempty_positive_rows": 8,
    "state_minus_donor_f1": 0.05,
    "state_minus_zero_f1": 0.05,
    "normal_minus_strongest_control_f1": 0.05,
    "max_predicted_to_gold_boundary_ratio": 2.0,
    "state_true_positives": 8,
    "empty_list_exact": 6,
    "empty_list_rows": 9,
    "recovered_outputs": 31,
    "canonical_outputs": 31,
}
SCENE_V15_HARD32_GATE_POLICY = {
    "metric_name": BENCHMARK_SCENE_METRIC_NAME,
    "operator": ">",
    "threshold": 0.0,
    "pass_iff": (
        "state_only_strict_f1_gt_state_only_donor_strict_f1_and_"
        "state_only_strict_f1_gt_state_only_no_write_strict_f1"
    ),
}
CONDITIONS = (
    "base_full",
    "normal_full",
    "no_write_full",
    "state_only",
    "state_only_donor",
    "state_only_no_write",
)
SCENE_FOCUSED_CONDITIONS = (
    "base_full",
    "no_write_full",
    "normal_full",
    "state_only",
    "state_only_donor",
    "state_only_shuffled",
    "state_only_no_write",
)
SUPPORTED_CONDITIONS = (*CONDITIONS, "state_only_shuffled")
SCENE_V15_HARD32_CONDITIONS = (
    "state_only",
    "state_only_donor",
    "state_only_no_write",
)
CONDITION_PROTOCOLS: dict[str, dict[str, Any]] = {
    "base_full": {
        "adapter": False,
        "attention_context": "system_and_user",
        "rwkv_prime": None,
        "rwkv_writes_during_generation": False,
        "description": "Frozen base model sees the complete scene prompt.",
    },
    "normal_full": {
        "adapter": True,
        "attention_context": "system_and_user",
        "rwkv_prime": "same_generation_prompt",
        "rwkv_writes_during_generation": True,
        "description": "Adapter and ordinary attention both see the complete scene prompt.",
    },
    "no_write_full": {
        "adapter": True,
        "attention_context": "system_and_user",
        "rwkv_prime": "same_generation_prompt",
        "rwkv_writes_during_generation": False,
        "description": (
            "Adapter sees the complete scene prompt with all RWKV-MS writes disabled; "
            "compare against normal_full to measure the contribution of state writes."
        ),
    },
    "state_only": {
        "adapter": True,
        "attention_context": "system_only",
        "rwkv_prime": "system_and_user",
        "rwkv_online_carrier": (
            "delta_state, rwkv_ms_positions, and rwkv_ms_previous_source"
        ),
        "kv_cache_carried_from_prime": False,
        "rwkv_writes_during_prime": True,
        "rwkv_writes_during_generation": False,
        "description": (
            "Prime online state from the complete prompt, discard ordinary KV state, "
            "then generate from a system-only prompt with memory writes frozen."
        ),
    },
    "state_only_donor": {
        "adapter": True,
        "attention_context": "current_system_only",
        "rwkv_prime": "next_selected_validation_row_system_and_user_cyclic",
        "rwkv_online_carrier": (
            "donor delta_state, rwkv_ms_positions, and rwkv_ms_previous_source"
        ),
        "donor_pool": "selected_validation_rows_only",
        "donor_rule": DONOR_RULE_CYCLIC,
        "score_target": "current_row_gold",
        "kv_cache_carried_from_prime": False,
        "rwkv_writes_during_prime": True,
        "rwkv_writes_during_generation": False,
        "description": (
            "Prime online state from the next selected validation row, discard ordinary "
            "KV state, then query with the current row's system-only prompt and score "
            "against the current gold."
        ),
    },
    "state_only_shuffled": {
        "adapter": True,
        "attention_context": "current_system_only",
        "rwkv_prime": "deterministic_label_distinct_shuffled_selected_row",
        "rwkv_online_carrier": (
            "shuffled delta_state, rwkv_ms_positions, and rwkv_ms_previous_source"
        ),
        "donor_pool": "selected_rows_bijective_without_replacement",
        "donor_rule": "deterministic_label_distinct_non_donor_permutation_v1",
        "score_target": "current_row_gold",
        "kv_cache_carried_from_prime": False,
        "rwkv_writes_during_prime": True,
        "rwkv_writes_during_generation": False,
        "description": (
            "Prime online state from an independently permuted, label-distinct row "
            "that is neither the source nor its locked donor; discard ordinary KV "
            "state, then query the current system-only prompt with writes frozen."
        ),
    },
    "state_only_no_write": {
        "adapter": True,
        "attention_context": "system_only",
        "rwkv_prime": "system_and_user",
        "rwkv_online_carrier": (
            "zero-initialized delta_state, rwkv_ms_positions, and "
            "rwkv_ms_previous_source"
        ),
        "kv_cache_carried_from_prime": False,
        "rwkv_writes_during_prime": False,
        "rwkv_writes_during_generation": False,
        "description": (
            "Match state_only token processing while disabling every RWKV-MS write; "
            "this is the zero-state control."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--row-indices",
        help="Comma-separated zero-based source row indices; at least one is required.",
    )
    selection.add_argument(
        "--row-indices-file",
        type=Path,
        help=(
            "JSON integer array, comma/newline-separated indices, or a JSON manifest "
            "with rows containing source_index and row_sha256."
        ),
    )
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument(
        "--donor-rule",
        choices=DONOR_RULES,
        default=DONOR_RULE_CYCLIC,
        help="State-only donor assignment; the default preserves historical evaluations.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--normal-fusion-profile",
        default="native",
        choices=NORMAL_FUSION_PROFILES,
    )
    parser.add_argument(
        "--expected-memory-layer-count",
        type=int,
        help="Defaults to the number of target_layers in delta_mem_config.json.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate protected historical-V6 or locked V14 Hard32 bindings "
            "without loading a model or creating output files."
        ),
    )
    parser.add_argument(
        "--evaluation-contract",
        choices=EVALUATION_CONTRACTS,
        default="generic",
    )
    parser.add_argument(
        "--focused-source-manifest",
        type=Path,
        help=(
            "Exact train-only hard-failure source manifest required by the focused "
            "train-overfit contract."
        ),
    )
    parser.add_argument(
        "--focused-train-selection-receipt",
        type=Path,
        help=(
            "Passing train-only selected-checkpoint receipt required before the "
            "focused hard-failure Hard32 contract may access validation data."
        ),
    )
    parser.add_argument(
        "--hard32-receipt",
        type=Path,
        help=(
            "Passed scene_v6_identity_hard32 receipt required before the protected "
            "full-170 matched-donor validation may run."
        ),
    )
    parser.add_argument(
        "--scene-v7-train32-receipt",
        type=Path,
        help=(
            "Passed scene_v7_train32_overfit receipt authorizing the exact bound "
            "checkpoint for the fixed scene_v6_identity_hard32 evaluation only."
        ),
    )
    parser.add_argument(
        "--scene-v8-train32-receipt",
        type=Path,
        help=(
            "Passed Scene-Memory V8 Train32 value/causal gate receipt required "
            "for scene_v8_authorized_hard32. It authorizes only the exact "
            "checkpoint and frozen Hard32."
        ),
    )
    parser.add_argument(
        "--scene-v14-value14-receipt",
        type=Path,
        help=(
            "Exact Scene-Memory V14 Value14 diagnostic receipt bound by the "
            "checkpoint-3 candidate lock. Its historical pass/fail status is "
            "preserved and does not itself authorize Hard32."
        ),
    )
    parser.add_argument(
        "--scene-v14-candidate-lock",
        type=Path,
        help=(
            "Canonical tracked V14 checkpoint-3 candidate lock required for "
            "scene_v14_authorized_hard32. No other lock path is accepted."
        ),
    )
    parser.add_argument(
        "--scene-v14-launch-receipt",
        type=Path,
        help=(
            "Attached V14 launch receipt bound by --scene-v14-value14-receipt."
        ),
    )
    parser.add_argument(
        "--scene-v14-completion-receipt",
        type=Path,
        help=(
            "Attached V14 completion receipt bound by --scene-v14-value14-receipt."
        ),
    )
    parser.add_argument(
        "--scene-v15-selection-receipt",
        type=Path,
        help=(
            "Complete post-save Train32 checkpoint-selection receipt required for "
            "scene_v15_authorized_hard32."
        ),
    )
    parser.add_argument(
        "--scene-v15-candidate-lock",
        type=Path,
        help=(
            "Canonical V15 candidate lock beside --scene-v15-selection-receipt."
        ),
    )
    parser.add_argument(
        "--scene-v15-launch-receipt",
        type=Path,
        help="Attached production V15 launch receipt bound by the selection receipt.",
    )
    parser.add_argument(
        "--scene-v15-completion-receipt",
        type=Path,
        help=(
            "Completed production V15 receipt binding all four post-update checkpoints."
        ),
    )
    return parser.parse_args()


def validate_scene_v7_train32_receipt_scope(
    *,
    evaluation_contract: str,
    receipt_path: Path | None,
) -> None:
    if (
        receipt_path is not None
        and evaluation_contract != "scene_v6_identity_hard32"
    ):
        raise ValueError(
            "--scene-v7-train32-receipt is accepted only by "
            "scene_v6_identity_hard32"
        )


def validate_scene_v8_train32_receipt_scope(
    *,
    evaluation_contract: str,
    receipt_path: Path | None,
) -> None:
    if receipt_path is not None and evaluation_contract != SCENE_V8_HARD32_CONTRACT:
        raise ValueError(
            "--scene-v8-train32-receipt is accepted only by "
            f"{SCENE_V8_HARD32_CONTRACT}"
        )
    if evaluation_contract == SCENE_V8_HARD32_CONTRACT and receipt_path is None:
        raise ValueError(
            f"{SCENE_V8_HARD32_CONTRACT} requires --scene-v8-train32-receipt"
        )


def validate_scene_v14_receipt_scope(
    *,
    evaluation_contract: str,
    gate_receipt: Path | None,
    candidate_lock: Path | None,
    launch_receipt: Path | None,
    completion_receipt: Path | None,
) -> None:
    supplied = {
        "--scene-v14-value14-receipt": gate_receipt,
        "--scene-v14-candidate-lock": candidate_lock,
        "--scene-v14-launch-receipt": launch_receipt,
        "--scene-v14-completion-receipt": completion_receipt,
    }
    provided = [name for name, value in supplied.items() if value is not None]
    if provided and evaluation_contract != SCENE_V14_HARD32_CONTRACT:
        raise ValueError(
            "V14 receipts are accepted only by "
            f"{SCENE_V14_HARD32_CONTRACT}: {', '.join(provided)}"
        )
    if evaluation_contract == SCENE_V14_HARD32_CONTRACT:
        missing = [name for name, value in supplied.items() if value is None]
        if missing:
            raise ValueError(
                f"{SCENE_V14_HARD32_CONTRACT} requires the canonical candidate "
                "lock plus exact diagnostic, launch, and completion receipts; "
                "missing: " + ", ".join(missing)
            )


def validate_scene_v15_receipt_scope(
    *,
    evaluation_contract: str,
    selection_receipt: Path | None,
    candidate_lock: Path | None,
    launch_receipt: Path | None,
    completion_receipt: Path | None,
) -> None:
    supplied = {
        "--scene-v15-selection-receipt": selection_receipt,
        "--scene-v15-candidate-lock": candidate_lock,
        "--scene-v15-launch-receipt": launch_receipt,
        "--scene-v15-completion-receipt": completion_receipt,
    }
    provided = [name for name, value in supplied.items() if value is not None]
    if provided and evaluation_contract != SCENE_V15_HARD32_CONTRACT:
        raise ValueError(
            "V15 receipts are accepted only by "
            f"{SCENE_V15_HARD32_CONTRACT}: {', '.join(provided)}"
        )
    if evaluation_contract == SCENE_V15_HARD32_CONTRACT:
        missing = [name for name, value in supplied.items() if value is None]
        if missing:
            raise ValueError(
                f"{SCENE_V15_HARD32_CONTRACT} requires the exact Train32 "
                "selection receipt, canonical candidate lock, launch receipt, "
                "and completion receipt; missing: " + ", ".join(missing)
            )


def validate_scene_focused_artifact_scope(
    *,
    evaluation_contract: str,
    source_manifest: Path | None,
    train_selection_receipt: Path | None,
) -> None:
    if evaluation_contract == SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT:
        if source_manifest is None:
            raise ValueError(
                f"{SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT} requires "
                "--focused-source-manifest"
            )
        if train_selection_receipt is not None:
            raise ValueError("Focused train-overfit forbids a Hard32 selection receipt")
        return
    if evaluation_contract == SCENE_HARD_FAILURE_HARD32_CONTRACT:
        if train_selection_receipt is None:
            raise ValueError(
                f"{SCENE_HARD_FAILURE_HARD32_CONTRACT} requires "
                "--focused-train-selection-receipt"
            )
        if source_manifest is not None:
            raise ValueError("Focused Hard32 does not accept a training source manifest")
        return
    supplied = []
    if source_manifest is not None:
        supplied.append("--focused-source-manifest")
    if train_selection_receipt is not None:
        supplied.append("--focused-train-selection-receipt")
    if supplied:
        raise ValueError(
            "Focused hard-failure artifacts are accepted only by their experiment "
            f"contracts: {', '.join(supplied)}"
        )


def is_identity_hard32_contract(contract: str | Mapping[str, Any]) -> bool:
    name = contract.get("name") if isinstance(contract, Mapping) else contract
    return name in {
        "scene_v6_identity_hard32",
        SCENE_V8_HARD32_CONTRACT,
        SCENE_V14_HARD32_CONTRACT,
        SCENE_V15_HARD32_CONTRACT,
        SCENE_HARD_FAILURE_HARD32_CONTRACT,
    }


def is_fixed_scope_hard32_contract(contract: str | Mapping[str, Any]) -> bool:
    name = contract.get("name") if isinstance(contract, Mapping) else contract
    return name in {
        SCENE_V8_HARD32_CONTRACT,
        SCENE_V14_HARD32_CONTRACT,
        SCENE_V15_HARD32_CONTRACT,
        SCENE_HARD_FAILURE_HARD32_CONTRACT,
    }


def parse_row_indices(raw: str) -> list[int]:
    text = raw.strip()
    if not text:
        raise ValueError("Row selection is empty")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list):
        values = decoded
    else:
        values = [item for item in text.replace("\n", ",").split(",") if item.strip()]
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Boolean row index is invalid: {value!r}")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row index: {value!r}") from exc
        if str(value).strip() != str(index):
            raise ValueError(f"Row index must be an integer: {value!r}")
        if index < 0:
            raise ValueError(f"Row index must be non-negative: {index}")
        if index in indices:
            raise ValueError(f"Duplicate row index: {index}")
        indices.append(index)
    if not indices:
        raise ValueError("Row selection is empty")
    return indices


def parse_selection_manifest(raw: str) -> tuple[list[int], dict[int, str]]:
    """Parse a row selection and optional expected source-row hashes."""

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return parse_row_indices(raw), {}
    if not isinstance(decoded, dict):
        return parse_row_indices(raw), {}

    rows = decoded.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Selection manifest must contain a non-empty 'rows' list")
    indices: list[int] = []
    expected_hashes: dict[int, str] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Selection row {ordinal} must be an object")
        if not {"source_index", "row_sha256"}.issubset(row):
            raise ValueError(
                f"Selection row {ordinal} must contain source_index and row_sha256"
            )
        index = parse_row_indices(str(row["source_index"]))[0]
        if index in expected_hashes:
            raise ValueError(f"Duplicate row index: {index}")
        row_hash = row["row_sha256"]
        if not isinstance(row_hash, str) or SHA256_RE.fullmatch(row_hash) is None:
            raise ValueError(f"Selection row {ordinal} has an invalid row_sha256")
        indices.append(index)
        expected_hashes[index] = row_hash
    return indices, expected_hashes


def parse_selection_dataset_contract(raw: str) -> dict[str, str] | None:
    """Read the optional dataset identity embedded in a selection manifest."""

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None

    dataset = decoded.get("dataset")
    if dataset is None:
        if decoded.get("schema") == "rwkv_ms_scene_eval_selection.v1":
            raise ValueError("Scene evaluation selection manifest is missing 'dataset'")
        return None
    if not isinstance(dataset, dict):
        raise ValueError("Selection manifest 'dataset' must be an object")
    split = dataset.get("split")
    path = dataset.get("path")
    digest = dataset.get("sha256")
    if split != "val":
        raise ValueError("Selection manifest dataset split must be 'val'")
    if not isinstance(path, str) or not path:
        raise ValueError("Selection manifest dataset path is missing")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise ValueError("Selection manifest dataset SHA-256 is invalid")
    return {"split": split, "path": path, "sha256": digest}


def selected_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    if not conditions:
        raise ValueError("At least one condition is required")
    unknown = [item for item in conditions if item not in SUPPORTED_CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(unknown)}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must not contain duplicates")
    return conditions


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def _validate_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    description: str,
) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        raise ValueError(f"{description} {field} is missing or invalid")
    if recorded != _canonical_sha256(unsigned):
        raise ValueError(f"{description} {field} differs")
    return recorded


def _reject_focused_protected_path(path: Path, *, description: str) -> None:
    protected = {"val", "validation", "test", "holdout", "hard32"}
    for component in path.expanduser().absolute().parts:
        normalized = component.lower()
        if normalized in protected or Path(normalized).stem in protected:
            raise ValueError(
                f"Focused train-only {description} resolves through a protected path: {path}"
            )


def _focused_regular_file(
    path: Path,
    *,
    description: str,
    train_only: bool,
) -> Path:
    lexical = path.expanduser().absolute()
    if train_only:
        _reject_focused_protected_path(lexical, description=description)
    current = lexical
    while True:
        if current.is_symlink():
            raise ValueError(f"Focused {description} forbids symlinks: {current}")
        if current.parent == current:
            break
        current = current.parent
    resolved = lexical.resolve()
    if not resolved.is_file():
        raise ValueError(f"Focused {description} is missing: {resolved}")
    if train_only:
        _reject_focused_protected_path(resolved, description=description)
    return resolved


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in focused {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Focused {description} must be a JSON object")
    return payload


def _nonblank_jsonl_rows(
    path: Path,
    *,
    description: str,
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in focused {description}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Focused {description}:{line_number} must be a JSON object"
                )
            rows.append((raw, payload))
    return rows


def validate_focused_train_source_manifest(
    source_manifest_path: Path,
    *,
    dataset_file: Path,
) -> dict[str, Any]:
    """Validate the exact train-only hard-failure curriculum and pair schedule."""

    source_path = _focused_regular_file(
        source_manifest_path,
        description="source manifest",
        train_only=True,
    )
    dataset_path = _focused_regular_file(
        dataset_file,
        description="dataset",
        train_only=True,
    )
    if sha256_file(source_path) != SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("Focused source-manifest file SHA-256 differs")
    source = _load_json_object(source_path, description="source manifest")
    source_self_hash = _validate_self_hash(
        source,
        field="manifest_sha256",
        description="focused source manifest",
    )
    if source_self_hash != SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256:
        raise ValueError("Focused source-manifest self hash differs")
    if source.get("schema") != SCENE_HARD_FAILURE_SOURCE_SCHEMA:
        raise ValueError("Focused source-manifest schema differs")
    if source.get("task") != TASK_NAME:
        raise ValueError("Focused source-manifest task differs")
    if source.get("purpose") != SCENE_HARD_FAILURE_SOURCE_PURPOSE:
        raise ValueError("Focused source-manifest purpose differs")
    source_contract = source.get("contract")
    expected_episode_contract = {
        "episode_recent_messages": 0,
        "write_phase": "system + user",
        "read_supervision": "system + assistant",
    }
    if not isinstance(source_contract, dict) or source_contract != {
        "source_split": "train",
        "val_rows": 0,
        "test_rows": 0,
        "episode_contract": expected_episode_contract,
    }:
        raise ValueError("Focused source-manifest train-only contract differs")
    partitions = source.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train"}:
        raise ValueError("Focused source manifest must contain only the train partition")
    train = partitions["train"]
    if (
        not isinstance(train, dict)
        or train.get("source_split") != "train"
        or train.get("rows") != SCENE_HARD_FAILURE_ROWS
    ):
        raise ValueError("Focused source train partition differs")
    data_binding = train.get("data")
    row_binding = train.get("row_manifest")
    if not isinstance(data_binding, dict) or not isinstance(row_binding, dict):
        raise ValueError("Focused source artifact bindings are missing")
    declared_dataset = Path(str(data_binding.get("path", ""))).expanduser().resolve()
    if declared_dataset != dataset_path:
        raise ValueError("Focused source dataset path differs from --dataset-file")
    dataset_sha256 = sha256_file(dataset_path)
    if (
        dataset_sha256 != SCENE_HARD_FAILURE_TRAIN_FILE_SHA256
        or data_binding.get("sha256") != dataset_sha256
    ):
        raise ValueError("Focused train dataset SHA-256 differs")
    rows_path = _focused_regular_file(
        Path(str(row_binding.get("path", ""))),
        description="row manifest",
        train_only=True,
    )
    row_manifest_sha256 = sha256_file(rows_path)
    if (
        row_manifest_sha256 != SCENE_HARD_FAILURE_ROW_MANIFEST_FILE_SHA256
        or row_binding.get("sha256") != row_manifest_sha256
    ):
        raise ValueError("Focused row-manifest SHA-256 differs")

    pairing_binding = source.get("v7_pairing")
    if not isinstance(pairing_binding, dict):
        raise ValueError("Focused source pairing binding is missing")
    pair_file_binding = pairing_binding.get("pair_manifest")
    if not isinstance(pair_file_binding, dict):
        raise ValueError("Focused pair-manifest file binding is missing")
    pair_path = _focused_regular_file(
        Path(str(pair_file_binding.get("path", ""))),
        description="pair manifest",
        train_only=True,
    )
    pair_file_sha256 = sha256_file(pair_path)
    if (
        pair_file_sha256 != SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256
        or pair_file_binding.get("sha256") != pair_file_sha256
    ):
        raise ValueError("Focused pair-manifest file SHA-256 differs")

    dataset_rows = _nonblank_jsonl_rows(dataset_path, description="train dataset")
    row_records = _nonblank_jsonl_rows(rows_path, description="row manifest")
    if len(dataset_rows) != len(row_records) or len(dataset_rows) != SCENE_HARD_FAILURE_ROWS:
        raise ValueError("Focused train data and row-manifest counts differ")
    expected_hashes: dict[int, str] = {}
    row_manifests: list[dict[str, Any]] = []
    for ordinal, ((raw_data, _), (_, row)) in enumerate(
        zip(dataset_rows, row_records, strict=True)
    ):
        _validate_self_hash(
            row,
            field="record_sha256",
            description=f"focused row manifest {ordinal}",
        )
        row_sha256 = sha256_text(raw_data)
        if (
            row.get("schema") != SCENE_HARD_FAILURE_ROW_SCHEMA
            or row.get("train_row_ordinal") != ordinal
            or row.get("source_split") != "train"
            or row.get("row_sha256") != row_sha256
        ):
            raise ValueError(f"Focused row-manifest identity differs at ordinal {ordinal}")
        expected_hashes[ordinal] = row_sha256
        row_manifests.append(row)

    pair_manifest = _load_json_object(pair_path, description="pair manifest")
    pair_self_hash = _validate_self_hash(
        pair_manifest,
        field="manifest_sha256",
        description="focused pair manifest",
    )
    if pair_self_hash != SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256:
        raise ValueError("Focused pair-manifest self hash differs")
    if pair_manifest.get("schema") != SCENE_HARD_FAILURE_PAIR_SCHEMA:
        raise ValueError("Focused pair-manifest schema differs")
    pair_dataset = pair_manifest.get("dataset")
    ordered_hashes = [expected_hashes[index] for index in range(SCENE_HARD_FAILURE_ROWS)]
    if not isinstance(pair_dataset, dict) or pair_dataset != {
        "path": str(dataset_path),
        "sha256": dataset_sha256,
        "rows": SCENE_HARD_FAILURE_ROWS,
        "ordered_row_sha256": _canonical_sha256(ordered_hashes),
    }:
        raise ValueError("Focused pair-manifest dataset binding differs")
    directed = pair_manifest.get("directed_pairs")
    if not isinstance(directed, list) or len(directed) != SCENE_HARD_FAILURE_ROWS:
        raise ValueError("Focused pair manifest must cover all 32 train rows")
    entries_sha256 = pair_manifest.get("entries_sha256")
    if (
        entries_sha256 != _canonical_sha256(directed)
        or entries_sha256 != SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256
    ):
        raise ValueError("Focused pair-manifest entry digest differs")
    donor_by_source: dict[int, int] = {}
    for entry in directed:
        if not isinstance(entry, dict):
            raise ValueError("Focused pair-manifest entry is invalid")
        _validate_self_hash(
            entry,
            field="entry_sha256",
            description="focused pair-manifest entry",
        )
        source_index = entry.get("train_row_ordinal")
        donor_index = entry.get("donor_train_row_ordinal")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or isinstance(donor_index, bool)
            or not isinstance(donor_index, int)
            or source_index not in expected_hashes
            or donor_index not in expected_hashes
            or source_index == donor_index
            or source_index in donor_by_source
        ):
            raise ValueError("Focused pair-manifest ordinals differ")
        if (
            entry.get("source_row_sha256") != expected_hashes[source_index]
            or entry.get("donor_row_sha256") != expected_hashes[donor_index]
            or entry.get("source_label_sha256") == entry.get("donor_label_sha256")
        ):
            raise ValueError("Focused pair-manifest row or label identity differs")
        donor_by_source[source_index] = donor_index
    if set(donor_by_source) != set(range(SCENE_HARD_FAILURE_ROWS)):
        raise ValueError("Focused pair-manifest source coverage differs")
    for source_index, donor_index in donor_by_source.items():
        if donor_by_source.get(donor_index) != source_index:
            raise ValueError("Focused pair-manifest mapping is not reciprocal")

    if (
        pairing_binding.get("dataset_sha256") != dataset_sha256
        or pairing_binding.get("directed_entry_count") != SCENE_HARD_FAILURE_ROWS
        or pairing_binding.get("entries_sha256") != entries_sha256
        or pair_file_binding.get("manifest_sha256") != pair_self_hash
    ):
        raise ValueError("Focused source pairing binding differs")
    return {
        "kind": "scene_hard_failure_train_only_source_v1",
        "source_manifest": {
            "path": str(source_path),
            "file_sha256": sha256_file(source_path),
            "manifest_sha256": source_self_hash,
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "split": "train",
            "rows": SCENE_HARD_FAILURE_ROWS,
        },
        "row_manifest": {
            "path": str(rows_path),
            "sha256": row_manifest_sha256,
        },
        "pair_manifest": {
            "path": str(pair_path),
            "file_sha256": pair_file_sha256,
            "manifest_sha256": pair_self_hash,
            "entries_sha256": entries_sha256,
        },
        "selection": [
            {"source_index": index, "row_sha256": expected_hashes[index]}
            for index in range(SCENE_HARD_FAILURE_ROWS)
        ],
        "expected_row_hashes": expected_hashes,
        "donor_by_source_index": donor_by_source,
        "protected_evaluation_accessed": False,
    }


def _receipt_artifact_digest(
    artifacts: Mapping[str, Any],
    name: str,
    *,
    checkpoint_dir: Path,
) -> str:
    record = artifacts.get(name)
    if isinstance(record, str):
        digest = record
        declared_path = checkpoint_dir / name
    elif isinstance(record, Mapping):
        digest = record.get("sha256")
        declared_path = Path(str(record.get("path", checkpoint_dir / name))).expanduser()
    else:
        raise ValueError(f"Focused selection receipt checkpoint artifact is missing: {name}")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"Focused selection receipt artifact hash is invalid: {name}")
    resolved = _focused_regular_file(
        declared_path,
        description=f"selected checkpoint {name}",
        train_only=True,
    )
    if resolved != (checkpoint_dir / name).resolve():
        raise ValueError(f"Focused selection receipt artifact path differs: {name}")
    if sha256_file(resolved) != digest:
        raise ValueError(f"Focused selection receipt artifact SHA-256 differs: {name}")
    return digest


def _focused_bound_artifact(
    record: Any,
    *,
    expected_path: Path,
    description: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Focused {description} binding is missing")
    path_value = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("bytes")
    if (
        not isinstance(path_value, str)
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 0
    ):
        raise ValueError(f"Focused {description} binding is invalid")
    resolved = _focused_regular_file(
        Path(path_value),
        description=description,
        train_only=True,
    )
    if (
        resolved != expected_path.resolve()
        or resolved.stat().st_size != byte_count
        or sha256_file(resolved) != digest
    ):
        raise ValueError(f"Focused {description} current bytes differ")
    return dict(record)


def _hard_failure_audit_module() -> Any:
    # Import lazily because the hard-failure data contract imports this evaluator.
    from experiments.rethinking_rwkv_ms_gemma import scene_hard_failure_run_audit

    return scene_hard_failure_run_audit


def _recompute_focused_adapter_change(
    *,
    initial_manifest_path: Path,
    initial_adapter_path: Path,
    checkpoint_adapter_path: Path,
    step: int,
) -> dict[str, Any]:
    audit_module = _hard_failure_audit_module()
    try:
        initial_manifest = _load_json_object(
            initial_manifest_path,
            description="initial adapter manifest",
        )
        initial_adapter = audit_module.load_finite_adapter(
            initial_adapter_path
        )
        checkpoint_adapter = audit_module.load_finite_adapter(
            checkpoint_adapter_path
        )
        trainable_names = audit_module._validate_initial_adapter_topology(
            initial_manifest.get("topology"),
            initial_adapter,
        )
        return audit_module.adapter_change_record(
            initial_adapter,
            checkpoint_adapter,
            trainable_names=trainable_names,
            checkpoint_step=step,
            smoke=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Focused checkpoint-{step} current adapter recomputation failed: {exc}"
        ) from exc


def _focused_checkpoint_audit_change(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    step: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the checkpoint audit that proves coverage for the current bytes."""

    audit_module = _hard_failure_audit_module()
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Focused endpoint checkpoint artifacts are missing")
    required_names = (
        "delta_mem_adapter.pt",
        audit_module.AUDIT_FILENAME,
    )
    digests = {
        name: _receipt_artifact_digest(
            artifacts,
            name,
            checkpoint_dir=checkpoint_dir,
        )
        for name in required_names
    }
    current = checkpoint.get("current_adapter_validation")
    if not isinstance(current, Mapping):
        raise ValueError("Focused endpoint current adapter validation is missing")
    run_root = checkpoint_dir.parent.parent
    initial_manifest_path = run_root / "initial_adapter" / "initial_adapter_manifest.json"
    initial_adapter_path = run_root / "initial_adapter" / "delta_mem_adapter.pt"
    checkpoint_adapter_path = checkpoint_dir / "delta_mem_adapter.pt"
    evidence_paths = {
        "initial_adapter_manifest": initial_manifest_path,
        "initial_adapter": initial_adapter_path,
        "checkpoint_adapter": checkpoint_adapter_path,
    }
    before = {
        name: _focused_bound_artifact(
            current.get(name),
            expected_path=evidence_path,
            description=name.replace("_", " "),
        )
        for name, evidence_path in evidence_paths.items()
    }
    if before["checkpoint_adapter"] != dict(
        artifacts["delta_mem_adapter.pt"]
    ):
        raise ValueError("Focused endpoint checkpoint adapter binding differs")

    audit_path = checkpoint_dir / audit_module.AUDIT_FILENAME
    audit = _load_json_object(audit_path, description="hard-failure checkpoint audit")
    audit_receipt_sha256 = _validate_self_hash(
        audit,
        field="receipt_sha256",
        description="hard-failure checkpoint audit",
    )
    if (
        audit.get("schema") != audit_module.AUDIT_SCHEMA
        or audit.get("run_root") != str(run_root)
        or audit.get("checkpoint") != str(checkpoint_dir)
        or audit.get("checkpoint_optimizer_step") != step
        or audit.get("nontrainable_adapter_tensors_unchanged") is not True
        or checkpoint.get("checkpoint_audit_receipt_sha256")
        != audit_receipt_sha256
    ):
        raise ValueError("Focused endpoint checkpoint audit binding differs")
    change = audit.get("adapter_change")
    if not isinstance(change, Mapping):
        raise ValueError("Focused endpoint checkpoint audit change is missing")
    recomputed = _recompute_focused_adapter_change(
        initial_manifest_path=initial_manifest_path,
        initial_adapter_path=initial_adapter_path,
        checkpoint_adapter_path=checkpoint_adapter_path,
        step=step,
    )
    after = {
        name: _focused_bound_artifact(
            current.get(name),
            expected_path=evidence_path,
            description=name.replace("_", " "),
        )
        for name, evidence_path in evidence_paths.items()
    }
    if before != after or recomputed != dict(change):
        raise ValueError("Focused endpoint audit differs from current adapter bytes")
    coverage_fields = (
        "trainable_tensor_family_count",
        "target_layer_count",
        "trainable_family_layer_coverage",
        "full_trainable_family_coverage",
    )
    if any(audit.get(field) != recomputed.get(field) for field in coverage_fields):
        raise ValueError("Focused endpoint audit top-level coverage differs")
    return recomputed, digests


def _validate_focused_coverage_claims(
    checkpoint: Mapping[str, Any],
    coverage: Mapping[str, Any],
    audit_change: Mapping[str, Any],
    *,
    step: int,
    adapter_sha256: str,
) -> bool:
    """Bind receipt coverage fields to the exact audited family map."""

    audit_module = _hard_failure_audit_module()
    suffixes = tuple(audit_module.TRAINABLE_ADAPTER_TENSOR_SUFFIXES)
    expected_family_coverage = audit_change.get("trainable_family_layer_coverage")
    expected_missing = audit_change.get("missing_trainable_family_layers")
    changed_count = audit_change.get("changed_trainable_tensor_count")
    full_coverage = audit_change.get("full_trainable_family_coverage")
    required = step == 64
    if (
        len(suffixes) != 27
        or not isinstance(expected_family_coverage, Mapping)
        or set(expected_family_coverage) != set(suffixes)
        or not all(
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= 42
            for count in expected_family_coverage.values()
        )
        or not isinstance(expected_missing, Mapping)
        or set(expected_missing)
        != {
            suffix
            for suffix, count in expected_family_coverage.items()
            if count != 42
        }
        or not all(
            isinstance(layers, list)
            and layers == sorted(set(layers))
            and all(
                isinstance(layer, int)
                and not isinstance(layer, bool)
                and 0 <= layer < 42
                for layer in layers
            )
            and len(layers) == 42 - expected_family_coverage[suffix]
            for suffix, layers in expected_missing.items()
        )
        or changed_count != sum(expected_family_coverage.values())
        or audit_change.get("expected_trainable_tensor_count") != 1134
        or audit_change.get("trainable_tensor_family_count") != 27
        or audit_change.get("target_layer_count") != 42
        or audit_change.get("changed_nontrainable_tensor_count") != 0
        or full_coverage
        is not (not expected_missing and changed_count == 1134)
        or audit_change.get("full_trainable_family_coverage_required") is not required
        or (required and full_coverage is not True)
        or audit_change.get("frozen_adapter_tensors_unchanged") is not True
    ):
        raise ValueError("Focused endpoint audited coverage is invalid")

    current = checkpoint.get("current_adapter_validation")
    artifacts = checkpoint.get("artifacts")
    if not isinstance(current, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("Focused endpoint current adapter coverage is missing")
    current_adapter = current.get("checkpoint_adapter")
    change_sha256 = _canonical_sha256(audit_change)
    coverage_fields = (
        "changed_trainable_tensor_count",
        "expected_trainable_tensor_count",
        "trainable_tensor_family_count",
        "target_layer_count",
        "trainable_family_layer_coverage",
        "missing_trainable_family_layers",
        "full_trainable_family_coverage",
        "full_trainable_family_coverage_required",
    )
    if (
        not isinstance(current_adapter, Mapping)
        or dict(current_adapter) != dict(artifacts["delta_mem_adapter.pt"])
        or current_adapter.get("sha256") != adapter_sha256
        or current.get("recomputed_adapter_change_canonical_sha256")
        != change_sha256
        or current.get("frozen_adapter_tensors_unchanged") is not True
        or any(current.get(field) != audit_change.get(field) for field in coverage_fields)
        or any(checkpoint.get(field) != audit_change.get(field) for field in coverage_fields)
        or any(coverage.get(field) != audit_change.get(field) for field in coverage_fields)
        or coverage.get("checkpoint_adapter_sha256") != adapter_sha256
        or coverage.get("recomputed_adapter_change_canonical_sha256")
        != change_sha256
        or coverage.get("frozen_adapter_tensors_unchanged") is not True
    ):
        raise ValueError("Focused endpoint current-byte coverage binding differs")
    return full_coverage is True


def _focused_endpoint_gate_passed(endpoint: Mapping[str, Any]) -> bool:
    binding = endpoint.get("gate")
    if not isinstance(binding, Mapping):
        raise ValueError("Focused endpoint gate binding is missing")
    path_value = binding.get("focused_gate_path")
    if not isinstance(path_value, str):
        raise ValueError("Focused endpoint gate path is missing")
    path = _focused_regular_file(
        Path(path_value),
        description="train-overfit endpoint gate",
        train_only=True,
    )
    gate = _load_json_object(path, description="train-overfit endpoint gate")
    passed = gate.get("all_gates_passed")
    if (
        not isinstance(passed, bool)
        or binding.get("file_sha256") != sha256_file(path)
        or binding.get("canonical_sha256") != _canonical_sha256(gate)
        or gate.get("schema") != "rwkv_ms_scene_focused_recovery_gate.v1"
        or gate.get("stage") != "train_overfit"
        or gate.get("task") != TASK_NAME
        or gate.get("source_indices") != list(range(SCENE_HARD_FAILURE_ROWS))
        or gate.get("status")
        != ("diagnostic_pass" if passed else "diagnostic_fail")
    ):
        raise ValueError("Focused endpoint gate binding differs")
    return passed


def validate_focused_train_selection_authorization(
    receipt_path: Path,
    *,
    memory_dir: Path,
) -> dict[str, Any]:
    """Validate the train-only selector before the protected Hard32 is opened."""

    audit_filename = _hard_failure_audit_module().AUDIT_FILENAME
    path = _focused_regular_file(
        receipt_path,
        description="train selection receipt",
        train_only=True,
    )
    receipt = _load_json_object(path, description="train selection receipt")
    receipt_sha256 = _validate_self_hash(
        receipt,
        field="receipt_sha256",
        description="focused train selection receipt",
    )
    if receipt.get("schema") != SCENE_HARD_FAILURE_TRAIN_SELECTION_SCHEMA:
        raise ValueError("Focused train selection receipt schema differs")
    if (
        receipt.get("status") != "pass"
        or receipt.get("stage") != "train_overfit"
        or receipt.get("task") != TASK_NAME
        or receipt.get("selection_policy")
        != "first_train32_gate_pass_with_full_current_adapter_coverage_v2"
        or receipt.get("endpoint_schedule") != [16, 32, 48, 64]
        or receipt.get("selection_reason")
        != "first_train32_gate_pass_with_full_current_adapter_coverage"
    ):
        raise ValueError("Focused train selection receipt did not pass train-overfit")
    authorization = receipt.get("authorization")
    if not isinstance(authorization, Mapping) or dict(authorization) != {
        "hard32_authorized": True,
        "selected_checkpoint_only": True,
        "full_validation": False,
        "test": False,
        "other_benchmarks": False,
    }:
        raise ValueError("Focused train selection authorization scope differs")

    checkpoint = receipt.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Focused train selection checkpoint is missing")
    selected_dir = Path(str(checkpoint.get("path", ""))).expanduser().resolve()
    expected_dir = memory_dir.expanduser().resolve()
    if selected_dir != expected_dir or not expected_dir.is_dir() or expected_dir.is_symlink():
        raise ValueError("Focused Hard32 checkpoint differs from train-only selection")
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("Focused selected checkpoint global_step is invalid")
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Focused selected checkpoint artifacts are missing")
    artifact_sha256 = {
        name: _receipt_artifact_digest(
            artifacts,
            name,
            checkpoint_dir=expected_dir,
        )
        for name in (
            "delta_mem_adapter.pt",
            "delta_mem_config.json",
            audit_filename,
        )
    }
    for name in (
        "training_protocol.json",
        "scene_state_identity_pairing_manifest.json",
        "scene_memory_v15_row_objective.json",
    ):
        if name in artifacts:
            artifact_sha256[name] = _receipt_artifact_digest(
                artifacts,
                name,
                checkpoint_dir=expected_dir,
            )

    evaluated_steps = receipt.get("evaluated_endpoint_steps")
    evaluated = receipt.get("evaluated_endpoints")
    selected_eligibility = receipt.get("selected_endpoint_eligibility")
    if (
        not isinstance(evaluated_steps, list)
        or evaluated_steps != [16, 32, 48, 64][: len(evaluated_steps)]
        or not isinstance(evaluated, list)
        or len(evaluated) != len(evaluated_steps)
        or not evaluated
        or not all(isinstance(item, Mapping) for item in evaluated)
        or not isinstance(selected_eligibility, Mapping)
    ):
        raise ValueError("Focused train selection endpoint sequence differs")
    trainer_dir = expected_dir.parent
    run_root = trainer_dir.parent
    if trainer_dir.name != "trainer" or expected_dir.name != f"checkpoint-{global_step}":
        raise ValueError("Focused selected checkpoint layout differs")
    eligible: list[Mapping[str, Any]] = []
    for step, endpoint in zip(evaluated_steps, evaluated):
        endpoint_checkpoint = endpoint.get("checkpoint")
        endpoint_coverage = endpoint.get("coverage_evidence")
        endpoint_dir = run_root / "trainer" / f"checkpoint-{step}"
        if (
            not isinstance(endpoint_checkpoint, Mapping)
            or endpoint_checkpoint.get("path") != str(endpoint_dir)
            or not isinstance(endpoint_coverage, Mapping)
        ):
            raise ValueError("Focused endpoint checkpoint binding differs")
        audit_change, endpoint_digests = _focused_checkpoint_audit_change(
            endpoint_checkpoint,
            checkpoint_dir=endpoint_dir,
            step=step,
        )
        full_coverage = _validate_focused_coverage_claims(
            endpoint_checkpoint,
            endpoint_coverage,
            audit_change,
            step=step,
            adapter_sha256=endpoint_digests["delta_mem_adapter.pt"],
        )
        benchmark_gate_passed = _focused_endpoint_gate_passed(endpoint)
        selection_eligible = endpoint.get("selection_eligible")
        if (
            endpoint.get("global_step") != step
            or endpoint.get("benchmark_gate_passed") is not benchmark_gate_passed
            or endpoint.get("full_coverage") is not full_coverage
            or not isinstance(selection_eligible, bool)
            or selection_eligible
            is not (benchmark_gate_passed and full_coverage)
        ):
            raise ValueError("Focused train selection endpoint eligibility differs")
        if selection_eligible:
            eligible.append(endpoint)
    selected_endpoint = evaluated[-1]
    if (
        eligible != [selected_endpoint]
        or selected_endpoint.get("global_step") != global_step
        or selected_endpoint.get("checkpoint") != checkpoint
        or selected_endpoint.get("gate") != receipt.get("gate")
        or dict(selected_eligibility)
        != {
            "benchmark_gate_passed": True,
            "full_coverage": True,
            "selection_eligible": True,
        }
    ):
        raise ValueError("Focused train selection did not choose first eligible endpoint")

    coverage = selected_endpoint.get("coverage_evidence")
    current = checkpoint.get("current_adapter_validation")
    if not isinstance(coverage, Mapping) or not isinstance(current, Mapping):
        raise ValueError("Focused selected checkpoint current coverage is missing")
    family_coverage = coverage.get("trainable_family_layer_coverage")
    current_adapter = current.get("checkpoint_adapter")
    change_sha256 = coverage.get("recomputed_adapter_change_canonical_sha256")
    if (
        coverage.get("changed_trainable_tensor_count") != 1134
        or coverage.get("expected_trainable_tensor_count") != 1134
        or coverage.get("trainable_tensor_family_count") != 27
        or coverage.get("target_layer_count") != 42
        or not isinstance(family_coverage, Mapping)
        or len(family_coverage) != 27
        or not all(
            isinstance(name, str) and count == 42
            for name, count in family_coverage.items()
        )
        or coverage.get("missing_trainable_family_layers") != {}
        or coverage.get("full_trainable_family_coverage") is not True
        or coverage.get("frozen_adapter_tensors_unchanged") is not True
        or not isinstance(change_sha256, str)
        or SHA256_RE.fullmatch(change_sha256) is None
        or coverage.get("checkpoint_adapter_sha256")
        != artifact_sha256["delta_mem_adapter.pt"]
        or not isinstance(current_adapter, Mapping)
        or dict(current_adapter) != dict(artifacts["delta_mem_adapter.pt"])
        or current_adapter.get("sha256") != artifact_sha256["delta_mem_adapter.pt"]
        or current.get("recomputed_adapter_change_canonical_sha256")
        != change_sha256
        or any(current.get(field) != coverage.get(field) for field in (
            "changed_trainable_tensor_count",
            "expected_trainable_tensor_count",
            "trainable_tensor_family_count",
            "target_layer_count",
            "trainable_family_layer_coverage",
            "missing_trainable_family_layers",
            "full_trainable_family_coverage",
            "full_trainable_family_coverage_required",
            "frozen_adapter_tensors_unchanged",
        ))
    ):
        raise ValueError("Focused selected checkpoint lacks full current-byte coverage")

    source = receipt.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Focused train selection source binding is missing")
    expected_source_fields = {
        "source_manifest_file_sha256": SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_sha256": SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256,
        "train_file_sha256": SCENE_HARD_FAILURE_TRAIN_FILE_SHA256,
        "pair_manifest_file_sha256": SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256,
        "pair_manifest_sha256": SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256,
        "entries_sha256": SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256,
    }
    if any(source.get(field) != expected for field, expected in expected_source_fields.items()):
        raise ValueError("Focused train selection source identity differs")
    source_manifest_path = source.get("source_manifest_path")
    if not isinstance(source_manifest_path, str):
        raise ValueError("Focused train selection source-manifest path is missing")
    bound_source_path = _focused_regular_file(
        Path(source_manifest_path),
        description="selection-bound source manifest",
        train_only=True,
    )
    if sha256_file(bound_source_path) != SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("Focused selection-bound source manifest differs")

    gate_binding = receipt.get("gate")
    if not isinstance(gate_binding, Mapping):
        raise ValueError("Focused train selection gate binding is missing")
    gate_path_value = gate_binding.get("focused_gate_path")
    if not isinstance(gate_path_value, str):
        raise ValueError("Focused train selection gate path is missing")
    gate_path = _focused_regular_file(
        Path(gate_path_value),
        description="train-overfit focused gate",
        train_only=True,
    )
    if gate_binding.get("file_sha256") != sha256_file(gate_path):
        raise ValueError("Focused train selection gate file SHA-256 differs")
    gate = _load_json_object(gate_path, description="train-overfit focused gate")
    if gate_binding.get("canonical_sha256") != _canonical_sha256(gate):
        raise ValueError("Focused train selection gate canonical SHA-256 differs")
    if (
        gate.get("schema") != "rwkv_ms_scene_focused_recovery_gate.v1"
        or gate.get("status") != "diagnostic_pass"
        or gate.get("stage") != "train_overfit"
        or gate.get("task") != TASK_NAME
        or gate.get("all_gates_passed") is not True
        or gate.get("source_indices") != list(range(SCENE_HARD_FAILURE_ROWS))
    ):
        raise ValueError("Focused train selection gate did not pass exact Train32")
    gate_input = gate.get("input")
    if not isinstance(gate_input, Mapping):
        raise ValueError("Focused train selection gate input binding is missing")
    evaluation_fingerprint = gate_input.get("evaluation_fingerprint")
    if (
        not isinstance(evaluation_fingerprint, str)
        or SHA256_RE.fullmatch(evaluation_fingerprint) is None
        or gate_binding.get("evaluation_fingerprint") != evaluation_fingerprint
    ):
        raise ValueError("Focused train selection evaluation fingerprint differs")
    evaluation_contract = gate_input.get("evaluation_contract")
    if (
        not isinstance(evaluation_contract, Mapping)
        or evaluation_contract.get("name")
        != SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT
        or evaluation_contract.get("split") != "train"
        or evaluation_contract.get("conditions") != list(SCENE_FOCUSED_CONDITIONS)
    ):
        raise ValueError("Focused train selection evaluation contract differs")
    contract_checkpoint = evaluation_contract.get("checkpoint")
    if not isinstance(contract_checkpoint, Mapping) or dict(contract_checkpoint) != {
        "memory_dir": str(expected_dir),
        "adapter_sha256": artifact_sha256["delta_mem_adapter.pt"],
        "config_sha256": artifact_sha256["delta_mem_config.json"],
    }:
        raise ValueError("Focused train gate checkpoint binding differs")
    contract_source = evaluation_contract.get("train_source")
    contract_source_manifest = (
        contract_source.get("source_manifest")
        if isinstance(contract_source, Mapping)
        else None
    )
    contract_source_dataset = (
        contract_source.get("dataset") if isinstance(contract_source, Mapping) else None
    )
    contract_source_pair = (
        contract_source.get("pair_manifest")
        if isinstance(contract_source, Mapping)
        else None
    )
    if (
        not isinstance(contract_source, Mapping)
        or not isinstance(contract_source_manifest, Mapping)
        or not isinstance(contract_source_dataset, Mapping)
        or not isinstance(contract_source_pair, Mapping)
        or contract_source_manifest.get("file_sha256")
        != SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256
        or contract_source_manifest.get("manifest_sha256")
        != SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256
        or contract_source_dataset.get("sha256")
        != SCENE_HARD_FAILURE_TRAIN_FILE_SHA256
        or contract_source_pair.get("file_sha256")
        != SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256
    ):
        raise ValueError("Focused train gate source binding differs")
    return {
        "authorization_kind": "scene_hard_failure_train_overfit_selection",
        "scope": SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
        "receipt": {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "receipt_sha256": receipt_sha256,
        },
        "selected_checkpoint": {
            "path": str(expected_dir),
            "global_step": global_step,
            "artifacts": artifact_sha256,
        },
        "evaluation_fingerprint": evaluation_fingerprint,
        "source": dict(source),
        "gate": {
            "path": str(gate_path),
            "file_sha256": sha256_file(gate_path),
            "canonical_sha256": _canonical_sha256(gate),
        },
    }


def _reject_symlink_components(path: Path, *, description: str) -> None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"Historical V6 contract forbids symlinks for {description}: {current}"
            )


def require_historical_tracked_worktree_clean() -> dict[str, Any]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Historical V6 contract cannot verify the tracked worktree"
        ) from exc
    if status:
        raise ValueError(
            "Historical V6 protected evaluation requires a clean tracked worktree"
        )
    return {
        "repository": str(PROJECT_ROOT),
        "tracked_worktree_clean": True,
        "untracked_files_ignored": True,
    }


def _historical_exact_path(
    path: Path,
    *,
    expected: Path,
    description: str,
    directory: bool = False,
) -> Path:
    _reject_symlink_components(path, description=description)
    _reject_symlink_components(expected, description=f"expected {description}")
    try:
        resolved = path.expanduser().resolve(strict=True)
        expected_resolved = expected.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Historical V6 {description} is missing: {exc.filename}"
        ) from exc
    if resolved != expected_resolved:
        raise ValueError(
            f"Historical V6 {description} path differs: "
            f"expected={expected_resolved} actual={resolved}"
        )
    correct_kind = resolved.is_dir() if directory else resolved.is_file()
    if not correct_kind:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"Historical V6 {description} must be a {kind}: {resolved}")
    return resolved


def _historical_artifact_binding(
    path: Path,
    *,
    description: str,
    expected_sha256: str,
) -> dict[str, Any]:
    _reject_symlink_components(path, description=description)
    if not path.is_file():
        raise ValueError(f"Historical V6 {description} is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Historical V6 {description} SHA-256 differs: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
    }


def validate_historical_v6_checkpoint(memory_dir: Path) -> dict[str, Any]:
    checkpoint = _historical_exact_path(
        memory_dir,
        expected=HISTORICAL_V6_CHECKPOINT,
        description="checkpoint",
        directory=True,
    )
    artifacts = {
        name: _historical_artifact_binding(
            checkpoint / name,
            description=f"checkpoint {name}",
            expected_sha256=digest,
        )
        for name, digest in HISTORICAL_V6_CHECKPOINT_ARTIFACT_SHA256.items()
    }
    try:
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Historical V6 trainer_state.json is invalid JSON") from exc
    if not isinstance(trainer_state, dict):
        raise ValueError("Historical V6 trainer_state.json must contain an object")
    if (
        trainer_state.get("global_step") != HISTORICAL_V6_EXPECTED_GLOBAL_STEP
        or trainer_state.get("max_steps") != HISTORICAL_V6_EXPECTED_GLOBAL_STEP
    ):
        raise ValueError(
            "Historical V6 trainer state must bind global_step=max_steps="
            f"{HISTORICAL_V6_EXPECTED_GLOBAL_STEP}"
        )
    return {
        "lineage_kind": "historical_artifact_identity_without_source_receipt",
        "checkpoint_step": HISTORICAL_V6_EXPECTED_GLOBAL_STEP,
        "memory_dir": str(checkpoint),
        "artifacts": artifacts,
        "training_source_commit": None,
        "training_source_lock_receipt": None,
        "lineage_limitation": HISTORICAL_V6_LINEAGE_LIMITATION,
    }


def _nonblank_raw_jsonl_rows(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            raw = raw_line.rstrip("\r\n")
            if raw.strip():
                rows.append(raw)
    return rows


def validate_historical_v6_hard32_artifacts(
    *,
    dataset_file: Path,
    selection_file: Path,
) -> dict[str, Any]:
    dataset = _historical_exact_path(
        dataset_file,
        expected=HISTORICAL_V6_OFFICIAL_VAL,
        description="official scene-v4 validation dataset",
    )
    selection = _historical_exact_path(
        selection_file,
        expected=HISTORICAL_V6_HARD32_SELECTION,
        description="Hard32 source-index selection",
    )
    holdout = _historical_exact_path(
        HISTORICAL_V6_HARD32_HOLDOUT,
        expected=HISTORICAL_V6_HARD32_HOLDOUT,
        description="frozen Hard32 holdout",
    )
    bindings = {
        "dataset": _historical_artifact_binding(
            dataset,
            description="official scene-v4 validation dataset",
            expected_sha256=OFFICIAL_SCENE_V4_VAL_SHA256,
        ),
        "selection": _historical_artifact_binding(
            selection,
            description="Hard32 source-index selection",
            expected_sha256=HARD32_SELECTION_SHA256,
        ),
        "holdout": _historical_artifact_binding(
            holdout,
            description="frozen Hard32 holdout",
            expected_sha256=HARD32_HOLDOUT_SHA256,
        ),
    }
    raw_selection = selection.read_text(encoding="utf-8")
    row_indices, expected_hashes = parse_selection_manifest(raw_selection)
    if row_indices != list(HARD32_ROW_INDICES):
        raise ValueError(
            "Historical V6 Hard32 selection source indices differ from the frozen order"
        )
    if expected_hashes != HARD32_ROW_HASHES:
        raise ValueError(
            "Historical V6 Hard32 selection row hashes differ from the frozen rows"
        )
    selection_dataset = parse_selection_dataset_contract(raw_selection)
    if selection_dataset is None:
        raise ValueError("Historical V6 Hard32 selection is not dataset-bound")
    validate_selection_dataset_contract(dataset, selection_dataset)

    official_rows = _nonblank_raw_jsonl_rows(dataset)
    if max(row_indices) >= len(official_rows):
        raise ValueError("Historical V6 Hard32 selection exceeds the official dataset")
    selected_rows = [official_rows[index] for index in row_indices]
    if any(
        sha256_text(raw) != expected_hashes[index]
        for index, raw in zip(row_indices, selected_rows, strict=True)
    ):
        raise ValueError(
            "Historical V6 selected official rows do not reproduce the frozen row hashes"
        )
    holdout_rows = _nonblank_raw_jsonl_rows(holdout)
    if len(holdout_rows) != len(HARD32_ROW_INDICES):
        raise ValueError("Historical V6 frozen Hard32 holdout must contain exactly 32 rows")
    if selected_rows != holdout_rows:
        raise ValueError(
            "Historical V6 official-val selection does not byte-reproduce the frozen "
            "Hard32 holdout rows"
        )
    bindings["official_selection_reproduction"] = {
        "rows": len(selected_rows),
        "source_indices": list(row_indices),
        "row_hashes": [expected_hashes[index] for index in row_indices],
        "raw_rows_match_frozen_holdout": True,
        "frozen_holdout_sha256": HARD32_HOLDOUT_SHA256,
    }
    return bindings


def validate_historical_v6_run_preflight(
    args: argparse.Namespace,
    *,
    conditions: list[str],
) -> dict[str, Any] | None:
    if args.evaluation_contract != HISTORICAL_V6_HARD32_CONTRACT:
        return None
    if conditions != list(HISTORICAL_V6_HARD32_CONDITIONS):
        raise ValueError(
            f"{HISTORICAL_V6_HARD32_CONTRACT} requires conditions in exact order: "
            + ",".join(HISTORICAL_V6_HARD32_CONDITIONS)
        )
    if args.overwrite:
        raise ValueError("Historical V6 protected evaluation forbids --overwrite")
    if args.row_indices is not None or args.row_indices_file is None:
        raise ValueError(
            "Historical V6 protected evaluation requires the exact source-index "
            "selection file; inline row indices are forbidden"
        )
    output_dir = args.output_dir.expanduser().absolute()
    _reject_symlink_components(output_dir, description="output directory")
    if output_dir.exists():
        raise ValueError(
            "Historical V6 protected evaluation requires a fresh, nonexistent output "
            f"directory: {output_dir}"
        )
    tracked_worktree = require_historical_tracked_worktree_clean()
    base_model = _historical_exact_path(
        Path(args.base_model),
        expected=HISTORICAL_V6_BASE_MODEL,
        description="base model",
        directory=True,
    )
    checkpoint_lineage = validate_historical_v6_checkpoint(args.memory_dir)
    hard32 = validate_historical_v6_hard32_artifacts(
        dataset_file=args.dataset_file,
        selection_file=args.row_indices_file,
    )
    return {
        "checkpoint": checkpoint_lineage,
        "hard32": hard32,
        "base_model": str(base_model),
        "tracked_worktree": tracked_worktree,
        "authorization_scope": HISTORICAL_V6_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
    }


def validate_historical_v6_hard32_contract(
    *,
    row_indices: list[int],
    expected_hashes: dict[int, str],
    selection_dataset_contract: dict[str, str] | None,
    conditions: list[str],
    donor_rule: str,
    max_new_tokens: int,
    normal_fusion_profile: str,
    expected_memory_layer_count: int,
    memory_target_layers: list[int],
    memory_delta_heads: list[str],
    memory_rank: int,
    rwkv_ms_semantics_version: int,
    memory_backend: str,
    selection_manifest_sha256: str | None,
) -> dict[str, Any]:
    if row_indices != list(HARD32_ROW_INDICES):
        raise ValueError(
            "Historical V6 Hard32 contract requires the exact frozen official-val "
            "source indices in order"
        )
    if expected_hashes != HARD32_ROW_HASHES:
        raise ValueError(
            "Historical V6 Hard32 contract requires every frozen official-val row hash"
        )
    if selection_manifest_sha256 != HARD32_SELECTION_SHA256:
        raise ValueError(
            "Historical V6 Hard32 contract requires the exact source-index selection file"
        )
    if selection_dataset_contract is None:
        raise ValueError(
            "Historical V6 Hard32 contract requires a dataset-bound selection manifest"
        )
    if selection_dataset_contract.get("split") != "val":
        raise ValueError("Historical V6 Hard32 contract requires split=val")
    if selection_dataset_contract.get("sha256") != OFFICIAL_SCENE_V4_VAL_SHA256:
        raise ValueError(
            "Historical V6 Hard32 contract requires the exact official scene-v4 val file"
        )
    declared_dataset = Path(selection_dataset_contract.get("path", "")).expanduser()
    if declared_dataset.resolve() != HISTORICAL_V6_OFFICIAL_VAL.expanduser().resolve():
        raise ValueError(
            "Historical V6 Hard32 contract selection declares a different dataset path"
        )
    if conditions != list(HISTORICAL_V6_HARD32_CONDITIONS):
        raise ValueError(
            f"{HISTORICAL_V6_HARD32_CONTRACT} requires conditions in exact order: "
            + ",".join(HISTORICAL_V6_HARD32_CONDITIONS)
        )
    if donor_rule != DONOR_RULE_CYCLIC:
        raise ValueError(
            "Historical V6 Hard32 contract requires the inert default donor rule; "
            "donor conditions are forbidden"
        )
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(
            "Historical V6 Hard32 contract requires max_new_tokens="
            f"{DEFAULT_MAX_NEW_TOKENS}"
        )
    if normal_fusion_profile != "native":
        raise ValueError(
            "Historical V6 Hard32 contract requires normal_fusion_profile=native"
        )
    if expected_memory_layer_count != HISTORICAL_V6_EXPECTED_LAYER_COUNT:
        raise ValueError(
            "Historical V6 Hard32 contract requires expected_memory_layer_count="
            f"{HISTORICAL_V6_EXPECTED_LAYER_COUNT}"
        )
    if memory_target_layers != list(range(HISTORICAL_V6_EXPECTED_LAYER_COUNT)):
        raise ValueError(
            "Historical V6 Hard32 contract requires checkpoint target_layers=0..23"
        )
    if memory_delta_heads != ["q", "o"]:
        raise ValueError(
            "Historical V6 Hard32 contract requires checkpoint delta_heads=q,o"
        )
    if memory_rank != 4:
        raise ValueError("Historical V6 Hard32 contract requires checkpoint rank=4")
    if rwkv_ms_semantics_version != 2:
        raise ValueError(
            "Historical V6 Hard32 contract requires checkpoint "
            "rwkv_ms_semantics_version=2"
        )
    if memory_backend != "rwkv_ms":
        raise ValueError(
            "Historical V6 Hard32 contract requires checkpoint memory_backend=rwkv_ms"
        )
    return {
        "name": HISTORICAL_V6_HARD32_CONTRACT,
        "phase": "historical_checkpoint_diagnostic",
        "split": "val",
        "task": TASK_NAME,
        "rows": len(HARD32_ROW_INDICES),
        "conditions": list(HISTORICAL_V6_HARD32_CONDITIONS),
        "donor_conditions": [],
        "semantic_nll_conditions": [],
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": HISTORICAL_V6_EXPECTED_LAYER_COUNT,
        "memory_target_layers": list(range(HISTORICAL_V6_EXPECTED_LAYER_COUNT)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "official_dataset_revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
        "official_dataset_path": str(HISTORICAL_V6_OFFICIAL_VAL.resolve()),
        "official_dataset_sha256": OFFICIAL_SCENE_V4_VAL_SHA256,
        "authoritative_selection_path": str(
            HISTORICAL_V6_HARD32_SELECTION.resolve()
        ),
        "authoritative_selection_sha256": HARD32_SELECTION_SHA256,
        "authoritative_holdout_path": str(HISTORICAL_V6_HARD32_HOLDOUT.resolve()),
        "authoritative_holdout_sha256": HARD32_HOLDOUT_SHA256,
        "checkpoint_path": str(HISTORICAL_V6_CHECKPOINT.resolve()),
        "checkpoint_artifact_sha256": dict(
            HISTORICAL_V6_CHECKPOINT_ARTIFACT_SHA256
        ),
        "checkpoint_step": HISTORICAL_V6_EXPECTED_GLOBAL_STEP,
        "lineage_status": "artifact_identity_only_pre_receipt",
        "lineage_limitation": HISTORICAL_V6_LINEAGE_LIMITATION,
        "authorization_scope": HISTORICAL_V6_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "checkpoint_selection_authorized": False,
    }


def validate_scene_hard_failure_contract(
    *,
    contract: str,
    row_indices: list[int],
    expected_hashes: dict[int, str],
    selection_dataset_contract: dict[str, str] | None,
    conditions: list[str],
    donor_rule: str,
    max_new_tokens: int,
    normal_fusion_profile: str,
    expected_memory_layer_count: int,
    memory_target_layers: list[int],
    memory_delta_heads: list[str],
    memory_rank: int,
    rwkv_ms_semantics_version: int,
    memory_backend: str,
    memory_dir: Path,
    selection_manifest_sha256: str | None,
    train_source: Mapping[str, Any] | None,
    train_selection_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if contract not in {
        SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        SCENE_HARD_FAILURE_HARD32_CONTRACT,
    }:
        raise ValueError(f"Unsupported focused hard-failure contract: {contract}")
    if conditions != list(SCENE_FOCUSED_CONDITIONS):
        raise ValueError(
            f"{contract} requires conditions in exact order: "
            + ",".join(SCENE_FOCUSED_CONDITIONS)
        )
    if donor_rule != DONOR_RULE_LENGTH_MATCHED:
        raise ValueError(f"{contract} requires donor_rule={DONOR_RULE_LENGTH_MATCHED}")
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(
            f"{contract} requires max_new_tokens={DEFAULT_MAX_NEW_TOKENS}"
        )
    if normal_fusion_profile != "native":
        raise ValueError(f"{contract} requires normal_fusion_profile=native")
    if expected_memory_layer_count != 42 or memory_target_layers != list(range(42)):
        raise ValueError(f"{contract} requires checkpoint target_layers=0..41")
    if memory_delta_heads != ["q", "o"]:
        raise ValueError(f"{contract} requires checkpoint delta_heads=q,o")
    if memory_rank != 4:
        raise ValueError(f"{contract} requires checkpoint rank=4")
    if rwkv_ms_semantics_version != 2 or memory_backend != "rwkv_ms":
        raise ValueError(f"{contract} requires RWKV-MS semantics v2")

    memory_dir = memory_dir.expanduser().resolve()
    checkpoint = {
        "memory_dir": str(memory_dir),
        "adapter_sha256": sha256_file(memory_dir / "delta_mem_adapter.pt"),
        "config_sha256": sha256_file(memory_dir / "delta_mem_config.json"),
    }
    if contract == SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT:
        if train_source is None or train_selection_authorization is not None:
            raise ValueError("Focused train-overfit requires only the bound train source")
        source_hashes = train_source.get("expected_row_hashes")
        if not isinstance(source_hashes, Mapping):
            raise ValueError("Focused train source row hashes are missing")
        normalized_hashes = {int(index): str(digest) for index, digest in source_hashes.items()}
        if row_indices != list(range(SCENE_HARD_FAILURE_ROWS)):
            raise ValueError("Focused train-overfit requires exact local rows 0..31")
        if expected_hashes != normalized_hashes:
            raise ValueError("Focused train-overfit row hashes differ from source manifest")
        if selection_dataset_contract is not None:
            raise ValueError("Focused train-overfit forbids a validation selection manifest")
        return {
            "name": contract,
            "phase": "train_only_overfit_proof",
            "split": "train",
            "task": TASK_NAME,
            "rows": SCENE_HARD_FAILURE_ROWS,
            "conditions": list(conditions),
            "donor_rule": donor_rule,
            "max_new_tokens": max_new_tokens,
            "normal_fusion_profile": normal_fusion_profile,
            "checkpoint": checkpoint,
            "train_source": dict(train_source),
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        }

    if train_source is not None or train_selection_authorization is None:
        raise ValueError(
            "Focused Hard32 requires only a passing train-only selection authorization"
        )
    if row_indices != list(HARD32_ROW_INDICES) or expected_hashes != HARD32_ROW_HASHES:
        raise ValueError("Focused Hard32 requires the frozen authoritative 32 rows")
    if selection_manifest_sha256 != HARD32_SELECTION_SHA256:
        raise ValueError("Focused Hard32 requires the authoritative selection manifest")
    if (
        selection_dataset_contract is None
        or selection_dataset_contract.get("split") != "val"
        or selection_dataset_contract.get("sha256") != OFFICIAL_SCENE_V4_VAL_SHA256
    ):
        raise ValueError("Focused Hard32 requires the official scene-v4 validation file")
    selected_checkpoint = train_selection_authorization.get("selected_checkpoint")
    selected_artifacts = (
        selected_checkpoint.get("artifacts")
        if isinstance(selected_checkpoint, Mapping)
        else None
    )
    consumption = train_selection_authorization.get("authorization_consumption")
    if (
        not isinstance(selected_checkpoint, Mapping)
        or not isinstance(selected_artifacts, Mapping)
        or selected_checkpoint.get("path") != str(memory_dir)
        or selected_artifacts.get("delta_mem_adapter.pt")
        != checkpoint["adapter_sha256"]
        or selected_artifacts.get("delta_mem_config.json")
        != checkpoint["config_sha256"]
        or not isinstance(consumption, Mapping)
        or consumption.get("schema")
        != SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA
        or not isinstance(consumption.get("path"), str)
        or not isinstance(consumption.get("file_sha256"), str)
        or not isinstance(consumption.get("claim_sha256"), str)
        or not isinstance(
            train_selection_authorization.get("hard32_output_dir"), str
        )
    ):
        raise ValueError("Focused Hard32 checkpoint authorization differs")
    return {
        "name": contract,
        "phase": "single_authorized_hard32",
        "split": "val",
        "task": TASK_NAME,
        "rows": len(HARD32_ROW_INDICES),
        "conditions": list(conditions),
        "donor_rule": donor_rule,
        "max_new_tokens": max_new_tokens,
        "normal_fusion_profile": normal_fusion_profile,
        "checkpoint": checkpoint,
        "official_dataset_revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
        "official_dataset_sha256": OFFICIAL_SCENE_V4_VAL_SHA256,
        "authoritative_selection_sha256": HARD32_SELECTION_SHA256,
        "authoritative_holdout_sha256": HARD32_HOLDOUT_SHA256,
        "authoritative_pair_manifest_sha256": HARD32_PAIR_MANIFEST_SHA256,
        "train_selection_authorization": dict(train_selection_authorization),
        "authorization_scope": SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def validate_scene_v6_matched_donor_contract(
    *,
    contract: str,
    row_indices: list[int],
    expected_hashes: dict[int, str],
    selection_dataset_contract: dict[str, str] | None,
    conditions: list[str],
    donor_rule: str,
    max_new_tokens: int,
    normal_fusion_profile: str,
    expected_memory_layer_count: int,
    memory_target_layers: list[int],
    memory_delta_heads: list[str],
    memory_rank: int,
    rwkv_ms_semantics_version: int,
    memory_backend: str,
    selection_manifest_sha256: str | None = None,
    hard32_receipt_authorization: dict[str, Any] | None = None,
    scene_v8_train32_authorization: dict[str, Any] | None = None,
    scene_v14_candidate_authorization: dict[str, Any] | None = None,
    scene_v15_candidate_authorization: dict[str, Any] | None = None,
    focused_train_source: dict[str, Any] | None = None,
    focused_train_selection_authorization: dict[str, Any] | None = None,
    memory_dir: Path | None = None,
) -> dict[str, Any]:
    if contract not in EVALUATION_CONTRACTS:
        raise ValueError(f"Unsupported scene-state evaluation contract: {contract}")
    if (
        scene_v14_candidate_authorization is not None
        and contract != SCENE_V14_HARD32_CONTRACT
    ):
        raise ValueError(
            "V14 candidate-lock authorization is restricted to "
            f"{SCENE_V14_HARD32_CONTRACT}"
        )
    if (
        scene_v15_candidate_authorization is not None
        and contract != SCENE_V15_HARD32_CONTRACT
    ):
        raise ValueError(
            "V15 candidate-lock authorization is restricted to "
            f"{SCENE_V15_HARD32_CONTRACT}"
        )
    if contract == SCENE_V14_HARD32_CONTRACT and (
        scene_v8_train32_authorization is not None
        or scene_v15_candidate_authorization is not None
        or hard32_receipt_authorization is not None
    ):
        raise ValueError("V14 Hard32 forbids mixed or broader benchmark authorization")
    if contract == SCENE_V15_HARD32_CONTRACT and (
        scene_v8_train32_authorization is not None
        or scene_v14_candidate_authorization is not None
        or hard32_receipt_authorization is not None
    ):
        raise ValueError("V15 Hard32 forbids mixed or broader benchmark authorization")
    if contract == "generic":
        return {"name": "generic", "phase": "focused_diagnostic"}
    if contract in {
        SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        SCENE_HARD_FAILURE_HARD32_CONTRACT,
    }:
        if memory_dir is None:
            raise ValueError("Focused hard-failure contract requires memory_dir")
        return validate_scene_hard_failure_contract(
            contract=contract,
            row_indices=row_indices,
            expected_hashes=expected_hashes,
            selection_dataset_contract=selection_dataset_contract,
            conditions=conditions,
            donor_rule=donor_rule,
            max_new_tokens=max_new_tokens,
            normal_fusion_profile=normal_fusion_profile,
            expected_memory_layer_count=expected_memory_layer_count,
            memory_target_layers=memory_target_layers,
            memory_delta_heads=memory_delta_heads,
            memory_rank=memory_rank,
            rwkv_ms_semantics_version=rwkv_ms_semantics_version,
            memory_backend=memory_backend,
            memory_dir=memory_dir,
            selection_manifest_sha256=selection_manifest_sha256,
            train_source=focused_train_source,
            train_selection_authorization=focused_train_selection_authorization,
        )
    if contract == HISTORICAL_V6_HARD32_CONTRACT:
        return validate_historical_v6_hard32_contract(
            row_indices=row_indices,
            expected_hashes=expected_hashes,
            selection_dataset_contract=selection_dataset_contract,
            conditions=conditions,
            donor_rule=donor_rule,
            max_new_tokens=max_new_tokens,
            normal_fusion_profile=normal_fusion_profile,
            expected_memory_layer_count=expected_memory_layer_count,
            memory_target_layers=memory_target_layers,
            memory_delta_heads=memory_delta_heads,
            memory_rank=memory_rank,
            rwkv_ms_semantics_version=rwkv_ms_semantics_version,
            memory_backend=memory_backend,
            selection_manifest_sha256=selection_manifest_sha256,
        )
    identity_hard32 = is_identity_hard32_contract(contract)
    if identity_hard32:
        if row_indices != list(HARD32_ROW_INDICES):
            raise ValueError(
                f"{contract} requires the authoritative fixed 32-row "
                "validation selection in source order"
            )
        if expected_hashes != HARD32_ROW_HASHES:
            raise ValueError(
                f"{contract} requires every authoritative fixed row hash"
            )
        if selection_manifest_sha256 != HARD32_SELECTION_SHA256:
            raise ValueError(
                f"{contract} requires the authoritative selection manifest"
            )
    elif row_indices != list(range(170)):
        raise ValueError(
            "scene_v6_matched_donor_validation requires all 170 official validation "
            "rows in source order"
        )
    if contract == "scene_v6_matched_donor_validation" and set(expected_hashes) != set(
        row_indices
    ):
        raise ValueError(
            "scene_v6_matched_donor_validation requires a source hash for every row"
        )
    if selection_dataset_contract is None:
        raise ValueError(
            "scene_v6_matched_donor_validation requires a dataset-bound selection manifest"
        )
    if selection_dataset_contract.get("sha256") != OFFICIAL_SCENE_V4_VAL_SHA256:
        raise ValueError(
            "scene_v6_matched_donor_validation requires the official scene-v4 val "
            f"file at revision {OFFICIAL_SCENE_V4_DATASET_REVISION}"
        )
    expected_conditions = (
        list(SCENE_V15_HARD32_CONDITIONS)
        if contract == SCENE_V15_HARD32_CONTRACT
        else (
            list(CONDITIONS)
            if identity_hard32
            else ["state_only", "state_only_donor"]
        )
    )
    if conditions != expected_conditions:
        raise ValueError(
            f"{contract} requires conditions in exact order: "
            + ",".join(expected_conditions)
        )
    if donor_rule != DONOR_RULE_LENGTH_MATCHED:
        raise ValueError(
            "scene_v6_matched_donor_validation requires donor_rule="
            f"{DONOR_RULE_LENGTH_MATCHED}"
        )
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(
            "scene_v6_matched_donor_validation requires max_new_tokens="
            f"{DEFAULT_MAX_NEW_TOKENS}"
        )
    if normal_fusion_profile != "native":
        raise ValueError(
            "scene_v6_matched_donor_validation requires normal_fusion_profile=native"
        )
    if expected_memory_layer_count != 42:
        raise ValueError(
            "scene_v6_matched_donor_validation requires "
            "expected_memory_layer_count=42"
        )
    if memory_target_layers != list(range(42)):
        raise ValueError(
            "scene_v6_matched_donor_validation requires checkpoint target_layers=0..41"
        )
    if memory_delta_heads != ["q", "o"]:
        raise ValueError(
            "scene_v6_matched_donor_validation requires checkpoint delta_heads=q,o"
        )
    if memory_rank != 4:
        raise ValueError(
            "scene_v6_matched_donor_validation requires checkpoint rank=4"
        )
    if rwkv_ms_semantics_version != 2:
        raise ValueError(
            "scene_v6_matched_donor_validation requires checkpoint "
            "rwkv_ms_semantics_version=2"
        )
    if memory_backend != "rwkv_ms":
        raise ValueError(
            "scene_v6_matched_donor_validation requires checkpoint "
            "memory_backend=rwkv_ms"
        )
    if (
        contract == "scene_v6_matched_donor_validation"
        and hard32_receipt_authorization is None
    ):
        raise ValueError(
            "scene_v6_matched_donor_validation requires a passed hard32 receipt "
            "binding this checkpoint"
        )
    if contract == SCENE_V8_HARD32_CONTRACT and scene_v8_train32_authorization is None:
        raise ValueError(
            f"{SCENE_V8_HARD32_CONTRACT} requires a passing V8 Train32 gate "
            "receipt binding this checkpoint"
        )
    if (
        contract == SCENE_V14_HARD32_CONTRACT
        and scene_v14_candidate_authorization is None
    ):
        raise ValueError(
            f"{SCENE_V14_HARD32_CONTRACT} requires the exact V14 candidate lock, "
            "diagnostic receipt, launch receipt, and completion receipt"
        )
    if contract == SCENE_V14_HARD32_CONTRACT and (
        scene_v14_candidate_authorization.get("authorization_kind")
        != SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
        or scene_v14_candidate_authorization.get("scope")
        != SCENE_V14_HARD32_AUTHORIZATION_SCOPE
        or scene_v14_candidate_authorization.get("hard32_authorized") is not True
        or scene_v14_candidate_authorization.get("full170_authorized") is not False
        or scene_v14_candidate_authorization.get("test_authorized") is not False
        or scene_v14_candidate_authorization.get("other_benchmarks_authorized")
        is not False
    ):
        raise ValueError("V14 Hard32 authorization scope differs")
    if (
        contract == SCENE_V15_HARD32_CONTRACT
        and scene_v15_candidate_authorization is None
    ):
        raise ValueError(
            f"{SCENE_V15_HARD32_CONTRACT} requires the complete post-save "
            "Train32 selection and exact selected-checkpoint lock"
        )
    if contract == SCENE_V15_HARD32_CONTRACT and (
        scene_v15_candidate_authorization.get("authorization_kind")
        != SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND
        or scene_v15_candidate_authorization.get("scope")
        != SCENE_V15_HARD32_AUTHORIZATION_SCOPE
        or scene_v15_candidate_authorization.get("hard32_authorized") is not True
        or scene_v15_candidate_authorization.get("full170_authorized") is not False
        or scene_v15_candidate_authorization.get("test_authorized") is not False
        or scene_v15_candidate_authorization.get("other_benchmarks_authorized")
        is not False
    ):
        raise ValueError("V15 Hard32 authorization scope differs")
    fixed_scope_hard32 = is_fixed_scope_hard32_contract(contract)
    return {
        "name": contract,
        "phase": "validation_selection",
        "split": "val",
        "task": TASK_NAME,
        "rows": 32 if identity_hard32 else 170,
        "conditions": list(conditions),
        "donor_rule": donor_rule,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(range(42)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "official_dataset_revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
        "official_dataset_sha256": OFFICIAL_SCENE_V4_VAL_SHA256,
        "test_selection_forbidden": True,
        "authoritative_selection_sha256": (
            HARD32_SELECTION_SHA256
            if identity_hard32
            else None
        ),
        "authoritative_holdout_sha256": (
            HARD32_HOLDOUT_SHA256
            if identity_hard32
            else None
        ),
        "authoritative_pair_manifest_sha256": (
            HARD32_PAIR_MANIFEST_SHA256
            if identity_hard32
            else None
        ),
        "gate_requirements": (
            dict(SCENE_V15_HARD32_GATE_POLICY)
            if contract == SCENE_V15_HARD32_CONTRACT
            else (
                dict(HARD32_GATE_REQUIREMENTS)
                if identity_hard32
                else None
            )
        ),
        "hard32_receipt_authorization": hard32_receipt_authorization,
        "scene_v8_train32_authorization": scene_v8_train32_authorization,
        "scene_v14_candidate_authorization": scene_v14_candidate_authorization,
        "scene_v15_candidate_authorization": scene_v15_candidate_authorization,
        "authorization_scope": (
            SCENE_V14_HARD32_AUTHORIZATION_SCOPE
            if fixed_scope_hard32
            else None
        ),
        "full170_authorized": False if fixed_scope_hard32 else None,
        "test_authorized": False if fixed_scope_hard32 else None,
        "other_benchmarks_authorized": (
            False if fixed_scope_hard32 else None
        ),
    }


def read_selection(
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, str], dict[str, str] | None]:
    if args.row_indices is not None:
        return parse_row_indices(args.row_indices), {}, None
    raw = args.row_indices_file.read_text(encoding="utf-8")
    indices, expected_hashes = parse_selection_manifest(raw)
    return indices, expected_hashes, parse_selection_dataset_contract(raw)


def resolve_validation_dataset_file(dataset_file: Path) -> Path:
    resolved = dataset_file.expanduser().resolve()
    if resolved.name != "val.jsonl":
        raise ValueError(
            "Focused scene-state evaluation requires the official val.jsonl; "
            "train inference belongs to the dedicated producer and test is unsupported"
        )
    return resolved


def resolve_scene_dataset_file(dataset_file: Path, *, evaluation_contract: str) -> Path:
    if evaluation_contract == SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT:
        resolved = dataset_file.expanduser().resolve()
        if resolved.name != "train.jsonl":
            raise ValueError(
                "Focused train-overfit requires the source-manifest-bound train.jsonl"
            )
        _reject_focused_protected_path(resolved, description="dataset")
        return resolved
    return resolve_validation_dataset_file(dataset_file)


def validate_selection_dataset_contract(
    dataset_file: Path,
    contract: dict[str, str] | None,
) -> None:
    if contract is None:
        return
    declared_path = Path(contract["path"]).expanduser().resolve()
    if declared_path != dataset_file:
        raise ValueError(
            "Selection manifest dataset path differs from --dataset-file: "
            f"declared={declared_path} actual={dataset_file}"
        )
    actual_sha256 = sha256_file(dataset_file)
    if contract["sha256"] != actual_sha256:
        raise ValueError(
            "Selection manifest dataset SHA-256 differs from --dataset-file: "
            f"declared={contract['sha256']} actual={actual_sha256}"
        )


def load_selected_rows(
    dataset_file: Path,
    row_indices: list[int],
    *,
    expected_hashes: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    expected_hashes = expected_hashes or {}
    requested = set(row_indices)
    selected: dict[int, dict[str, Any]] = {}
    with dataset_file.open("r", encoding="utf-8") as handle:
        source_index = 0
        for raw_line in handle:
            if not raw_line.strip():
                continue
            if source_index in requested:
                row = json.loads(raw_line)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) != 3:
                    raise ValueError(
                        f"Expected a three-message scene row at source index {source_index}"
                    )
                roles = [message.get("role") for message in messages]
                if roles != ["system", "user", "assistant"]:
                    raise ValueError(
                        f"Unexpected message roles at source index {source_index}: {roles}"
                    )
                gold = extract_json(str(messages[-1].get("content", "")))
                perfect = score_prediction("scene", gold, gold)
                if gold is None or not perfect["schema_valid"] or perfect["fp"] or perfect["fn"]:
                    raise ValueError(f"Invalid scene gold at source index {source_index}")
                row_sha256 = sha256_text(raw_line.rstrip("\n"))
                expected_hash = expected_hashes.get(source_index)
                if expected_hash is not None and row_sha256 != expected_hash:
                    raise ValueError(
                        f"Selection row hash differs at source index {source_index}"
                    )
                prime_messages = messages[:-1]
                selected[source_index] = {
                    "source_index": source_index,
                    "messages": prime_messages,
                    "gold": gold,
                    "gold_content": str(messages[-1].get("content", "")),
                    "row_sha256": row_sha256,
                    "prime_messages_sha256": fingerprint_payload_sha256(
                        {"messages": prime_messages}
                    ),
                }
            source_index += 1
    missing = [index for index in row_indices if index not in selected]
    if missing:
        raise IndexError(f"Selected row indices are outside the dataset: {missing}")
    return [selected[index] for index in row_indices]


def build_cyclic_donor_mapping(
    samples: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if len(samples) < 2:
        raise ValueError("state_only_donor requires at least two selected rows")
    source_indices = [sample["source_index"] for sample in samples]
    row_hashes = [sample["row_sha256"] for sample in samples]
    prime_hashes = [sample["prime_messages_sha256"] for sample in samples]
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("Donor mapping requires unique selected source indices")
    if len(set(row_hashes)) != len(row_hashes):
        raise ValueError("Donor mapping requires unique selected row hashes")
    if len(set(prime_hashes)) != len(prime_hashes):
        raise ValueError("Donor mapping requires unique priming prompts")
    return {
        int(sample["source_index"]): samples[(ordinal + 1) % len(samples)]
        for ordinal, sample in enumerate(samples)
    }


def annotate_write_token_counts(samples: list[dict[str, Any]], tokenizer) -> None:
    for sample in samples:
        rendered = apply_chat_template(
            tokenizer,
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(rendered, add_special_tokens=False)
        sample["write_token_count"] = len(encoded.input_ids)


def build_length_matched_label_distinct_donor_mapping(
    samples: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Pair rows by write length while forbidding equal boundary labels."""

    if len(samples) < 2 or len(samples) % 2:
        raise ValueError(
            f"{DONOR_RULE_LENGTH_MATCHED} requires a positive even row count"
        )
    for sample in samples:
        count = sample.get("write_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Length-matched donors require positive write_token_count values")

    ordered = sorted(
        samples,
        key=lambda sample: (
            int(sample["write_token_count"]),
            str(sample["row_sha256"]),
        ),
    )

    def pair_remaining(
        remaining: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
        if not remaining:
            return []
        current = remaining[0]
        candidates = sorted(
            (
                candidate_index
                for candidate_index in range(1, len(remaining))
                if remaining[candidate_index]["gold"] != current["gold"]
            ),
            key=lambda candidate_index: (
                abs(
                    int(remaining[candidate_index]["write_token_count"])
                    - int(current["write_token_count"])
                ),
                int(remaining[candidate_index]["write_token_count"]),
                str(remaining[candidate_index]["row_sha256"]),
            ),
        )
        for candidate_index in candidates:
            candidate = remaining[candidate_index]
            tail = remaining[1:candidate_index] + remaining[candidate_index + 1 :]
            paired_tail = pair_remaining(tail)
            if paired_tail is not None:
                return [(current, candidate), *paired_tail]
        return None

    pairs = pair_remaining(ordered)
    if pairs is None:
        raise ValueError(
            "No complete label-distinct donor pairing exists for the selected rows"
        )
    mapping: dict[int, dict[str, Any]] = {}
    for left, right in pairs:
        mapping[int(left["source_index"])] = right
        mapping[int(right["source_index"])] = left
    if len(mapping) != len(samples):
        raise RuntimeError("Length-matched donor pairing did not cover every selected row")
    return mapping


def build_frozen_hard32_donor_mapping(
    samples: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Materialize and exhaustively validate the protected global matching."""

    if json.dumps(HARD32_FROZEN_DONOR_PAIRS, separators=(",", ":")) != json.dumps(
        [list(pair) for pair in HARD32_FROZEN_DONOR_PAIRS], separators=(",", ":")
    ):
        raise RuntimeError("Hard32 donor pair serialization is not canonical")
    pair_list_sha256 = sha256_text(
        json.dumps(HARD32_FROZEN_DONOR_PAIRS, separators=(",", ":"))
    )
    if pair_list_sha256 != HARD32_FROZEN_DONOR_PAIRS_SHA256:
        raise RuntimeError("Hard32 frozen donor pair-list hash differs")

    by_source = {int(sample["source_index"]): sample for sample in samples}
    if tuple(by_source) != HARD32_ROW_INDICES:
        raise ValueError("Hard32 frozen donor mapping requires the protected row order")
    if any(
        sample["row_sha256"] != HARD32_ROW_HASHES[source_index]
        for source_index, sample in by_source.items()
    ):
        raise ValueError("Hard32 frozen donor mapping row hashes differ")

    directed: dict[int, int] = {}
    for left, right in HARD32_FROZEN_DONOR_PAIRS:
        if left in directed or right in directed or left == right:
            raise RuntimeError("Hard32 frozen donor pairs are not disjoint")
        if left not in by_source or right not in by_source:
            raise RuntimeError("Hard32 frozen donor pair references an unknown row")
        directed[left] = right
        directed[right] = left
    if set(directed) != set(by_source):
        raise RuntimeError("Hard32 frozen donor pairs do not cover the selection")

    mapping = {source: by_source[donor] for source, donor in directed.items()}
    deltas: list[int] = []
    strata = {name: 0 for name in HARD32_FROZEN_DONOR_STRATUM_ROWS}
    for left, right in HARD32_FROZEN_DONOR_PAIRS:
        source = by_source[left]
        donor = by_source[right]
        if source["gold"] == donor["gold"]:
            raise RuntimeError("Hard32 frozen donor pair has an identical gold label")
        source_tokens = source.get("write_token_count")
        donor_tokens = donor.get("write_token_count")
        if (
            isinstance(source_tokens, bool)
            or not isinstance(source_tokens, int)
            or isinstance(donor_tokens, bool)
            or not isinstance(donor_tokens, int)
        ):
            raise ValueError("Hard32 frozen donors require tokenizer-derived lengths")
        deltas.append(abs(source_tokens - donor_tokens))
        source_cardinality = len(strict_gold_boundaries(source["gold"]))
        donor_cardinality = len(strict_gold_boundaries(donor["gold"]))
        if source_cardinality == 0 or donor_cardinality == 0:
            stratum = "empty_vs_nonempty"
        elif source_cardinality == donor_cardinality:
            stratum = "nonempty_same_cardinality"
        else:
            stratum = "nonempty_different_cardinality"
        strata[stratum] += 2
    if max(deltas) != HARD32_FROZEN_DONOR_MAX_WRITE_TOKEN_DIFFERENCE:
        raise ValueError("Hard32 frozen donor maximum write-length delta differs")
    if sum(deltas) != HARD32_FROZEN_DONOR_TOTAL_WRITE_TOKEN_DIFFERENCE:
        raise ValueError("Hard32 frozen donor total write-length delta differs")
    if strata != HARD32_FROZEN_DONOR_STRATUM_ROWS:
        raise ValueError("Hard32 frozen donor cardinality strata differ")

    mapping_rows = donor_mapping_fingerprint_rows(samples, mapping)
    mapping_rows_sha256 = sha256_text(
        json.dumps(mapping_rows, sort_keys=True, separators=(",", ":"))
    )
    if mapping_rows_sha256 != HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256:
        raise ValueError("Hard32 frozen donor mapping fingerprint differs")
    return mapping


def materialize_focused_train_donor_mapping(
    samples: list[dict[str, Any]],
    donor_by_source_index: Mapping[int, int],
) -> dict[int, dict[str, Any]]:
    by_source = {int(sample["source_index"]): sample for sample in samples}
    expected_sources = set(range(SCENE_HARD_FAILURE_ROWS))
    if set(by_source) != expected_sources or set(donor_by_source_index) != expected_sources:
        raise ValueError("Focused train donor mapping does not cover exact local rows 0..31")
    mapping = {
        source_index: by_source[int(donor_by_source_index[source_index])]
        for source_index in range(SCENE_HARD_FAILURE_ROWS)
    }
    for source_index, donor in mapping.items():
        if (
            int(donor["source_index"]) == source_index
            or donor["gold"] == by_source[source_index]["gold"]
            or int(mapping[int(donor["source_index"])]["source_index"]) != source_index
        ):
            raise ValueError("Focused train donor mapping identity or reciprocity differs")
    return mapping


def build_deterministic_shuffled_mapping(
    samples: list[dict[str, Any]],
    donor_by_index: Mapping[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Build an independent, label-distinct bijection excluding locked donors."""

    by_source = {int(sample["source_index"]): sample for sample in samples}
    if len(by_source) != len(samples) or set(donor_by_index) != set(by_source):
        raise ValueError("Shuffled mapping requires complete unique source and donor rows")
    candidates: dict[int, list[int]] = {}
    for source_index, sample in by_source.items():
        locked_donor_index = int(donor_by_index[source_index]["source_index"])
        eligible = [
            candidate_index
            for candidate_index, candidate in by_source.items()
            if candidate_index not in {source_index, locked_donor_index}
            and candidate["gold"] != sample["gold"]
        ]
        candidates[source_index] = sorted(
            eligible,
            key=lambda candidate_index: (
                sha256_text(
                    "rwkv_ms_scene_state_only_shuffled.v1\0"
                    + str(sample["row_sha256"])
                    + "\0"
                    + str(by_source[candidate_index]["row_sha256"])
                ),
                str(by_source[candidate_index]["row_sha256"]),
                candidate_index,
            ),
        )
        if not candidates[source_index]:
            raise ValueError(
                f"No label-distinct non-donor shuffle candidate for source {source_index}"
            )

    donor_owner: dict[int, int] = {}

    def assign(source_index: int, visited: set[int]) -> bool:
        for candidate_index in candidates[source_index]:
            if candidate_index in visited:
                continue
            visited.add(candidate_index)
            previous_owner = donor_owner.get(candidate_index)
            if previous_owner is None or assign(previous_owner, visited):
                donor_owner[candidate_index] = source_index
                return True
        return False

    source_order = sorted(
        by_source,
        key=lambda source_index: (
            len(candidates[source_index]),
            str(by_source[source_index]["row_sha256"]),
            source_index,
        ),
    )
    for source_index in source_order:
        if not assign(source_index, set()):
            raise ValueError("No complete deterministic shuffled-state matching exists")
    shuffled_index_by_source = {
        source_index: candidate_index
        for candidate_index, source_index in donor_owner.items()
    }
    if set(shuffled_index_by_source) != set(by_source):
        raise RuntimeError("Shuffled-state matching did not cover every source row")
    mapping = {
        source_index: by_source[shuffled_index_by_source[source_index]]
        for source_index in by_source
    }
    if len({int(sample["source_index"]) for sample in mapping.values()}) != len(samples):
        raise RuntimeError("Shuffled-state matching is not bijective")
    for source_index, shuffled in mapping.items():
        shuffled_index = int(shuffled["source_index"])
        if (
            shuffled_index == source_index
            or shuffled_index == int(donor_by_index[source_index]["source_index"])
            or shuffled["gold"] == by_source[source_index]["gold"]
        ):
            raise RuntimeError("Shuffled-state matching violates identity constraints")
    return mapping


def resolved_condition_protocols(
    conditions: list[str],
    *,
    donor_rule: str,
) -> dict[str, dict[str, Any]]:
    protocols = {condition: copy.deepcopy(CONDITION_PROTOCOLS[condition]) for condition in conditions}
    donor_protocol = protocols.get("state_only_donor")
    if donor_protocol is not None:
        donor_protocol["donor_rule"] = donor_rule
    if donor_rule in {
        DONOR_RULE_LENGTH_MATCHED,
        DONOR_RULE_LENGTH_MATCHED_LEGACY,
    } and donor_protocol is not None:
        donor_protocol.update(
            {
                "rwkv_prime": "write_token_length_matched_label_distinct_validation_row",
                "donor_pool": "selected_validation_rows_paired_without_replacement",
                "description": (
                    "Prime online state from a different-label validation row paired by "
                    "nearest write-token length, discard ordinary KV state, then query "
                    "with the current row's system-only prompt and score current gold."
                ),
            }
        )
    return protocols


def donor_mapping_fingerprint_rows(
    samples: list[dict[str, Any]],
    donor_by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        donor = donor_by_index[int(sample["source_index"])]
        row = {
            "source_index": int(sample["source_index"]),
            "row_sha256": sample["row_sha256"],
            "prime_messages_sha256": sample["prime_messages_sha256"],
            "donor_source_index": int(donor["source_index"]),
            "donor_row_sha256": donor["row_sha256"],
            "donor_prime_messages_sha256": donor["prime_messages_sha256"],
        }
        if "write_token_count" in sample or "write_token_count" in donor:
            source_count = int(sample["write_token_count"])
            donor_count = int(donor["write_token_count"])
            row.update(
                {
                    "write_token_count": source_count,
                    "donor_write_token_count": donor_count,
                    "absolute_write_token_difference": abs(source_count - donor_count),
                }
            )
        if "gold" in sample and "gold" in donor:
            source_cardinality = len(strict_gold_boundaries(sample["gold"]))
            donor_cardinality = len(strict_gold_boundaries(donor["gold"]))
            if source_cardinality == 0 or donor_cardinality == 0:
                cardinality_stratum = "empty_vs_nonempty"
            elif source_cardinality == donor_cardinality:
                cardinality_stratum = "nonempty_same_cardinality"
            else:
                cardinality_stratum = "nonempty_different_cardinality"
            row.update(
                {
                    "source_gold_boundary_count": source_cardinality,
                    "donor_gold_boundary_count": donor_cardinality,
                    "donor_minus_source_gold_cardinality": (
                        donor_cardinality - source_cardinality
                    ),
                    "absolute_gold_cardinality_difference": abs(
                        donor_cardinality - source_cardinality
                    ),
                    "gold_cardinality_stratum": cardinality_stratum,
                }
            )
        rows.append(row)
    return rows


def shuffled_mapping_fingerprint_rows(
    samples: list[dict[str, Any]],
    shuffled_by_index: Mapping[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        shuffled = shuffled_by_index[int(sample["source_index"])]
        rows.append(
            {
                "source_index": int(sample["source_index"]),
                "row_sha256": sample["row_sha256"],
                "prime_messages_sha256": sample["prime_messages_sha256"],
                "shuffled_source_index": int(shuffled["source_index"]),
                "shuffled_row_sha256": shuffled["row_sha256"],
                "shuffled_prime_messages_sha256": shuffled[
                    "prime_messages_sha256"
                ],
            }
        )
    return rows


def resolved_memory_layer_count(memory_dir: Path, requested: int | None) -> int:
    config = json.loads((memory_dir / "delta_mem_config.json").read_text(encoding="utf-8"))
    target_layers = config.get("target_layers")
    if not isinstance(target_layers, list) or not target_layers:
        raise ValueError("Memory config must contain a non-empty target_layers list")
    inferred = len(target_layers)
    if requested is None:
        return inferred
    if requested <= 0:
        raise ValueError("Expected memory layer count must be positive")
    if requested != inferred:
        raise ValueError(
            f"Expected memory layer count {requested} differs from config target count {inferred}"
        )
    return requested


def artifact_identity(root: Path, relative_names: Iterable[str]) -> dict[str, Any]:
    root = root.expanduser().resolve()
    files: list[dict[str, Any]] = []
    for relative_name in sorted(set(relative_names)):
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe model artifact path: {relative_name!r}")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact referenced by config: {path}")
        files.append(
            {
                "relative_path": relative_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"No model artifacts selected under {root}")
    return {
        "files": files,
        "combined_sha256": sha256_text(
            json.dumps(files, sort_keys=True, separators=(",", ":"))
        ),
    }


def base_model_weight_identity(base_model: Path) -> dict[str, Any]:
    root = base_model.expanduser().resolve()
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid model weight map: {index_path}")
        raw_shard_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in raw_shard_names):
            raise ValueError(f"Invalid model shard name in {index_path}")
        shard_names = set(raw_shard_names)
        identity = artifact_identity(root, [index_name, *shard_names])
        return {"layout": "sharded", "index": index_name, **identity}

    safetensors = sorted(path.name for path in root.glob("*.safetensors"))
    weight_files = safetensors or sorted(
        path.name for path in root.glob("pytorch_model*.bin")
    )
    if not weight_files:
        raise FileNotFoundError(f"No local model weight files found under {root}")
    return {"layout": "unsharded", **artifact_identity(root, weight_files)}


def base_model_prompt_identity(base_model: Path) -> dict[str, Any]:
    root = base_model.expanduser().resolve()
    exact_names = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    }
    available = [name for name in exact_names if (root / name).is_file()]
    return artifact_identity(root, available)


def runtime_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for distribution in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def render_and_tokenize(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    device: str,
):
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return rendered, encoded.input_ids.to(device), encoded.attention_mask.to(device)


def scene_boundary_array_char_span(content: str) -> tuple[int, int]:
    decoder = json.JSONDecoder()

    def skip_whitespace(position: int) -> int:
        while position < len(content) and content[position].isspace():
            position += 1
        return position

    position = skip_whitespace(0)
    if position >= len(content) or content[position] != "{":
        raise ValueError("Scene semantic NLL requires a top-level JSON object")
    position += 1
    boundary_span: tuple[int, int] | None = None
    try:
        while True:
            position = skip_whitespace(position)
            if position >= len(content):
                break
            if content[position] == "}":
                position = skip_whitespace(position + 1)
                break
            key, position = decoder.raw_decode(content, position)
            if not isinstance(key, str):
                break
            position = skip_whitespace(position)
            if position >= len(content) or content[position] != ":":
                break
            value_start = skip_whitespace(position + 1)
            value, value_end = decoder.raw_decode(content, value_start)
            if key == "boundaries":
                if boundary_span is not None:
                    break
                if not isinstance(value, list) or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                ):
                    break
                boundary_span = (value_start, value_end)
            position = skip_whitespace(value_end)
            if position < len(content) and content[position] == ",":
                position += 1
                continue
            if position < len(content) and content[position] == "}":
                position = skip_whitespace(position + 1)
                break
            break
    except (json.JSONDecodeError, TypeError, ValueError):
        boundary_span = None
    if boundary_span is None or position != len(content):
        raise ValueError(
            "Scene semantic NLL requires one top-level integer boundaries list"
        )
    return boundary_span


def scene_boundary_decision_token_mask_from_offsets(
    *,
    content: str,
    content_start: int,
    offsets: list[tuple[int, int]],
) -> list[bool]:
    array_start, array_end = scene_boundary_array_char_span(content)
    decision_positions = {
        content_start + index
        for index in range(array_start, array_end)
        if content[index] in "[],0123456789"
    }
    if not decision_positions:
        raise ValueError("Scene semantic NLL found no serialized boundary decisions")
    mask = [
        any(start <= position < end for position in decision_positions)
        if end > start
        else False
        for start, end in offsets
    ]
    if not any(mask):
        raise ValueError("Scene semantic decisions did not align to tokenizer tokens")
    return mask


def rendered_scene_decision_features(tokenizer, sample: dict[str, Any], device: str):
    content = sample.get("gold_content")
    if not isinstance(content, str) or not content:
        raise ValueError("Scene sample is missing canonical assistant gold content")
    messages = [sample["messages"][0], {"role": "assistant", "content": content}]
    sentinel = "__DELTAMEM_SCENE_IDENTITY_GOLD_SENTINEL__"
    while sentinel in content:
        sentinel += "_"
    probe_messages = [dict(message) for message in messages]
    probe_messages[-1]["content"] = sentinel
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    probe_rendered = apply_chat_template(
        tokenizer,
        probe_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not isinstance(probe_rendered, str):
        raise ValueError("Scene semantic NLL chat template must render text")
    if probe_rendered.count(sentinel) != 1:
        raise ValueError("Scene semantic NLL could not locate assistant content")
    prefix, suffix = probe_rendered.split(sentinel)
    if rendered != prefix + content + suffix:
        raise ValueError("Scene semantic NLL chat template transformed gold content")
    prompt_rendered = apply_chat_template(
        tokenizer,
        [sample["messages"][0]],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_rendered, str) or not rendered.startswith(prompt_rendered):
        raise ValueError("Scene semantic NLL read prefix differs from generation prefix")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    raw_offsets = encoded.offset_mapping
    if raw_offsets.ndim != 3 or raw_offsets.size(0) != 1:
        raise ValueError("Scene semantic NLL tokenizer offsets have invalid shape")
    offsets = [(int(start), int(end)) for start, end in raw_offsets[0].tolist()]
    token_mask = scene_boundary_decision_token_mask_from_offsets(
        content=content,
        content_start=len(prefix),
        offsets=offsets,
    )
    selected_positions = [index for index, selected in enumerate(token_mask) if selected]
    if not selected_positions or selected_positions[0] == 0:
        raise ValueError("Scene semantic NLL selected a target without a predictor")
    return {
        "rendered": rendered,
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "selected_positions": selected_positions,
    }


def first_pair_distinguishing_scene_target(
    *,
    source_features: dict[str, Any],
    donor_features: dict[str, Any],
    donor_sample: dict[str, Any],
) -> dict[str, Any]:
    source_ids = source_features["input_ids"][0].detach().cpu().tolist()
    donor_ids = donor_features["input_ids"][0].detach().cpu().tolist()
    source_trace = [
        (position, int(source_ids[position]))
        for position in source_features["selected_positions"]
    ]
    donor_trace = [
        (position, int(donor_ids[position]))
        for position in donor_features["selected_positions"]
    ]
    ordinal = next(
        (
            index
            for index, (source_item, donor_item) in enumerate(
                zip(source_trace, donor_trace)
            )
            if source_item[1] != donor_item[1]
        ),
        min(len(source_trace), len(donor_trace)),
    )
    if ordinal == len(source_trace) == len(donor_trace):
        raise ValueError("Scene semantic donor must have a distinct tokenized label")
    if ordinal >= len(source_trace):
        raise ValueError(
            "Scene source semantic sequence ends before its first donor distinction"
        )
    source_position, source_token_id = source_trace[ordinal]
    donor_item = None if ordinal >= len(donor_trace) else donor_trace[ordinal]
    donor_position = None if donor_item is None else donor_item[0]
    donor_token_id = None if donor_item is None else donor_item[1]

    if donor_position is None or source_position != donor_position:
        raise ValueError(
            "Scene pair target does not have an identical causal-prefix length"
        )
    source_prefix = source_ids[:source_position]
    donor_prefix = donor_ids[:donor_position]
    if source_prefix != donor_prefix:
        raise ValueError(
            "Scene pair target is not the first distinction on the full causal prefix"
        )
    return {
        "target_mode": PAIR_TARGET_DECISION_MASK_MODE,
        "first_differing_semantic_ordinal": ordinal,
        "selected_target_positions": [source_position],
        "selected_target_token_ids": [source_token_id],
        "donor_target_token_ids": [donor_token_id],
        "causal_prefix_sha256": sha256_text(
            json.dumps(source_prefix, separators=(",", ":"))
        ),
        "donor_source_index": int(donor_sample["source_index"]),
        "donor_row_sha256": donor_sample["row_sha256"],
    }


def semantic_decision_nll_from_current_state(
    *,
    model,
    tokenizer,
    sample: dict[str, Any],
    donor_sample: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    features = rendered_scene_decision_features(tokenizer, sample, device)
    donor_features = rendered_scene_decision_features(tokenizer, donor_sample, device)
    pair_target = first_pair_distinguishing_scene_target(
        source_features=features,
        donor_features=donor_features,
        donor_sample=donor_sample,
    )
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **logits_to_keep_kwargs(model, 0),
        )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

    def selected_nll(
        positions: list[int],
        *,
        mask_mode: str,
        normalization: str,
        alternative_target_token_ids: list[int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_positions = torch.tensor(positions, device=logits.device)
        predictor_positions = target_positions - 1
        selected_logits = logits[0].index_select(0, predictor_positions).float()
        selected_targets = input_ids[0].index_select(0, target_positions)
        token_nll = F.cross_entropy(
            selected_logits,
            selected_targets,
            reduction="none",
        )
        if not bool(torch.isfinite(token_nll).all().item()):
            raise ValueError("Scene semantic decision NLL is non-finite")
        alternative_metrics: dict[str, Any] = {}
        if alternative_target_token_ids is not None:
            if len(alternative_target_token_ids) != len(positions) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in alternative_target_token_ids
            ):
                raise ValueError(
                    "Scene semantic alternative targets do not align to selected logits"
                )
            alternative_targets = torch.tensor(
                alternative_target_token_ids,
                dtype=selected_targets.dtype,
                device=selected_targets.device,
            )
            alternative_nll = F.cross_entropy(
                selected_logits,
                alternative_targets,
                reduction="none",
            )
            if not bool(torch.isfinite(alternative_nll).all().item()):
                raise ValueError("Scene semantic alternative-target NLL is non-finite")
            alternative_metrics = {
                "alternative_target_token_ids": list(alternative_target_token_ids),
                "alternative_target_nll_sum": float(alternative_nll.sum().item()),
                "alternative_target_mean_nll": float(
                    alternative_nll.mean().item()
                ),
                "selected_over_alternative_logprob_margin": float(
                    (alternative_nll - token_nll).mean().item()
                ),
            }
        return {
            "mask_mode": mask_mode,
            "normalization": normalization,
            "selected_target_positions": positions,
            "selected_target_token_ids": [
                int(value) for value in selected_targets.detach().cpu().tolist()
            ],
            "token_count": len(positions),
            "nll_sum": float(token_nll.sum().item()),
            "mean_nll": float(token_nll.mean().item()),
            "read_rendered_sha256": sha256_text(features["rendered"]),
            **alternative_metrics,
            **(extra or {}),
        }

    all_semantic = selected_nll(
        features["selected_positions"],
        mask_mode=SEMANTIC_DECISION_MASK_MODE,
        normalization=SEMANTIC_DECISION_NLL_NORMALIZATION,
    )
    pair_positions = pair_target.pop("selected_target_positions")
    pair_target.pop("selected_target_token_ids")
    donor_target_token_ids = list(pair_target["donor_target_token_ids"])
    pair = selected_nll(
        pair_positions,
        mask_mode=PAIR_TARGET_DECISION_MASK_MODE,
        normalization=PAIR_TARGET_DECISION_NLL_NORMALIZATION,
        alternative_target_token_ids=donor_target_token_ids,
        extra=pair_target,
    )
    return {"all_semantic": all_semantic, "pair_target": pair}


def online_state_stats(model) -> dict[str, Any]:
    modules = list(iter_delta_modules(model))
    state_rows: list[dict[str, Any]] = []
    for name, module in modules:
        state = getattr(module, "delta_state", None)
        if state is None:
            continue
        value = state.detach().float()
        positions = getattr(module, "rwkv_ms_positions", None)
        previous_source = getattr(module, "rwkv_ms_previous_source", None)
        position_value = None if positions is None else positions.detach().long()
        previous_value = (
            None if previous_source is None else previous_source.detach().float()
        )
        state_rows.append(
            {
                "module": name,
                "layer_idx": int(getattr(module, "layer_idx", -1)),
                "matrix_norm": float(value.norm().item()),
                "matrix_max_abs": (
                    float(value.abs().max().item()) if value.numel() else 0.0
                ),
                "matrix_elements": int(value.numel()),
                "rwkv_ms_positions": (
                    None
                    if position_value is None
                    else [int(item) for item in position_value.cpu().tolist()]
                ),
                "rwkv_ms_previous_source_norm": (
                    None
                    if previous_value is None
                    else float(previous_value.norm().item())
                ),
                "rwkv_ms_previous_source_max_abs": (
                    None
                    if previous_value is None or not previous_value.numel()
                    else float(previous_value.abs().max().item())
                ),
            }
        )
    return {
        "adapter_modules": len(modules),
        "materialized_state_modules": len(state_rows),
        "nonzero_matrix_modules": sum(
            row["matrix_max_abs"] > 0.0 for row in state_rows
        ),
        "nonzero_previous_source_modules": sum(
            (row["rwkv_ms_previous_source_max_abs"] or 0.0) > 0.0
            for row in state_rows
        ),
        "mean_matrix_norm": (
            sum(row["matrix_norm"] for row in state_rows) / len(state_rows)
            if state_rows
            else 0.0
        ),
        "max_matrix_norm": max(
            (row["matrix_norm"] for row in state_rows), default=0.0
        ),
        "max_matrix_abs": max(
            (row["matrix_max_abs"] for row in state_rows), default=0.0
        ),
        "max_previous_source_norm": max(
            (
                row["rwkv_ms_previous_source_norm"] or 0.0
                for row in state_rows
            ),
            default=0.0,
        ),
        "max_position": max(
            (
                max(row["rwkv_ms_positions"] or [0])
                for row in state_rows
            ),
            default=0,
        ),
        "by_layer": state_rows,
    }


def prime_online_state(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    device: str,
) -> dict[str, Any]:
    import torch

    rendered, input_ids, attention_mask = render_and_tokenize(
        tokenizer,
        messages,
        add_generation_prompt=False,
        device=device,
    )
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **logits_to_keep_kwargs(model, 1),
        )
    return {
        "tokens": int(input_ids.size(1)),
        "rendered_sha256": sha256_text(rendered),
        "kv_cache_retained": False,
        "online_state": online_state_stats(model),
    }


def generation_config(model, tokenizer, max_new_tokens: int):
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.max_new_tokens = max_new_tokens
    config.use_cache = True
    config.temperature = None
    config.top_p = None
    config.top_k = None
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    return config


def generate_messages(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    import torch

    rendered, input_ids, attention_mask = render_and_tokenize(
        tokenizer,
        messages,
        add_generation_prompt=True,
        device=device,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config(model, tokenizer, max_new_tokens),
        )
    elapsed_seconds = time.perf_counter() - started_at
    generated_ids = output_ids[:, input_ids.size(1) :]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    peak_memory = None
    if device.startswith("cuda"):
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return {
        "status": "ok",
        "raw_generation": response,
        "parsed_json": extract_json(response),
        "input_tokens": int(input_ids.size(1)),
        "input_rendered_sha256": sha256_text(rendered),
        "output_tokens": int(generated_ids.size(1)),
        "hit_max_new_tokens": int(generated_ids.size(1)) >= max_new_tokens,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_memory,
        "memory_trace": collect_rwkv_trace(model),
        "online_state_after_generation": online_state_stats(model),
    }


def recovered_scene_score(prediction: Any, gold: Any) -> dict[str, Any]:
    recovered = recover_scene(prediction)
    predicted = recovered or set()
    gold_boundaries = strict_gold_boundaries(gold)
    tp = len(gold_boundaries & predicted)
    fp = len(predicted - gold_boundaries)
    fn = len(gold_boundaries - predicted)
    denominator = 2 * tp + fp + fn
    return {
        "schema_recovered": recovered is not None,
        "gold_boundaries": sorted(gold_boundaries),
        "predicted_boundaries": sorted(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sample_f1": 0.0 if denominator == 0 else (2 * tp) / denominator,
    }


def is_canonical_scene_prediction(prediction: Any) -> bool:
    if not isinstance(prediction, dict) or set(prediction) != {"boundaries"}:
        return False
    boundaries = prediction["boundaries"]
    return bool(
        isinstance(boundaries, list)
        and all(
            isinstance(boundary, int) and not isinstance(boundary, bool)
            for boundary in boundaries
        )
        and boundaries == sorted(set(boundaries))
    )


def evaluate_semantic_decision_condition(
    *,
    model,
    tokenizer,
    sample: dict[str, Any],
    donor_sample: dict[str, Any] | None,
    condition: str,
    device: str,
) -> dict[str, Any]:
    if condition not in {
        "state_only",
        "state_only_donor",
        "state_only_no_write",
    }:
        raise ValueError(f"Semantic decision NLL is unsupported for {condition}")
    if donor_sample is None:
        raise ValueError("Semantic decision NLL requires the frozen donor identity")
    reset_delta_state(model)
    try:
        if condition == "state_only_donor":
            if donor_sample is None:
                raise ValueError("Semantic donor NLL requires a donor sample")
            prime_sample = donor_sample
        else:
            prime_sample = sample
        if condition == "state_only_no_write":
            with memory_condition(model, "no_write"):
                prime_online_state(
                    model=model,
                    tokenizer=tokenizer,
                    messages=prime_sample["messages"],
                    device=device,
                )
                return semantic_decision_nll_from_current_state(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    donor_sample=donor_sample,
                    device=device,
                )
        prime_online_state(
            model=model,
            tokenizer=tokenizer,
            messages=prime_sample["messages"],
            device=device,
        )
        with memory_condition(model, "no_write"):
            return semantic_decision_nll_from_current_state(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                donor_sample=donor_sample,
                device=device,
            )
    finally:
        reset_delta_state(model)


def evaluate_condition(
    *,
    model,
    tokenizer,
    sample: dict[str, Any],
    donor_sample: dict[str, Any] | None = None,
    shuffled_sample: dict[str, Any] | None = None,
    condition: str,
    max_new_tokens: int,
    device: str,
    collect_semantic_nll: bool = False,
) -> dict[str, Any]:
    full_messages = sample["messages"]
    system_only = [full_messages[0]]
    reset_delta_state(model)
    prime: dict[str, Any] | None = None
    try:
        if condition in {"base_full", "normal_full"}:
            result = generate_messages(
                model=model,
                tokenizer=tokenizer,
                messages=full_messages,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        elif condition == "no_write_full":
            with memory_condition(model, "no_write"):
                result = generate_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=full_messages,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
        elif condition == "state_only":
            prime = prime_online_state(
                model=model,
                tokenizer=tokenizer,
                messages=full_messages,
                device=device,
            )
            with memory_condition(model, "no_write"):
                result = generate_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=system_only,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
        elif condition == "state_only_donor":
            if donor_sample is None:
                raise ValueError("state_only_donor requires a donor sample")
            if donor_sample["source_index"] == sample["source_index"]:
                raise ValueError("state_only_donor cannot prime from the current row")
            prime = prime_online_state(
                model=model,
                tokenizer=tokenizer,
                messages=donor_sample["messages"],
                device=device,
            )
            with memory_condition(model, "no_write"):
                result = generate_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=system_only,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
        elif condition == "state_only_shuffled":
            if shuffled_sample is None:
                raise ValueError("state_only_shuffled requires a shuffled sample")
            if shuffled_sample["source_index"] == sample["source_index"]:
                raise ValueError("state_only_shuffled cannot prime from the current row")
            if (
                donor_sample is not None
                and shuffled_sample["source_index"] == donor_sample["source_index"]
            ):
                raise ValueError("state_only_shuffled cannot reuse the locked donor row")
            if shuffled_sample["gold"] == sample["gold"]:
                raise ValueError("state_only_shuffled requires a label-distinct row")
            prime = prime_online_state(
                model=model,
                tokenizer=tokenizer,
                messages=shuffled_sample["messages"],
                device=device,
            )
            with memory_condition(model, "no_write"):
                result = generate_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=system_only,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
        elif condition == "state_only_no_write":
            with memory_condition(model, "no_write"):
                prime = prime_online_state(
                    model=model,
                    tokenizer=tokenizer,
                    messages=full_messages,
                    device=device,
                )
                result = generate_messages(
                    model=model,
                    tokenizer=tokenizer,
                    messages=system_only,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        parsed = result["parsed_json"]
        semantic_nll = (
            evaluate_semantic_decision_condition(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                donor_sample=donor_sample,
                condition=condition,
                device=device,
            )
            if collect_semantic_nll
            and condition
            in {"state_only", "state_only_donor", "state_only_no_write"}
            else None
        )
        return {
            **result,
            "prime": prime,
            "donor_source_index": (
                int(donor_sample["source_index"])
                if condition == "state_only_donor" and donor_sample is not None
                else None
            ),
            "donor_row_sha256": (
                donor_sample["row_sha256"]
                if condition == "state_only_donor" and donor_sample is not None
                else None
            ),
            **(
                {
                    "shuffled_source_index": int(shuffled_sample["source_index"]),
                    "shuffled_row_sha256": shuffled_sample["row_sha256"],
                }
                if condition == "state_only_shuffled" and shuffled_sample is not None
                else {}
            ),
            "score_strict": score_prediction("scene", parsed, sample["gold"]),
            "score_recovered": recovered_scene_score(parsed, sample["gold"]),
            "semantic_decision_nll": semantic_nll,
        }
    finally:
        reset_delta_state(model)


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    tp = sum(int(row["score_recovered"]["tp"]) for row in rows)
    fp = sum(int(row["score_recovered"]["fp"]) for row in rows)
    fn = sum(int(row["score_recovered"]["fn"]) for row in rows)
    strict_tp = sum(int(row["score_strict"]["tp"]) for row in rows)
    strict_fp = sum(int(row["score_strict"]["fp"]) for row in rows)
    strict_fn = sum(int(row["score_strict"]["fn"]) for row in rows)
    empty_rows = [
        row for row in rows if not row["score_strict"].get("gold_boundaries", [])
    ]
    recovered_outputs = sum(
        bool(row["score_recovered"]["schema_recovered"]) for row in rows
    )
    canonical_outputs = sum(
        is_canonical_scene_prediction(row.get("parsed_json")) for row in rows
    )
    strict_predicted_boundaries = strict_tp + strict_fp
    strict_gold_boundaries = strict_tp + strict_fn

    def f1(a: int, b: int, c: int) -> float:
        denominator = 2 * a + b + c
        return 0.0 if denominator == 0 else (2 * a) / denominator

    return {
        "samples": len(rows),
        "format_recovered": {
            "metric_name": "format_recovered_micro_f1",
            "primary_metric": f1(tp, fp, fn),
            "precision": 0.0 if tp + fp == 0 else tp / (tp + fp),
            "recall": 0.0 if tp + fn == 0 else tp / (tp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "coverage": (
                sum(bool(row["score_recovered"]["schema_recovered"]) for row in rows)
                / len(rows)
                if rows
                else 0.0
            ),
        },
        "strict": {
            "metric_name": BENCHMARK_SCENE_METRIC_NAME,
            "primary_metric": f1(strict_tp, strict_fp, strict_fn),
            "precision": (
                0.0
                if strict_tp + strict_fp == 0
                else strict_tp / (strict_tp + strict_fp)
            ),
            "recall": (
                0.0
                if strict_tp + strict_fn == 0
                else strict_tp / (strict_tp + strict_fn)
            ),
            "tp": strict_tp,
            "fp": strict_fp,
            "fn": strict_fn,
            "predicted_boundary_count": strict_predicted_boundaries,
            "gold_boundary_count": strict_gold_boundaries,
            "predicted_boundaries_per_sample": (
                strict_predicted_boundaries / len(rows) if rows else 0.0
            ),
            "gold_boundaries_per_sample": (
                strict_gold_boundaries / len(rows) if rows else 0.0
            ),
            "predicted_to_gold_boundary_ratio": (
                strict_predicted_boundaries / strict_gold_boundaries
                if strict_gold_boundaries
                else 0.0
            ),
            "schema_valid_rate": (
                sum(bool(row["score_strict"]["schema_valid"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        },
        "decision_quality": {
            "recovered_outputs": recovered_outputs,
            "canonical_outputs": canonical_outputs,
            "empty_list_rows": len(empty_rows),
            "recovered_empty_list_exact": sum(
                bool(row["score_recovered"]["schema_recovered"])
                and not row["score_recovered"].get("predicted_boundaries", [])
                for row in empty_rows
            ),
            "canonical_empty_list_exact": sum(
                is_canonical_scene_prediction(row.get("parsed_json"))
                and not row["score_strict"].get("predicted_boundaries", [])
                for row in empty_rows
            ),
            "gold_positive_boundaries": tp + fn,
            "recovered_gold_positives": tp,
            "strict_gold_positive_boundaries": strict_gold_boundaries,
            "strict_gold_positives_recovered": strict_tp,
        },
        "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
    }


def progress_log_line(
    *, condition: str, source_index: int, record: dict[str, Any]
) -> str:
    """Label benchmark and recovery scores explicitly in live progress."""

    return (
        "SCENE_STATE_EVAL "
        f"condition={condition} source_index={source_index} "
        f"strict_f1={float(record['score_strict']['sample_f1']):.4f} "
        f"recovered_f1={float(record['score_recovered']['sample_f1']):.4f}"
    )


def build_comparisons(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, left, right in (
        ("normal_full_minus_base_full", "normal_full", "base_full"),
        ("normal_full_minus_no_write_full", "normal_full", "no_write_full"),
        (
            "state_only_minus_state_only_no_write",
            "state_only",
            "state_only_no_write",
        ),
        (
            "state_only_minus_state_only_donor",
            "state_only",
            "state_only_donor",
        ),
        (
            "state_only_minus_state_only_shuffled",
            "state_only",
            "state_only_shuffled",
        ),
    ):
        if left not in summaries or right not in summaries:
            continue
        left_summary = summaries[left]["strict"]
        right_summary = summaries[right]["strict"]
        if (
            left_summary.get("metric_name") != BENCHMARK_SCENE_METRIC_NAME
            or right_summary.get("metric_name") != BENCHMARK_SCENE_METRIC_NAME
        ):
            raise ValueError("Scene comparison requires benchmark-compatible strict F1")
        left_metric = float(left_summary["primary_metric"])
        right_metric = float(right_summary["primary_metric"])
        comparisons[name] = {
            "metric_name": BENCHMARK_SCENE_METRIC_NAME,
            left: left_metric,
            right: right_metric,
            "delta": left_metric - right_metric,
        }
    return comparisons


def _strict_exact_record(record: dict[str, Any]) -> bool:
    score = record.get("score_strict")
    return bool(
        isinstance(score, dict)
        and score.get("schema_valid") is True
        and score.get("fp") == 0
        and score.get("fn") == 0
    )


def _recovered_exact_record(record: dict[str, Any]) -> bool:
    score = record.get("score_recovered")
    return bool(
        isinstance(score, dict)
        and score.get("schema_recovered") is True
        and score.get("fp") == 0
        and score.get("fn") == 0
    )


def build_historical_v6_hard32_evidence(
    ordered_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    required = list(HISTORICAL_V6_HARD32_CONDITIONS)
    if set(ordered_records) != set(required):
        raise ValueError(
            "Historical V6 evidence requires exactly base_full, no_write_full, and "
            "normal_full"
        )
    expected_order = list(HARD32_ROW_INDICES)
    for condition in required:
        records = ordered_records[condition]
        if [int(record["source_index"]) for record in records] != expected_order:
            raise ValueError(
                f"Historical V6 {condition} result order differs from frozen Hard32"
            )
    for records in zip(*(ordered_records[condition] for condition in required), strict=True):
        identities = {
            (
                int(record["source_index"]),
                str(record["row_sha256"]),
                json.dumps(record["gold"], sort_keys=True, separators=(",", ":")),
            )
            for record in records
        }
        if len(identities) != 1:
            raise ValueError(
                "Historical V6 condition outputs differ in row identity or gold label"
            )

    reference_by_source = {
        int(record["source_index"]): record
        for record in ordered_records["base_full"]
    }
    stratum_by_source: dict[int, str] = {}
    for left_index, right_index in HARD32_FROZEN_DONOR_PAIRS:
        left_gold = reference_by_source[left_index]["score_strict"]["gold_boundaries"]
        right_gold = reference_by_source[right_index]["score_strict"]["gold_boundaries"]
        left_count = len(left_gold)
        right_count = len(right_gold)
        if (left_count == 0) != (right_count == 0):
            stratum = "presence"
        elif left_count == right_count:
            stratum = "same_cardinality_value"
        else:
            stratum = "cross_cardinality_value"
        stratum_by_source[left_index] = stratum
        stratum_by_source[right_index] = stratum
    if set(stratum_by_source) != set(HARD32_ROW_INDICES):
        raise ValueError("Historical V6 Hard32 strata do not cover every frozen row")
    observed_strata = {
        name: sum(value == name for value in stratum_by_source.values())
        for name in (
            "presence",
            "same_cardinality_value",
            "cross_cardinality_value",
        )
    }
    expected_strata = {
        "presence": HARD32_FROZEN_DONOR_STRATUM_ROWS["empty_vs_nonempty"],
        "same_cardinality_value": HARD32_FROZEN_DONOR_STRATUM_ROWS[
            "nonempty_same_cardinality"
        ],
        "cross_cardinality_value": HARD32_FROZEN_DONOR_STRATUM_ROWS[
            "nonempty_different_cardinality"
        ],
    }
    if observed_strata != expected_strata:
        raise ValueError(
            "Historical V6 Hard32 observed label-cardinality strata differ from the "
            "frozen contract"
        )

    records_by_condition = {
        condition: {
            int(record["source_index"]): record
            for record in ordered_records[condition]
        }
        for condition in required
    }

    def generation_differences(indices: list[int]) -> dict[str, Any]:
        comparisons: dict[str, Any] = {}
        for name, left, right in (
            ("normal_full_vs_base_full", "normal_full", "base_full"),
            ("normal_full_vs_no_write_full", "normal_full", "no_write_full"),
            ("no_write_full_vs_base_full", "no_write_full", "base_full"),
        ):
            left_rows = records_by_condition[left]
            right_rows = records_by_condition[right]
            comparisons[name] = {
                "raw_generation_different_rows": sum(
                    left_rows[index]["raw_generation"]
                    != right_rows[index]["raw_generation"]
                    for index in indices
                ),
                "parsed_json_different_rows": sum(
                    left_rows[index]["parsed_json"] != right_rows[index]["parsed_json"]
                    for index in indices
                ),
                "left_strict_exact_right_not_rows": sum(
                    _strict_exact_record(left_rows[index])
                    and not _strict_exact_record(right_rows[index])
                    for index in indices
                ),
                "right_strict_exact_left_not_rows": sum(
                    _strict_exact_record(right_rows[index])
                    and not _strict_exact_record(left_rows[index])
                    for index in indices
                ),
            }
        return comparisons

    def summarize_indices(indices: list[int]) -> dict[str, Any]:
        condition_summaries: dict[str, Any] = {}
        for condition in required:
            records = [records_by_condition[condition][index] for index in indices]
            condition_summaries[condition] = {
                "strict_exact_rows": sum(_strict_exact_record(record) for record in records),
                "recovered_exact_rows": sum(
                    _recovered_exact_record(record) for record in records
                ),
                "summary": summarize_records(records),
            }
        strict_metrics = {
            condition: float(
                condition_summaries[condition]["summary"]["strict"]["primary_metric"]
            )
            for condition in required
        }
        recovered_metrics = {
            condition: float(
                condition_summaries[condition]["summary"]["format_recovered"][
                    "primary_metric"
                ]
            )
            for condition in required
        }
        strongest_strict = max(
            strict_metrics["base_full"], strict_metrics["no_write_full"]
        )
        strongest_recovered = max(
            recovered_metrics["base_full"], recovered_metrics["no_write_full"]
        )
        return {
            "rows": len(indices),
            "source_indices": list(indices),
            "conditions": condition_summaries,
            "strict_uplift": {
                "metric_name": BENCHMARK_SCENE_METRIC_NAME,
                "normal_full": strict_metrics["normal_full"],
                "base_full": strict_metrics["base_full"],
                "no_write_full": strict_metrics["no_write_full"],
                "strongest_control": strongest_strict,
                "normal_full_minus_strongest_control": (
                    strict_metrics["normal_full"] - strongest_strict
                ),
            },
            "format_recovered_uplift": {
                "metric_name": "format_recovered_micro_f1",
                "normal_full": recovered_metrics["normal_full"],
                "base_full": recovered_metrics["base_full"],
                "no_write_full": recovered_metrics["no_write_full"],
                "strongest_control": strongest_recovered,
                "normal_full_minus_strongest_control": (
                    recovered_metrics["normal_full"] - strongest_recovered
                ),
            },
            "generation_differences": generation_differences(indices),
        }

    overall_indices = list(HARD32_ROW_INDICES)
    return {
        "schema": "rwkv_ms_scene_historical_v6_hard32_evidence.v1",
        "interpretation": (
            "normal_full - max(base_full, no_write_full) measures benchmark uplift "
            "for this frozen diagnostic slice; normal_full - no_write_full isolates "
            "the contribution of online writes while the full prompt remains visible."
        ),
        "overall": summarize_indices(overall_indices),
        "strata": {
            stratum: summarize_indices(
                [
                    index
                    for index in overall_indices
                    if stratum_by_source[index] == stratum
                ]
            )
            for stratum in expected_strata
        },
        "observed_stratum_rows": observed_strata,
        "full170_authorized": False,
        "test_authorized": False,
    }


def build_semantic_decision_evidence(
    ordered_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    required = ("state_only", "state_only_donor", "state_only_no_write")
    if any(condition not in ordered_records for condition in required):
        raise ValueError("Hard32 semantic NLL conditions are incomplete")
    row_counts = {len(ordered_records[condition]) for condition in required}
    if row_counts != {32}:
        raise ValueError("Hard32 semantic NLL requires exactly 32 rows per condition")
    correct_by_source = {
        int(record["source_index"]): record
        for record in ordered_records["state_only"]
    }
    if len(correct_by_source) != 32:
        raise ValueError("Hard32 semantic NLL current-state rows are not unique")
    rows: list[dict[str, Any]] = []
    for correct, donor, zero in zip(
        *(ordered_records[condition] for condition in required),
        strict=True,
    ):
        source_indices = {
            int(correct["source_index"]),
            int(donor["source_index"]),
            int(zero["source_index"]),
        }
        if len(source_indices) != 1:
            raise ValueError("Hard32 semantic NLL row order differs between conditions")
        nlls = {
            "correct": correct.get("semantic_decision_nll"),
            "donor": donor.get("semantic_decision_nll"),
            "zero": zero.get("semantic_decision_nll"),
        }
        if any(not isinstance(value, dict) for value in nlls.values()):
            raise ValueError("Hard32 semantic decision NLL evidence is missing")
        all_semantic = {
            name: value.get("all_semantic") for name, value in nlls.items()
        }
        pair_target = {
            name: value.get("pair_target") for name, value in nlls.items()
        }
        if any(not isinstance(value, dict) for value in all_semantic.values()) or any(
            not isinstance(value, dict) for value in pair_target.values()
        ):
            raise ValueError("Hard32 all-semantic or pair-target NLL evidence is missing")

        def shared_target_identity(
            reports: dict[str, dict[str, Any]],
            *,
            description: str,
        ) -> tuple[tuple[int, ...], tuple[int, ...], str]:
            positions = {
                tuple(value["selected_target_positions"])
                for value in reports.values()
            }
            token_ids = {
                tuple(value["selected_target_token_ids"])
                for value in reports.values()
            }
            rendered_hashes = {
                str(value["read_rendered_sha256"]) for value in reports.values()
            }
            if len(positions) != 1 or len(token_ids) != 1 or len(rendered_hashes) != 1:
                raise ValueError(
                    f"Hard32 {description} target identity differs across controls"
                )
            return (
                next(iter(positions)),
                next(iter(token_ids)),
                next(iter(rendered_hashes)),
            )

        all_positions, all_token_ids, all_rendered_hash = shared_target_identity(
            all_semantic,
            description="all-semantic",
        )
        pair_positions, pair_token_ids, pair_rendered_hash = shared_target_identity(
            pair_target,
            description="pair-target",
        )
        if pair_rendered_hash != all_rendered_hash:
            raise ValueError("Hard32 pair and all-semantic reads differ")
        if len(pair_positions) != 1 or len(pair_token_ids) != 1:
            raise ValueError("Hard32 pair-target NLL must select exactly one token")
        if pair_positions[0] not in all_positions:
            raise ValueError("Hard32 pair target is outside the all-semantic mask")
        all_token_by_position = dict(zip(all_positions, all_token_ids, strict=True))
        if all_token_by_position[pair_positions[0]] != pair_token_ids[0]:
            raise ValueError("Hard32 pair target token differs from all-semantic evidence")

        pair_identity_fields = (
            "target_mode",
            "first_differing_semantic_ordinal",
            "donor_target_token_ids",
            "causal_prefix_sha256",
            "donor_source_index",
            "donor_row_sha256",
        )
        if any(
            len({json.dumps(value[field], sort_keys=True) for value in pair_target.values()})
            != 1
            for field in pair_identity_fields
        ):
            raise ValueError("Hard32 pair-target donor identity differs across controls")

        all_means = {
            name: float(value["mean_nll"]) for name, value in all_semantic.items()
        }
        pair_means = {
            name: float(value["mean_nll"]) for name, value in pair_target.items()
        }
        if any(
            not math.isfinite(value)
            for value in (*all_means.values(), *pair_means.values())
        ):
            raise ValueError("Hard32 semantic decision NLL is non-finite")
        donor_source_index = donor.get("donor_source_index")
        if (
            isinstance(donor_source_index, bool)
            or not isinstance(donor_source_index, int)
            or donor_source_index not in correct_by_source
            or donor_source_index == int(correct["source_index"])
        ):
            raise ValueError("Hard32 semantic donor source identity is invalid")
        if pair_target["correct"]["donor_source_index"] != donor_source_index:
            raise ValueError("Hard32 pair-target donor differs from generation donor")
        if (
            pair_target["correct"]["donor_row_sha256"]
            != correct_by_source[donor_source_index]["row_sha256"]
        ):
            raise ValueError("Hard32 pair-target donor row hash differs")
        source_cardinality = len(strict_gold_boundaries(correct["gold"]))
        donor_cardinality = len(
            strict_gold_boundaries(correct_by_source[donor_source_index]["gold"])
        )
        if source_cardinality == 0 or donor_cardinality == 0:
            cardinality_stratum = "empty_vs_nonempty"
        elif source_cardinality == donor_cardinality:
            cardinality_stratum = "nonempty_same_cardinality"
        else:
            cardinality_stratum = "nonempty_different_cardinality"
        donor_pair_gap = pair_means["donor"] - pair_means["correct"]
        donor_all_gap = all_means["donor"] - all_means["correct"]
        zero_all_gap = all_means["zero"] - all_means["correct"]
        rows.append(
            {
                "source_index": next(iter(source_indices)),
                "donor_source_index": donor_source_index,
                "row_sha256": correct["row_sha256"],
                "gold": correct["gold"],
                "source_gold_boundary_count": source_cardinality,
                "donor_gold_boundary_count": donor_cardinality,
                "donor_minus_source_gold_cardinality": (
                    donor_cardinality - source_cardinality
                ),
                "absolute_gold_cardinality_difference": abs(
                    donor_cardinality - source_cardinality
                ),
                "gold_cardinality_stratum": cardinality_stratum,
                "all_semantic_target_positions": list(all_positions),
                "all_semantic_target_token_ids": list(all_token_ids),
                "all_semantic_target_tokens": len(all_token_ids),
                "pair_target_positions": list(pair_positions),
                "pair_target_token_ids": list(pair_token_ids),
                "pair_target_donor_token_ids": pair_target["correct"][
                    "donor_target_token_ids"
                ],
                "pair_target_first_differing_semantic_ordinal": pair_target[
                    "correct"
                ]["first_differing_semantic_ordinal"],
                "pair_target_causal_prefix_sha256": pair_target["correct"][
                    "causal_prefix_sha256"
                ],
                "correct_state_all_semantic_mean_nll": all_means["correct"],
                "donor_state_all_semantic_mean_nll": all_means["donor"],
                "zero_state_all_semantic_mean_nll": all_means["zero"],
                "correct_state_pair_target_nll": pair_means["correct"],
                "donor_state_pair_target_nll": pair_means["donor"],
                "zero_state_pair_target_nll": pair_means["zero"],
                "donor_minus_correct_pair_target_nll_gap": donor_pair_gap,
                "donor_minus_correct_all_semantic_nll_gap_diagnostic": (
                    donor_all_gap
                ),
                "zero_minus_correct_all_semantic_nll_gap": zero_all_gap,
                "correct_better_than_donor_pair_target": donor_pair_gap > 0.0,
                "correct_better_than_donor_all_semantic_diagnostic": (
                    donor_all_gap > 0.0
                ),
                "correct_better_than_zero_all_semantic": zero_all_gap > 0.0,
            }
        )

    donor_map = {
        int(row["source_index"]): int(row["donor_source_index"]) for row in rows
    }
    if any(donor_map.get(donor) != source for source, donor in donor_map.items()):
        raise ValueError("Hard32 semantic donor mapping is not symmetric")

    def gap_summary(
        selected_rows: list[dict[str, Any]],
        *,
        gap_field: str,
        positive_field: str,
    ) -> dict[str, Any]:
        return {
            "rows": len(selected_rows),
            "source_indices": [int(row["source_index"]) for row in selected_rows],
            "source_donor_pairs": [
                {
                    "source_index": int(row["source_index"]),
                    "donor_source_index": int(row["donor_source_index"]),
                }
                for row in selected_rows
            ],
            "positive_rows": sum(bool(row[positive_field]) for row in selected_rows),
            "positive_fraction": (
                sum(bool(row[positive_field]) for row in selected_rows)
                / len(selected_rows)
                if selected_rows
                else 0.0
            ),
            "mean_gap": (
                sum(float(row[gap_field]) for row in selected_rows)
                / len(selected_rows)
                if selected_rows
                else 0.0
            ),
        }

    donor_strata = {
        stratum: gap_summary(
            [row for row in rows if row["gold_cardinality_stratum"] == stratum],
            gap_field="donor_minus_correct_pair_target_nll_gap",
            positive_field="correct_better_than_donor_pair_target",
        )
        for stratum in (
            "empty_vs_nonempty",
            "nonempty_same_cardinality",
            "nonempty_different_cardinality",
        )
    }

    def histogram(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            key = str(row[field])
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))

    return {
        "all_semantic_mask_mode": SEMANTIC_DECISION_MASK_MODE,
        "all_semantic_normalization": SEMANTIC_DECISION_NLL_NORMALIZATION,
        "pair_target_mode": PAIR_TARGET_DECISION_MASK_MODE,
        "pair_target_normalization": PAIR_TARGET_DECISION_NLL_NORMALIZATION,
        "gap_sign_convention": (
            "comparator mean NLL minus correct-state mean NLL; positive means "
            "the current row's state predicts its semantic targets better"
        ),
        "rows": rows,
        "donor_pair_target_minus_correct": gap_summary(
            rows,
            gap_field="donor_minus_correct_pair_target_nll_gap",
            positive_field="correct_better_than_donor_pair_target",
        ),
        "donor_all_semantic_minus_correct_diagnostic": gap_summary(
            rows,
            gap_field="donor_minus_correct_all_semantic_nll_gap_diagnostic",
            positive_field="correct_better_than_donor_all_semantic_diagnostic",
        ),
        "zero_all_semantic_minus_correct": gap_summary(
            rows,
            gap_field="zero_minus_correct_all_semantic_nll_gap",
            positive_field="correct_better_than_zero_all_semantic",
        ),
        "donor_label_cardinality": {
            "signed_donor_minus_source_counts": histogram(
                "donor_minus_source_gold_cardinality"
            ),
            "absolute_difference_counts": histogram(
                "absolute_gold_cardinality_difference"
            ),
            "strata": donor_strata,
        },
    }


def build_base_outcome_evidence(
    ordered_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    required = ("base_full", "normal_full", "state_only")
    if any(condition not in ordered_records for condition in required):
        raise ValueError("Hard32 base-outcome evidence conditions are incomplete")
    rows: list[dict[str, Any]] = []
    for base, normal, state in zip(
        *(ordered_records[condition] for condition in required),
        strict=True,
    ):
        if len({base["source_index"], normal["source_index"], state["source_index"]}) != 1:
            raise ValueError("Hard32 base-outcome row order differs")
        gold = base["score_recovered"]["gold_boundaries"]
        predictions = {
            "base_full": base["score_recovered"]["predicted_boundaries"],
            "normal_full": normal["score_recovered"]["predicted_boundaries"],
            "state_only": state["score_recovered"]["predicted_boundaries"],
        }
        exact = {name: prediction == gold for name, prediction in predictions.items()}
        rows.append(
            {
                "source_index": int(base["source_index"]),
                "row_sha256": base["row_sha256"],
                "gold_boundaries": gold,
                "predicted_boundaries": predictions,
                "exact": exact,
                "base_failure": not exact["base_full"],
                "base_failure_recovered_by_normal_full": (
                    not exact["base_full"] and exact["normal_full"]
                ),
                "base_failure_recovered_by_state_only": (
                    not exact["base_full"] and exact["state_only"]
                ),
                "base_success_sentinel": exact["base_full"],
                "base_success_regressed_by_normal_full": (
                    exact["base_full"] and not exact["normal_full"]
                ),
                "base_success_regressed_by_state_only": (
                    exact["base_full"] and not exact["state_only"]
                ),
            }
        )
    return {
        "rows": rows,
        "counts": {
            "base_failures": sum(row["base_failure"] for row in rows),
            "base_failures_recovered_by_normal_full": sum(
                row["base_failure_recovered_by_normal_full"] for row in rows
            ),
            "base_failures_recovered_by_state_only": sum(
                row["base_failure_recovered_by_state_only"] for row in rows
            ),
            "base_success_sentinels": sum(row["base_success_sentinel"] for row in rows),
            "base_success_regressed_by_normal_full": sum(
                row["base_success_regressed_by_normal_full"] for row in rows
            ),
            "base_success_regressed_by_state_only": sum(
                row["base_success_regressed_by_state_only"] for row in rows
            ),
        },
    }


def build_scene_v6_identity_hard32_gate(
    *,
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    semantic_evidence: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not is_identity_hard32_contract(contract):
        return {"status": "not_requested", "contract": contract}
    fixed_scope_only = is_fixed_scope_hard32_contract(contract)
    state = summaries["state_only"]
    normal = summaries["normal_full"]
    base = summaries["base_full"]
    no_write = summaries["no_write_full"]
    state_donor_delta = comparisons["state_only_minus_state_only_donor"]["delta"]
    state_zero_delta = comparisons["state_only_minus_state_only_no_write"]["delta"]
    focused_contract = contract.get("name") == SCENE_HARD_FAILURE_HARD32_CONTRACT
    state_shuffled_delta = (
        comparisons["state_only_minus_state_only_shuffled"]["delta"]
        if focused_contract
        else None
    )
    strongest_full_control = max(
        float(base["strict"]["primary_metric"]),
        float(no_write["strict"]["primary_metric"]),
    )
    benchmark_metric_evidence = {
        condition: {
            field: summaries[condition]["strict"][field]
            for field in (
                "metric_name",
                "primary_metric",
                "precision",
                "recall",
                "tp",
                "fp",
                "fn",
                "predicted_boundary_count",
                "gold_boundary_count",
                "predicted_boundaries_per_sample",
                "gold_boundaries_per_sample",
                "predicted_to_gold_boundary_ratio",
                "schema_valid_rate",
            )
        }
        for condition in (
            SCENE_FOCUSED_CONDITIONS if focused_contract else CONDITIONS
        )
    }
    gates = {
        "correct_better_than_donor_rows": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS[
                "semantic_advantage_positive_rows"
            ],
            "value": semantic_evidence["donor_pair_target_minus_correct"][
                "positive_rows"
            ],
        },
        "correct_better_than_zero_rows": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS[
                "semantic_advantage_positive_rows"
            ],
            "value": semantic_evidence["zero_all_semantic_minus_correct"][
                "positive_rows"
            ],
        },
        "correct_better_than_same_cardinality_nonempty_donor_rows": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS[
                "same_cardinality_nonempty_positive_rows"
            ],
            "value": semantic_evidence["donor_label_cardinality"]["strata"][
                "nonempty_same_cardinality"
            ]["positive_rows"],
        },
        "state_only_minus_donor_f1": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["state_minus_donor_f1"],
            "value": state_donor_delta,
        },
        "state_only_minus_zero_f1": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["state_minus_zero_f1"],
            "value": state_zero_delta,
        },
        "normal_full_minus_strongest_control_f1": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS[
                "normal_minus_strongest_control_f1"
            ],
            "value": (
                float(normal["strict"]["primary_metric"])
                - strongest_full_control
            ),
            "normal_full_f1": normal["strict"]["primary_metric"],
            "strongest_control_f1": strongest_full_control,
        },
        "state_only_gold_positives_recovered": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["state_true_positives"],
            "value": state["strict"]["tp"],
        },
        "state_only_empty_list_exact": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["empty_list_exact"],
            "value": state["decision_quality"]["canonical_empty_list_exact"],
            "expected_empty_rows": HARD32_GATE_REQUIREMENTS["empty_list_rows"],
            "observed_empty_rows": state["decision_quality"]["empty_list_rows"],
        },
        "normal_full_predicted_boundary_density": {
            "operator": "<=",
            "threshold": HARD32_GATE_REQUIREMENTS[
                "max_predicted_to_gold_boundary_ratio"
            ],
            "value": normal["strict"]["predicted_to_gold_boundary_ratio"],
        },
        "state_only_predicted_boundary_density": {
            "operator": "<=",
            "threshold": HARD32_GATE_REQUIREMENTS[
                "max_predicted_to_gold_boundary_ratio"
            ],
            "value": state["strict"]["predicted_to_gold_boundary_ratio"],
        },
        "state_only_recovered_outputs": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["recovered_outputs"],
            "value": state["decision_quality"]["recovered_outputs"],
        },
        "state_only_canonical_outputs": {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["canonical_outputs"],
            "value": state["decision_quality"]["canonical_outputs"],
        },
    }
    if focused_contract:
        gates["state_only_minus_shuffled_f1"] = {
            "operator": ">=",
            "threshold": HARD32_GATE_REQUIREMENTS["state_minus_donor_f1"],
            "value": state_shuffled_delta,
        }
    if state["decision_quality"]["empty_list_rows"] != HARD32_GATE_REQUIREMENTS[
        "empty_list_rows"
    ]:
        raise ValueError("Hard32 empty-list stratum differs from the protected contract")
    same_cardinality_rows = semantic_evidence["donor_label_cardinality"]["strata"][
        "nonempty_same_cardinality"
    ]["rows"]
    if same_cardinality_rows != HARD32_GATE_REQUIREMENTS[
        "same_cardinality_nonempty_rows"
    ]:
        raise ValueError(
            "Hard32 same-cardinality nonempty donor stratum differs from the "
            "protected contract"
        )
    for gate in gates.values():
        value = float(gate["value"])
        threshold = float(gate["threshold"])
        if not math.isfinite(value) or not math.isfinite(threshold):
            raise ValueError("Hard32 gate contains a non-finite value")
        if gate["operator"] == ">=":
            gate["passed"] = value >= threshold
        elif gate["operator"] == "<=":
            gate["passed"] = value <= threshold
        elif gate["operator"] == ">":
            gate["passed"] = value > threshold
        else:
            raise ValueError(f"Unsupported Hard32 gate operator: {gate['operator']}")
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "status": "pass" if passed else "fail",
        "contract": contract,
        "gates": gates,
        "benchmark_metric_evidence": benchmark_metric_evidence,
        "format_recovery_diagnostic": {
            condition: summaries[condition]["format_recovered"]
            for condition in (
                SCENE_FOCUSED_CONDITIONS if focused_contract else CONDITIONS
            )
        },
        "all_gates_passed": passed,
        "full170_authorized_for_bound_checkpoint": passed and not fixed_scope_only,
        "authorization_scope": (
            SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE
            if focused_contract
            else (SCENE_V14_HARD32_AUTHORIZATION_SCOPE if fixed_scope_only else None)
        ),
        "other_benchmarks_authorized": False if fixed_scope_only else None,
        "test_selection_forbidden": True,
    }


def build_scene_v15_hard32_gate(
    *,
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    semantic_evidence: dict[str, Any],
    ordered_records: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply only the frozen V15 correct-state versus donor/zero decision."""

    if contract.get("name") != SCENE_V15_HARD32_CONTRACT:
        return {"status": "not_requested", "contract": contract}
    if list(summaries) != list(SCENE_V15_HARD32_CONDITIONS):
        raise ValueError("V15 Hard32 summaries must contain exactly three conditions")
    donor_comparison = comparisons.get("state_only_minus_state_only_donor")
    zero_comparison = comparisons.get("state_only_minus_state_only_no_write")
    if not isinstance(donor_comparison, Mapping) or not isinstance(
        zero_comparison,
        Mapping,
    ):
        raise ValueError("V15 Hard32 correct-state comparisons are incomplete")
    gates = {
        "state_only_strict_f1_greater_than_donor": {
            "metric_name": BENCHMARK_SCENE_METRIC_NAME,
            "operator": ">",
            "threshold": 0.0,
            "value": float(donor_comparison["delta"]),
            "correct_state_f1": summaries["state_only"]["strict"][
                "primary_metric"
            ],
            "control_f1": summaries["state_only_donor"]["strict"][
                "primary_metric"
            ],
        },
        "state_only_strict_f1_greater_than_zero": {
            "metric_name": BENCHMARK_SCENE_METRIC_NAME,
            "operator": ">",
            "threshold": 0.0,
            "value": float(zero_comparison["delta"]),
            "correct_state_f1": summaries["state_only"]["strict"][
                "primary_metric"
            ],
            "control_f1": summaries["state_only_no_write"]["strict"][
                "primary_metric"
            ],
        },
    }
    for gate in gates.values():
        value = float(gate["value"])
        if not math.isfinite(value):
            raise ValueError("V15 Hard32 gate contains a non-finite value")
        gate["passed"] = value > 0.0
    passed = all(gate["passed"] for gate in gates.values())
    exact_generation = {
        condition: {
            "strict_parsed_exact_rows": sum(
                _strict_exact_record(record) for record in ordered_records[condition]
            ),
            "format_recovered_exact_rows": sum(
                _recovered_exact_record(record) for record in ordered_records[condition]
            ),
            "rows": len(ordered_records[condition]),
        }
        for condition in SCENE_V15_HARD32_CONDITIONS
    }
    benchmark_scores = {
        condition: {
            field: summaries[condition]["strict"][field]
            for field in (
                "metric_name",
                "primary_metric",
                "precision",
                "recall",
                "tp",
                "fp",
                "fn",
                "schema_valid_rate",
            )
        }
        for condition in SCENE_V15_HARD32_CONDITIONS
    }
    return {
        "status": "pass" if passed else "fail",
        "contract": contract,
        "policy": dict(SCENE_V15_HARD32_GATE_POLICY),
        "gates": gates,
        "benchmark_metric_evidence": benchmark_scores,
        "exact_generation_evidence": exact_generation,
        "semantic_nll_evidence": {
            "donor_pair_target_minus_correct": semantic_evidence[
                "donor_pair_target_minus_correct"
            ],
            "donor_all_semantic_minus_correct_diagnostic": semantic_evidence[
                "donor_all_semantic_minus_correct_diagnostic"
            ],
            "zero_all_semantic_minus_correct": semantic_evidence[
                "zero_all_semantic_minus_correct"
            ],
        },
        "format_recovery_diagnostic": {
            condition: summaries[condition]["format_recovered"]
            for condition in SCENE_V15_HARD32_CONDITIONS
        },
        "all_gates_passed": passed,
        "full170_authorized_for_bound_checkpoint": False,
        "authorization_scope": SCENE_V15_HARD32_AUTHORIZATION_SCOPE,
        "other_benchmarks_authorized": False,
        "test_selection_forbidden": True,
    }


def build_scene_v6_matched_donor_gate(
    comparisons: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if contract.get("name") in {
        "generic",
        "scene_v6_identity_hard32",
        SCENE_V8_HARD32_CONTRACT,
        SCENE_V14_HARD32_CONTRACT,
        SCENE_V15_HARD32_CONTRACT,
        SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        SCENE_HARD_FAILURE_HARD32_CONTRACT,
        HISTORICAL_V6_HARD32_CONTRACT,
    }:
        return {"status": "not_requested", "contract": contract}
    if contract.get("name") != "scene_v6_matched_donor_validation":
        raise ValueError("Unsupported matched-donor gate contract")
    comparison = comparisons.get("state_only_minus_state_only_donor")
    if not isinstance(comparison, dict):
        raise ValueError("Matched-donor gate comparison is missing")
    delta = float(comparison["delta"])
    passed = delta > 0.0
    return {
        "status": "pass" if passed else "fail",
        "contract": contract,
        "metric_name": BENCHMARK_SCENE_METRIC_NAME,
        "state_only_minus_state_only_donor": delta,
        "gate": {
            "operator": ">",
            "threshold": 0.0,
            "value": delta,
            "passed": passed,
        },
        "selection_authorized": True,
        "test_selection_forbidden": True,
    }


def fingerprint_payload_sha256(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Manifest fingerprint_payload must be an object")
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Required receipt artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_candidate_lineage_record_binding(
    candidate_lineage: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate_lineage is None:
        return None
    lineage_kind = candidate_lineage.get("lineage_kind")
    if lineage_kind == "scene_hard_failure_train_overfit_selection":
        authorization = candidate_lineage.get("authorization")
        consumption = (
            authorization.get("authorization_consumption")
            if isinstance(authorization, Mapping)
            else None
        )
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("authorization_kind") != lineage_kind
            or authorization.get("scope") != SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE
            or authorization.get("hard32_authorized") is not True
            or authorization.get("full170_authorized") is not False
            or authorization.get("test_authorized") is not False
            or authorization.get("other_benchmarks_authorized") is not False
            or not isinstance(authorization.get("receipt"), Mapping)
            or not isinstance(authorization.get("selected_checkpoint"), Mapping)
            or not isinstance(authorization.get("gate"), Mapping)
            or not isinstance(authorization.get("hard32_output_dir"), str)
            or not isinstance(consumption, Mapping)
            or consumption.get("schema")
            != SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA
            or not isinstance(consumption.get("path"), str)
            or not isinstance(consumption.get("file_sha256"), str)
            or not isinstance(consumption.get("claim_sha256"), str)
        ):
            raise ValueError("Focused train selection authorization lineage differs")
        return {
            "lineage_kind": lineage_kind,
            "lineage_sha256": fingerprint_payload_sha256(candidate_lineage),
            "authorization_scope": SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE,
            "receipt": dict(authorization["receipt"]),
            "selected_checkpoint": dict(authorization["selected_checkpoint"]),
            "evaluation_fingerprint": authorization.get("evaluation_fingerprint"),
            "gate": dict(authorization["gate"]),
            "hard32_output_dir": authorization["hard32_output_dir"],
            "authorization_consumption": dict(consumption),
        }
    if lineage_kind == SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND:
        authorization = candidate_lineage.get("authorization")
        if not isinstance(authorization, Mapping):
            raise ValueError("V15 candidate authorization is missing")
        checkpoint = authorization.get("checkpoint")
        artifacts = (
            checkpoint.get("artifacts") if isinstance(checkpoint, Mapping) else None
        )
        provenance = authorization.get("training_provenance")
        selector_manifest = authorization.get("selector_manifest")
        consumption = authorization.get("authorization_consumption")
        if (
            authorization.get("authorization_kind")
            != SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND
            or authorization.get("scope") != SCENE_V15_HARD32_AUTHORIZATION_SCOPE
            or not isinstance(checkpoint, Mapping)
            or not isinstance(artifacts, Mapping)
            or not isinstance(provenance, Mapping)
            or not isinstance(authorization.get("candidate_lock"), Mapping)
            or not isinstance(authorization.get("selection_receipt"), Mapping)
            or not isinstance(selector_manifest, Mapping)
            or not isinstance(consumption, Mapping)
            or consumption.get("schema")
            != SCENE_V15_HARD32_CONSUMPTION_MARKER_SCHEMA
            or not isinstance(consumption.get("path"), str)
            or not isinstance(consumption.get("file_sha256"), str)
            or not isinstance(consumption.get("claim_sha256"), str)
            or not isinstance(authorization.get("base_model"), str)
        ):
            raise ValueError("V15 candidate authorization lineage differs")
        compact_provenance: dict[str, Any] = {}
        for stage in ("launch", "completion"):
            entry = provenance.get(stage)
            if (
                not isinstance(entry, Mapping)
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("file_sha256"), str)
                or not isinstance(entry.get("receipt_sha256"), str)
            ):
                raise ValueError(f"V15 candidate {stage} receipt binding is missing")
            compact_provenance[stage] = dict(entry)
        return {
            "lineage_kind": SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
            "lineage_sha256": fingerprint_payload_sha256(candidate_lineage),
            "authorization_scope": SCENE_V15_HARD32_AUTHORIZATION_SCOPE,
            "candidate_lock": dict(authorization["candidate_lock"]),
            "selection_receipt": dict(authorization["selection_receipt"]),
            "selection_fingerprint": authorization.get("selection_fingerprint"),
            "selector_manifest": dict(selector_manifest),
            "authorization_consumption": dict(consumption),
            "selected_checkpoint_step": authorization.get(
                "selected_checkpoint_step"
            ),
            "post_save_audit": dict(authorization["post_save_audit"]),
            "base_model": authorization["base_model"],
            "checkpoint": {
                "path": checkpoint.get("path"),
                "checkpoint_step": checkpoint.get("checkpoint_step"),
                "delta_mem_adapter": artifacts.get("delta_mem_adapter.pt"),
                "delta_mem_config": artifacts.get("delta_mem_config.json"),
            },
            "training_provenance": compact_provenance,
        }
    if lineage_kind != SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND:
        return None
    authorization = candidate_lineage.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("V14 candidate authorization is missing")
    checkpoint = authorization.get("checkpoint")
    provenance = authorization.get("training_provenance")
    diagnostic_receipt = authorization.get("diagnostic_receipt")
    candidate_lock = authorization.get("candidate_lock")
    evaluator_code_bridge = authorization.get("evaluator_code_bridge")
    base_model = authorization.get("base_model")
    if (
        authorization.get("authorization_kind")
        != SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
        or authorization.get("scope") != SCENE_V14_HARD32_AUTHORIZATION_SCOPE
        or not isinstance(checkpoint, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(diagnostic_receipt, Mapping)
        or not isinstance(candidate_lock, Mapping)
        or not isinstance(evaluator_code_bridge, Mapping)
        or not isinstance(base_model, str)
    ):
        raise ValueError("V14 candidate authorization lineage differs")
    checkpoint_artifacts = checkpoint.get("artifacts")
    if not isinstance(checkpoint_artifacts, Mapping):
        raise ValueError("V14 candidate checkpoint artifacts are missing")
    compact_provenance: dict[str, Any] = {}
    for stage in ("launch", "completion"):
        entry = provenance.get(stage)
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("artifact"), Mapping)
            or not isinstance(entry.get("receipt_sha256"), str)
        ):
            raise ValueError(f"V14 candidate {stage} receipt binding is missing")
        compact_provenance[stage] = {
            "artifact": dict(entry["artifact"]),
            "receipt_sha256": entry["receipt_sha256"],
        }
    return {
        "lineage_kind": SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "lineage_sha256": fingerprint_payload_sha256(candidate_lineage),
        "authorization_scope": SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "candidate_lock": dict(candidate_lock),
        "diagnostic_receipt": dict(diagnostic_receipt),
        "evaluator_code_bridge": dict(evaluator_code_bridge),
        "base_model": base_model,
        "checkpoint": {
            "memory_dir": checkpoint.get("memory_dir"),
            "global_step": checkpoint.get("global_step"),
            "delta_mem_adapter": checkpoint_artifacts.get("delta_mem_adapter"),
            "delta_mem_config": checkpoint_artifacts.get("delta_mem_config"),
        },
        "training_provenance": compact_provenance,
    }


def _historical_result_binding(path: Path, *, description: str) -> dict[str, Any]:
    _reject_symlink_components(path, description=description)
    if not path.is_file():
        raise ValueError(f"Historical V6 result artifact is missing: {path}")
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_historical_v6_hard32_receipt(
    *,
    output_dir: Path,
    fingerprint: str,
    contract: dict[str, Any],
    candidate_lineage: dict[str, Any],
    code_fingerprint: dict[str, Any],
    repository_revision: dict[str, Any],
    base_model: Path,
    base_model_binding: dict[str, Any],
    dataset_file: Path,
    selection_file: Path,
    memory_dir: Path,
    conditions: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if contract.get("name") != HISTORICAL_V6_HARD32_CONTRACT:
        raise ValueError("Historical V6 receipt requires its protected contract")
    if conditions != list(HISTORICAL_V6_HARD32_CONDITIONS):
        raise ValueError("Historical V6 receipt requires exactly three protected conditions")
    if (
        contract.get("full170_authorized") is not False
        or contract.get("test_authorized") is not False
        or contract.get("checkpoint_selection_authorized") is not False
    ):
        raise ValueError("Historical V6 receipt contract must not authorize escalation")
    current_lineage = validate_historical_v6_checkpoint(memory_dir)
    if candidate_lineage != current_lineage:
        raise ValueError("Historical V6 checkpoint lineage changed during evaluation")
    hard32_bindings = validate_historical_v6_hard32_artifacts(
        dataset_file=dataset_file,
        selection_file=selection_file,
    )
    evaluator_sha256 = sha256_file(Path(__file__))
    if (
        code_fingerprint.get("evaluator_sha256") != evaluator_sha256
        or code_fingerprint != scene_state_code_fingerprint(PROJECT_ROOT)
    ):
        raise ValueError("Historical V6 runtime code changed during evaluation")
    current_base_model_binding = historical_base_model_binding(base_model)
    if base_model_binding != current_base_model_binding:
        raise ValueError("Historical V6 base-model artifacts changed during evaluation")
    current_revision = git_revision(PROJECT_ROOT)
    commit = repository_revision.get("commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
        or repository_revision != current_revision
    ):
        raise ValueError("Historical V6 repository revision changed during evaluation")
    if evidence.get("schema") != "rwkv_ms_scene_historical_v6_hard32_evidence.v1":
        raise ValueError("Historical V6 receipt evidence schema differs")
    tracked_worktree = require_historical_tracked_worktree_clean()
    output_bindings = {
        "manifest": _historical_result_binding(
            output_dir / "manifest.json", description="evaluation manifest"
        ),
        "summary": _historical_result_binding(
            output_dir / "summary.json", description="evaluation summary"
        ),
        "progress": _historical_result_binding(
            output_dir / "progress.json", description="evaluation progress"
        ),
        "conditions": {
            condition: _historical_result_binding(
                output_dir / f"{condition}.jsonl",
                description=f"{condition} results",
            )
            for condition in conditions
        },
    }
    receipt = {
        "schema": HISTORICAL_V6_HARD32_RECEIPT_SCHEMA,
        "issued_at": utc_now(),
        "status": "complete_diagnostic",
        "authorization_scope": HISTORICAL_V6_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "checkpoint_selection_authorized": False,
        "evaluation_fingerprint": fingerprint,
        "contract": contract,
        "checkpoint": current_lineage,
        "base_model": current_base_model_binding,
        "lineage_limitation": HISTORICAL_V6_LINEAGE_LIMITATION,
        "dataset_and_selection": hard32_bindings,
        "code": {
            "repository_revision": repository_revision,
            "tracked_worktree": tracked_worktree,
            "artifacts": code_fingerprint,
        },
        "outputs": output_bindings,
        "evidence": evidence,
    }
    receipt["receipt_sha256"] = fingerprint_payload_sha256(receipt)
    return receipt


def _validate_historical_receipt_bound_file(
    record: Any,
    *,
    description: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Historical V6 receipt {description} binding is missing")
    raw_path = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("bytes")
    if (
        not isinstance(raw_path, str)
        or not isinstance(digest, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ValueError(f"Historical V6 receipt {description} binding is invalid")
    path = Path(raw_path)
    _reject_symlink_components(path, description=f"receipt-bound {description}")
    if (
        not path.is_file()
        or path.stat().st_size != byte_count
        or sha256_file(path) != digest
    ):
        raise ValueError(f"Historical V6 receipt {description} artifact differs")


def validate_historical_v6_hard32_receipt(
    receipt_path: Path,
    *,
    fingerprint: str,
    memory_dir: Path,
    base_model: Path,
    base_model_binding: dict[str, Any],
    dataset_file: Path,
    selection_file: Path,
    code_fingerprint: dict[str, Any],
    repository_revision: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    _reject_symlink_components(receipt_path, description="historical receipt")
    if not receipt_path.is_file():
        raise ValueError(f"Historical V6 receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Historical V6 receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("Historical V6 receipt must contain an object")
    unsigned = dict(receipt)
    recorded_sha256 = unsigned.pop("receipt_sha256", None)
    if recorded_sha256 != fingerprint_payload_sha256(unsigned):
        raise ValueError("Historical V6 receipt checksum differs")
    if (
        receipt.get("schema") != HISTORICAL_V6_HARD32_RECEIPT_SCHEMA
        or receipt.get("status") != "complete_diagnostic"
        or receipt.get("authorization_scope")
        != HISTORICAL_V6_HARD32_AUTHORIZATION_SCOPE
        or receipt.get("full170_authorized") is not False
        or receipt.get("test_authorized") is not False
        or receipt.get("checkpoint_selection_authorized") is not False
        or receipt.get("evaluation_fingerprint") != fingerprint
    ):
        raise ValueError("Historical V6 receipt scope or fingerprint differs")
    contract = receipt.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("name") != HISTORICAL_V6_HARD32_CONTRACT
        or contract.get("conditions") != list(HISTORICAL_V6_HARD32_CONDITIONS)
        or contract.get("rows") != len(HARD32_ROW_INDICES)
    ):
        raise ValueError("Historical V6 receipt contract differs")
    if receipt.get("checkpoint") != validate_historical_v6_checkpoint(memory_dir):
        raise ValueError("Historical V6 receipt checkpoint binding differs")
    current_base_model_binding = historical_base_model_binding(base_model)
    if (
        base_model_binding != current_base_model_binding
        or receipt.get("base_model") != current_base_model_binding
    ):
        raise ValueError("Historical V6 receipt base-model binding differs")
    if receipt.get("dataset_and_selection") != validate_historical_v6_hard32_artifacts(
        dataset_file=dataset_file,
        selection_file=selection_file,
    ):
        raise ValueError("Historical V6 receipt dataset binding differs")
    if receipt.get("lineage_limitation") != HISTORICAL_V6_LINEAGE_LIMITATION:
        raise ValueError("Historical V6 receipt lineage limitation differs")
    code = receipt.get("code")
    tracked_worktree = require_historical_tracked_worktree_clean()
    if (
        not isinstance(code, dict)
        or code.get("artifacts") != code_fingerprint
        or code_fingerprint.get("evaluator_sha256") != sha256_file(Path(__file__))
        or code_fingerprint != scene_state_code_fingerprint(PROJECT_ROOT)
        or code.get("repository_revision") != repository_revision
        or repository_revision != git_revision(PROJECT_ROOT)
        or code.get("tracked_worktree") != tracked_worktree
    ):
        raise ValueError("Historical V6 receipt code binding differs")
    if receipt.get("evidence") != evidence:
        raise ValueError("Historical V6 receipt evidence differs")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Historical V6 receipt output bindings are missing")
    for name in ("manifest", "summary", "progress"):
        _validate_historical_receipt_bound_file(
            outputs.get(name), description=name
        )
    condition_outputs = outputs.get("conditions")
    if (
        not isinstance(condition_outputs, dict)
        or list(condition_outputs) != list(HISTORICAL_V6_HARD32_CONDITIONS)
    ):
        raise ValueError("Historical V6 receipt condition outputs differ")
    for condition in HISTORICAL_V6_HARD32_CONDITIONS:
        _validate_historical_receipt_bound_file(
            condition_outputs[condition], description=f"{condition} output"
        )
    return {
        "path": str(receipt_path.resolve()),
        "file_sha256": sha256_file(receipt_path),
        "payload_sha256": recorded_sha256,
        "evaluation_fingerprint": fingerprint,
        "authorization_scope": HISTORICAL_V6_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
    }


def build_hard32_receipt(
    *,
    output_dir: Path,
    fingerprint: str,
    contract: dict[str, Any],
    candidate_lineage: dict[str, Any],
    code_fingerprint: dict[str, Any],
    dataset_file: Path,
    selection_file: Path,
    donor_mapping: list[dict[str, Any]],
    shuffled_mapping: list[dict[str, Any]] | None = None,
    gate: dict[str, Any],
    semantic_evidence: dict[str, Any],
    base_outcome_evidence: dict[str, Any] | None,
    memory_dir: Path,
    conditions: list[str],
) -> dict[str, Any]:
    if not is_identity_hard32_contract(contract):
        raise ValueError("Hard32 receipt requires the protected hard32 contract")
    expected_conditions = (
        list(SCENE_FOCUSED_CONDITIONS)
        if contract.get("name") == SCENE_HARD_FAILURE_HARD32_CONTRACT
        else (
            list(SCENE_V15_HARD32_CONDITIONS)
            if contract.get("name") == SCENE_V15_HARD32_CONTRACT
            else list(CONDITIONS)
        )
    )
    if conditions != expected_conditions:
        raise ValueError(
            "Hard32 receipt conditions differ from the protected contract"
        )
    donor_mapping_sha256 = sha256_text(
        json.dumps(donor_mapping, sort_keys=True, separators=(",", ":"))
    )
    if donor_mapping_sha256 != HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256:
        raise ValueError("Hard32 receipt donor mapping differs from the frozen optimum")
    is_v7_authorized = (
        candidate_lineage.get("lineage_kind")
        == SCENE_V7_TRAIN32_AUTHORIZATION_KIND
    )
    is_v8_authorized = (
        candidate_lineage.get("lineage_kind")
        == SCENE_V8_TRAIN32_AUTHORIZATION_KIND
    )
    is_v14_authorized = (
        candidate_lineage.get("lineage_kind")
        == SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
    )
    is_v15_authorized = (
        candidate_lineage.get("lineage_kind")
        == SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND
    )
    is_focused_authorized = (
        candidate_lineage.get("lineage_kind")
        == "scene_hard_failure_train_overfit_selection"
    )
    receipt_gate = copy.deepcopy(gate)
    if (
        is_v7_authorized
        or is_v8_authorized
        or is_v14_authorized
        or is_v15_authorized
        or is_focused_authorized
    ):
        receipt_gate["full170_authorized_for_bound_checkpoint"] = False
        receipt_gate["authorization_scope"] = (
            SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE
            if is_focused_authorized
            else (
                SCENE_V14_HARD32_AUTHORIZATION_SCOPE
                if is_v8_authorized or is_v14_authorized or is_v15_authorized
                else "fixed_hard32_only_no_full170"
            )
        )
        receipt_gate["other_benchmarks_authorized"] = False
    output_bindings = {
        "manifest": file_binding(output_dir / "manifest.json"),
        "summary": file_binding(output_dir / "summary.json"),
        "conditions": {
            condition: file_binding(output_dir / f"{condition}.jsonl")
            for condition in conditions
        },
    }
    candidate_lineage_binding = build_candidate_lineage_record_binding(
        candidate_lineage
    )
    receipt = {
        "schema": HARD32_RECEIPT_SCHEMA,
        "issued_at": utc_now(),
        "status": gate["status"],
        "contract": contract,
        "evaluation_fingerprint": fingerprint,
        "checkpoint": {
            "memory_dir": str(memory_dir.expanduser().resolve()),
            "adapter_sha256": sha256_file(memory_dir / "delta_mem_adapter.pt"),
            "config_sha256": sha256_file(memory_dir / "delta_mem_config.json"),
            "candidate_lineage": candidate_lineage,
            "candidate_lineage_binding": candidate_lineage_binding,
        },
        "objective_interpretation": (
            SCENE_V15_HARD32_OBJECTIVE_INTERPRETATION
            if is_v15_authorized
            else (
                SCENE_V14_HARD32_OBJECTIVE_INTERPRETATION
                if is_v14_authorized
                else (
                    SCENE_V8_HARD32_OBJECTIVE_INTERPRETATION
                    if is_v8_authorized
                    else (
                        SCENE_V7_HARD32_OBJECTIVE_INTERPRETATION
                        if is_v7_authorized
                        else SCENE_V6_IDENTITY_OBJECTIVE_INTERPRETATION
                    )
                )
            )
        ),
        **(
            {
                "authorization_scope": (
                    SCENE_HARD_FAILURE_AUTHORIZATION_SCOPE
                    if is_focused_authorized
                    else (
                        SCENE_V14_HARD32_AUTHORIZATION_SCOPE
                        if is_v8_authorized or is_v14_authorized or is_v15_authorized
                        else "fixed_hard32_only_no_full170"
                    )
                ),
                "upstream_authorization_kind": (
                    "scene_hard_failure_train_overfit_selection"
                    if is_focused_authorized
                    else (
                        SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND
                        if is_v15_authorized
                        else (
                            SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
                            if is_v14_authorized
                            else (
                                SCENE_V8_TRAIN32_AUTHORIZATION_KIND
                                if is_v8_authorized
                                else SCENE_V7_TRAIN32_AUTHORIZATION_KIND
                            )
                        )
                    )
                ),
                "full170_authorized": False,
                "test_authorized": False,
                "other_benchmarks_authorized": False,
            }
            if is_v7_authorized
            or is_v8_authorized
            or is_v14_authorized
            or is_v15_authorized
            or is_focused_authorized
            else {}
        ),
        "dataset": {
            "path": str(dataset_file.expanduser().resolve()),
            "sha256": sha256_file(dataset_file),
            "revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
            "split": "val",
        },
        "selection": {
            **file_binding(selection_file),
            "rows": [
                {"source_index": index, "row_sha256": HARD32_ROW_HASHES[index]}
                for index in HARD32_ROW_INDICES
            ],
            "holdout_sha256": HARD32_HOLDOUT_SHA256,
            "pair_manifest_sha256": HARD32_PAIR_MANIFEST_SHA256,
        },
        "donor_mapping": {
            "rule": DONOR_RULE_LENGTH_MATCHED,
            "protected_pairing": "global_minimum_bottleneck_then_total_v1",
            "frozen_pairs": [list(pair) for pair in HARD32_FROZEN_DONOR_PAIRS],
            "frozen_pairs_sha256": HARD32_FROZEN_DONOR_PAIRS_SHA256,
            "maximum_write_token_difference": (
                HARD32_FROZEN_DONOR_MAX_WRITE_TOKEN_DIFFERENCE
            ),
            "total_write_token_difference": (
                HARD32_FROZEN_DONOR_TOTAL_WRITE_TOKEN_DIFFERENCE
            ),
            "cardinality_stratum_rows": HARD32_FROZEN_DONOR_STRATUM_ROWS,
            "rows": donor_mapping,
            "sha256": donor_mapping_sha256,
        },
        **(
            {
                "shuffled_mapping": {
                    "rule": (
                        "deterministic_label_distinct_non_donor_permutation_v1"
                    ),
                    "rows": list(shuffled_mapping or []),
                    "sha256": _canonical_sha256(list(shuffled_mapping or [])),
                }
            }
            if contract.get("name") == SCENE_HARD_FAILURE_HARD32_CONTRACT
            else {}
        ),
        "code": code_fingerprint,
        "outputs": output_bindings,
        "gate": receipt_gate,
        "semantic_evidence": semantic_evidence,
        "base_outcome_evidence": base_outcome_evidence,
    }
    receipt["receipt_sha256"] = fingerprint_payload_sha256(receipt)
    return receipt


def _validate_bound_file(record: Any, *, description: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Hard32 receipt {description} binding is missing")
    raw_path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise ValueError(f"Hard32 receipt {description} binding is invalid")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Hard32 receipt {description} artifact differs")
    return path


def validate_scene_v7_train32_hard32_authorization(
    receipt_path: Path,
    *,
    memory_dir: Path,
) -> dict[str, Any]:
    """Validate the V7 Train32 gate for one exact fixed-Hard32 candidate."""

    # This import must stay lazy: the V7 evaluator reuses this module's runtime.
    from experiments.rethinking_rwkv_ms_gemma import run_scene_train32_eval as v7

    expanded_receipt = receipt_path.expanduser()
    if expanded_receipt.is_symlink():
        raise ValueError("V7 Train32 receipt must not be a symlink")
    resolved_receipt = expanded_receipt.resolve()
    if not resolved_receipt.is_file():
        raise ValueError(f"V7 Train32 receipt is missing: {resolved_receipt}")
    try:
        payload = json.loads(resolved_receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("V7 Train32 receipt is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("V7 Train32 receipt must contain an object")
    if payload.get("contract") != "scene_v7_train32_overfit":
        raise ValueError("Fixed Hard32 requires a Train32 receipt, not a Tiny2 receipt")
    gate = payload.get("gate")
    if not isinstance(gate, dict) or gate.get("status") != "pass":
        raise ValueError("Fixed Hard32 requires a passed V7 Train32 receipt")

    input_artifacts = payload.get("input_artifacts")
    expected_artifact_names = {
        "dataset",
        "row_manifest",
        "pair_manifest",
        "source_manifest",
    }
    if not isinstance(input_artifacts, dict) or set(input_artifacts) != expected_artifact_names:
        raise ValueError("V7 Train32 receipt input-artifact bindings differ")

    artifact_paths: dict[str, Path] = {}
    artifact_sha256: dict[str, str] = {}
    for name in sorted(expected_artifact_names):
        binding = input_artifacts[name]
        if not isinstance(binding, dict):
            raise ValueError(f"V7 Train32 receipt {name} binding is invalid")
        raw_path = binding.get("path")
        digest = binding.get("actual_sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"V7 Train32 receipt {name} path is invalid")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"V7 Train32 receipt {name} SHA-256 is invalid")
        artifact_paths[name] = Path(raw_path)
        artifact_sha256[name] = digest

    input_contract = v7.validate_v7_contract(
        contract="scene_v7_train32_overfit",
        dataset_file=artifact_paths["dataset"],
        row_manifest_file=artifact_paths["row_manifest"],
        pair_manifest_file=artifact_paths["pair_manifest"],
        source_manifest_file=artifact_paths["source_manifest"],
        expected_dataset_sha256=artifact_sha256["dataset"],
        expected_row_manifest_sha256=artifact_sha256["row_manifest"],
        expected_pair_manifest_sha256=artifact_sha256["pair_manifest"],
        expected_source_manifest_sha256=artifact_sha256["source_manifest"],
        source_lock_file=v7.DEFAULT_SOURCE_LOCK,
    )
    checkpoint = v7.validate_v7_checkpoint(
        memory_dir,
        input_contract=input_contract,
    )
    validated = v7.validate_fixed_hard32_authorization(
        resolved_receipt,
        expected_checkpoint=checkpoint,
    )
    return {
        "authorization_kind": SCENE_V7_TRAIN32_AUTHORIZATION_KIND,
        "contract": "scene_v7_train32_overfit",
        "scope": "fixed_hard32_only_no_full170",
        "receipt": {
            "path": str(resolved_receipt),
            "file_sha256": sha256_file(resolved_receipt),
            "payload_sha256": validated["receipt_sha256"],
            "evaluation_fingerprint": validated["evaluation_fingerprint"],
        },
        "checkpoint": checkpoint,
        "source_lock": validated["source_lock"],
    }


def validate_scene_v8_train32_hard32_authorization(
    receipt_path: Path,
    *,
    memory_dir: Path,
    dataset_file: Path,
    selection_file: Path | None,
) -> dict[str, Any]:
    """Authorize only fixed Hard32 after the exact V8 checkpoint gate passes.

    Receipt and checkpoint validation intentionally happens before any call that
    resolves, opens, or hashes the official validation/Hard32 artifacts.
    """

    from experiments.rethinking_rwkv_ms_gemma import (  # lazy circular import
        run_scene_memory_v8_gate as v8,
    )

    validated = v8.validate_gate_receipt_for_checkpoint(
        receipt_path,
        memory_dir=memory_dir,
    )
    if selection_file is None:
        raise ValueError(
            f"{SCENE_V8_HARD32_CONTRACT} requires the frozen selection file"
        )

    # Protected files are touched only after all train-derived authorization
    # evidence above has passed.
    hard32 = validate_historical_v6_hard32_artifacts(
        dataset_file=dataset_file,
        selection_file=selection_file,
    )
    protected = validated["benchmark_selection_lock"]["protected_benchmark"]
    expected = {
        "official_val_sha256": OFFICIAL_SCENE_V4_VAL_SHA256,
        "selection_sha256": HARD32_SELECTION_SHA256,
        "holdout_sha256": HARD32_HOLDOUT_SHA256,
        "pair_manifest_sha256": HARD32_PAIR_MANIFEST_SHA256,
        "rows": 32,
        "authorization_scope": SCENE_V8_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    mismatches = [name for name, value in expected.items() if protected.get(name) != value]
    if mismatches:
        raise ValueError(
            "V8 receipt frozen Hard32 binding differs: " + ", ".join(mismatches)
        )
    return {
        "authorization_kind": SCENE_V8_TRAIN32_AUTHORIZATION_KIND,
        "contract": v8.GATE_CONTRACT,
        "scope": SCENE_V8_HARD32_AUTHORIZATION_SCOPE,
        "receipt": {
            "path": validated["receipt_path"],
            "file_sha256": validated["receipt_file_sha256"],
            "payload_sha256": validated["receipt_sha256"],
            "evaluation_fingerprint": validated["evaluation_fingerprint"],
        },
        "checkpoint": validated["checkpoint"],
        "benchmark_selection_lock": validated["benchmark_selection_lock"],
        "frozen_hard32": hard32,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def validate_scene_v14_candidate_lock(
    lock_path: Path,
) -> dict[str, Any]:
    """Load the one canonical tracked V14 Hard32 candidate designation."""

    supplied = lock_path.expanduser().absolute()
    canonical = SCENE_V14_CANDIDATE_LOCK_PATH.absolute()
    if supplied != canonical:
        raise ValueError(
            "V14 Hard32 requires the canonical tracked checkpoint-3 candidate lock"
        )
    _reject_symlink_components(supplied, description="V14 candidate lock")
    if not supplied.is_file():
        raise ValueError(f"V14 candidate lock is missing: {supplied}")
    file_sha256 = sha256_file(supplied)
    if file_sha256 != SCENE_V14_CANDIDATE_LOCK_FILE_SHA256:
        raise ValueError("V14 candidate lock file hash differs")
    try:
        payload = json.loads(supplied.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("V14 candidate lock is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("V14 candidate lock must contain an object")
    required_keys = {
        "schema",
        "authorization_kind",
        "selection_policy",
        "candidate_count",
        "rejected_checkpoint_steps",
        "hard32_output_dir",
        "selected_candidate",
        "authorization",
        "exception_reason",
        "lock_sha256",
    }
    if set(payload) != required_keys:
        raise ValueError("V14 candidate lock fields differ")
    if payload.get("schema") != SCENE_V14_CANDIDATE_LOCK_SCHEMA:
        raise ValueError("V14 candidate lock schema differs")
    if (
        payload.get("authorization_kind")
        != SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
        or payload.get("selection_policy")
        != SCENE_V14_CANDIDATE_SELECTION_POLICY
        or payload.get("candidate_count") != 1
        or payload.get("rejected_checkpoint_steps") != [1, 2, 4]
    ):
        raise ValueError("V14 candidate designation differs")
    recorded_sha256 = payload.get("lock_sha256")
    if (
        not isinstance(recorded_sha256, str)
        or SHA256_RE.fullmatch(recorded_sha256) is None
        or recorded_sha256
        != fingerprint_payload_sha256(
            {key: value for key, value in payload.items() if key != "lock_sha256"}
        )
    ):
        raise ValueError("V14 candidate lock self-hash differs")
    if payload.get("authorization") != {
        "scope": SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }:
        raise ValueError("V14 candidate lock authorization scope differs")
    if payload.get("exception_reason") != SCENE_V14_EVALUATOR_BRIDGE_REASON:
        raise ValueError("V14 candidate lock exception reason differs")
    selected = payload.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("V14 candidate lock selected candidate is missing")
    if set(selected) != {
        "checkpoint",
        "diagnostic_receipt",
        "legacy_gate",
        "launch_receipt",
        "completion_receipt",
        "historical_state_runtime",
    }:
        raise ValueError("V14 candidate lock selected-candidate fields differ")
    checkpoint = selected.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("global_step") != 3
        or checkpoint.get("consumed_pair_presentations") != 21
        or Path(str(checkpoint.get("memory_dir", ""))).name != "checkpoint-3"
    ):
        raise ValueError("V14 candidate lock must designate only checkpoint-3")
    for name in (
        "delta_mem_adapter",
        "delta_mem_config",
        "trainer_state",
    ):
        binding = checkpoint.get(name)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("bytes"), int)
            or not isinstance(binding.get("sha256"), str)
            or SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError(f"V14 candidate lock {name} binding differs")
    for name in ("diagnostic_receipt", "launch_receipt", "completion_receipt"):
        binding = selected.get(name)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("bytes"), int)
            or not isinstance(binding.get("file_sha256"), str)
            or SHA256_RE.fullmatch(binding["file_sha256"]) is None
            or not isinstance(binding.get("payload_sha256"), str)
            or SHA256_RE.fullmatch(binding["payload_sha256"]) is None
        ):
            raise ValueError(f"V14 candidate lock {name} binding differs")
    diagnostic_receipt = selected["diagnostic_receipt"]
    if (
        diagnostic_receipt.get("status") != "fail"
        or not isinstance(diagnostic_receipt.get("evaluation_fingerprint"), str)
        or SHA256_RE.fullmatch(diagnostic_receipt["evaluation_fingerprint"]) is None
    ):
        raise ValueError("V14 candidate diagnostic receipt status differs")
    if selected.get("legacy_gate") != {
        "status": "fail",
        "all_gates_passed": False,
        "hard32_authorized": False,
        "correct_strict_exact_rows": 6,
        "donor_identity_strict_exact_rows": 6,
        "correct_donor_semantic_switch_rows": 4,
        "role": "diagnostic_only_superseded_by_exact_candidate_lock",
    }:
        raise ValueError("V14 candidate lock must preserve the failed legacy gate")
    historical_runtime = selected.get("historical_state_runtime")
    if (
        not isinstance(historical_runtime, Mapping)
        or not isinstance(historical_runtime.get("path"), str)
        or not isinstance(historical_runtime.get("bytes"), int)
        or not isinstance(historical_runtime.get("sha256"), str)
        or SHA256_RE.fullmatch(historical_runtime["sha256"]) is None
    ):
        raise ValueError("V14 historical evaluator binding differs")
    return {
        "payload": payload,
        "binding": {
            "path": str(supplied),
            "bytes": supplied.stat().st_size,
            "file_sha256": file_sha256,
            "payload_sha256": recorded_sha256,
            "selection_policy": SCENE_V14_CANDIDATE_SELECTION_POLICY,
        },
    }


def _validate_scene_v14_bound_artifact(
    path: Path,
    binding: Mapping[str, Any],
    *,
    description: str,
) -> Path:
    supplied = path.expanduser().absolute()
    expected = Path(str(binding.get("path", ""))).expanduser().absolute()
    if supplied != expected or not supplied.is_file():
        raise ValueError(f"V14 {description} path differs")
    if (
        supplied.stat().st_size != binding.get("bytes")
        or sha256_file(supplied) != binding.get("file_sha256")
    ):
        raise ValueError(f"V14 {description} artifact differs")
    return supplied


def _replay_scene_v14_locked_diagnostic_receipt(
    *,
    v14: Any,
    receipt_path: Path,
    lock_payload: Mapping[str, Any],
    launch_receipt: Path,
    completion_receipt: Path,
    memory_dir: Path,
    input_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay one old receipt after bridging only its evaluator self-binding."""

    selected = lock_payload["selected_candidate"]
    receipt_binding = selected["diagnostic_receipt"]
    resolved_receipt = _validate_scene_v14_bound_artifact(
        receipt_path,
        receipt_binding,
        description="checkpoint-3 diagnostic receipt",
    )
    try:
        original = json.loads(resolved_receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("V14 diagnostic receipt is invalid JSON") from exc
    if not isinstance(original, dict):
        raise ValueError("V14 diagnostic receipt must contain an object")
    original_payload_sha256 = original.get("receipt_sha256")
    if (
        original_payload_sha256 != receipt_binding.get("payload_sha256")
        or original_payload_sha256
        != v14.self_hash_payload(original, hash_field="receipt_sha256")
    ):
        raise ValueError("V14 diagnostic receipt self-hash differs")
    if (
        original.get("schema") != v14.GATE_RECEIPT_SCHEMA
        or original.get("status") != "fail"
    ):
        raise ValueError("V14 locked diagnostic receipt must preserve status=fail")

    historical_code = original.get("code")
    historical_runtime = selected.get("historical_state_runtime")
    live_code = v14.evaluator_code_binding()
    if (
        not isinstance(historical_code, Mapping)
        or historical_code.get("state_runtime") != historical_runtime
        or not isinstance(live_code, Mapping)
        or not isinstance(live_code.get("state_runtime"), Mapping)
    ):
        raise ValueError("V14 historical evaluator binding differs")
    historical_other = dict(historical_code)
    live_other = dict(live_code)
    historical_other.pop("state_runtime")
    live_other.pop("state_runtime")
    if historical_other != live_other:
        raise ValueError("V14 diagnostic receipt has non-evaluator code drift")

    replay_code = dict(live_code)
    replay_code["state_runtime"] = dict(historical_runtime)
    if replay_code != historical_code:
        raise ValueError("V14 diagnostic replay code binding differs")
    live_binding_function = v14.evaluator_code_binding
    try:
        v14.evaluator_code_binding = lambda: dict(replay_code)
        validated = v14.validate_gate_receipt_for_checkpoint(
            original,
            memory_dir=memory_dir,
            launch_receipt=launch_receipt,
            completion_receipt=completion_receipt,
            input_contract=input_contract,
        )
    finally:
        v14.evaluator_code_binding = live_binding_function
    bridge = {
        "exception_reason": lock_payload["exception_reason"],
        "substitution_scope": (
            "diagnostic_replay.evaluator_code_binding.state_runtime_only"
        ),
        "historical_state_runtime": dict(historical_runtime),
        "live_state_runtime": dict(live_code["state_runtime"]),
        "other_code_bindings_unchanged": True,
        "original_receipt_payload_sha256": original_payload_sha256,
        "original_evaluation_fingerprint": original.get("evaluation_fingerprint"),
        "original_receipt_replayed_unchanged": True,
    }
    return validated, bridge


def validate_scene_v14_candidate_hard32_authorization(
    candidate_lock_path: Path,
    receipt_path: Path,
    *,
    launch_receipt: Path,
    completion_receipt: Path,
    base_model: Path | str,
    memory_dir: Path,
    output_dir: Path,
    overwrite: bool,
    dataset_file: Path,
    selection_file: Path | None,
) -> dict[str, Any]:
    """Authorize frozen Hard32 for exactly the tracked V14 checkpoint-3."""

    from experiments.rethinking_rwkv_ms_gemma import (  # lazy circular import
        run_scene_memory_v14_gate as v14,
    )

    lock = validate_scene_v14_candidate_lock(candidate_lock_path)
    lock_payload = lock["payload"]
    expected_output = Path(str(lock_payload["hard32_output_dir"])).expanduser().absolute()
    if output_dir.expanduser().absolute() != expected_output:
        raise ValueError("V14 candidate lock permits only its exact Hard32 output directory")
    if overwrite:
        raise ValueError("V14 candidate-lock Hard32 evaluation forbids --overwrite")
    if v14.HARD32_AUTHORIZATION_SCOPE != SCENE_V14_HARD32_AUTHORIZATION_SCOPE:
        raise ValueError("V14 gate and Hard32 authorization scopes differ")
    if (
        v14.HARD32_AUTHORIZATION_KIND
        != SCENE_V14_VALUE14_DIAGNOSTIC_RECEIPT_KIND
    ):
        raise ValueError("V14 diagnostic receipt kind differs")
    inputs = v14.validate_v14_train_inputs()
    validated_base_model = v14.validate_base_model_path(base_model)
    selected = lock_payload["selected_candidate"]
    if memory_dir.expanduser().absolute() != Path(
        selected["checkpoint"]["memory_dir"]
    ).expanduser().absolute():
        raise ValueError("V14 candidate lock rejects every checkpoint except checkpoint-3")
    _validate_scene_v14_bound_artifact(
        launch_receipt,
        selected["launch_receipt"],
        description="launch receipt",
    )
    _validate_scene_v14_bound_artifact(
        completion_receipt,
        selected["completion_receipt"],
        description="completion receipt",
    )
    validated, code_bridge = _replay_scene_v14_locked_diagnostic_receipt(
        v14=v14,
        receipt_path=receipt_path,
        lock_payload=lock_payload,
        launch_receipt=launch_receipt,
        completion_receipt=completion_receipt,
        memory_dir=memory_dir,
        input_contract=inputs,
    )
    gate = validated.get("gate")
    candidate = validated.get("candidate_designation")
    authorization = validated.get("training_authorization")
    if (
        validated.get("status") != "fail"
        or not isinstance(gate, Mapping)
        or gate.get("status") != "fail"
        or gate.get("all_gates_passed") is not False
        or gate.get("hard32_authorized") is not False
        or gate.get("full170_authorized") is not False
        or gate.get("test_authorized") is not False
        or gate.get("other_benchmarks_authorized") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("kind") != v14.CANDIDATE_DESIGNATION_KIND
        or candidate.get("designated") is not False
        or not isinstance(authorization, Mapping)
        or authorization.get("hard32_authorized") is not False
        or authorization.get("full170_authorized") is not False
        or authorization.get("test_authorized") is not False
        or authorization.get("other_benchmarks_authorized") is not False
    ):
        raise ValueError(
            "V14 candidate lock requires its exact failed diagnostic receipt with "
            "no benchmark authorization encoded in that historical receipt"
        )
    checkpoint = validated.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or candidate.get("checkpoint_binding") != checkpoint
        or checkpoint.get("training_provenance")
        != validated.get("training_provenance")
    ):
        raise ValueError("V14 diagnostic receipt candidate or training lineage differs")
    locked_checkpoint = selected["checkpoint"]
    expected_checkpoint = {
        "memory_dir": checkpoint.get("memory_dir"),
        "global_step": checkpoint.get("global_step"),
        "consumed_pair_presentations": checkpoint.get("consumed_pair_presentations"),
        "delta_mem_adapter": checkpoint.get("artifacts", {}).get("delta_mem_adapter"),
        "delta_mem_config": checkpoint.get("artifacts", {}).get("delta_mem_config"),
        "trainer_state": checkpoint.get("artifacts", {}).get("trainer_state"),
    }
    if locked_checkpoint != expected_checkpoint:
        raise ValueError("V14 candidate lock checkpoint-3 artifact binding differs")
    if (
        validated.get("evaluation_fingerprint")
        != selected["diagnostic_receipt"]["evaluation_fingerprint"]
    ):
        raise ValueError("V14 candidate lock evaluation fingerprint differs")
    provenance = validated.get("training_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V14 diagnostic receipt training provenance is missing")
    for stage in ("launch", "completion"):
        entry = provenance.get(stage)
        locked = selected[f"{stage}_receipt"]
        if (
            not isinstance(entry, Mapping)
            or entry.get("artifact", {}).get("path") != locked["path"]
            or entry.get("artifact", {}).get("bytes") != locked["bytes"]
            or entry.get("artifact", {}).get("sha256") != locked["file_sha256"]
            or entry.get("receipt_sha256") != locked["payload_sha256"]
        ):
            raise ValueError(f"V14 candidate lock {stage} provenance differs")
    benchmark_lock = inputs.get("benchmark_lock")
    if (
        not isinstance(benchmark_lock, Mapping)
        or benchmark_lock.get("selected_task") != TASK_NAME
        or benchmark_lock.get("authorization_scope")
        != SCENE_V14_HARD32_AUTHORIZATION_SCOPE
    ):
        raise ValueError("V14 benchmark-selection lock differs")
    protected = (
        benchmark_lock.get("protected_benchmark")
        if isinstance(benchmark_lock, Mapping)
        else None
    )
    expected_protected = {
        "official_val_path": str(HISTORICAL_V6_OFFICIAL_VAL.resolve()),
        "official_val_sha256": OFFICIAL_SCENE_V4_VAL_SHA256,
        "official_dataset_revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
        "selection_path": str(HISTORICAL_V6_HARD32_SELECTION.resolve()),
        "selection_sha256": HARD32_SELECTION_SHA256,
        "holdout_path": str(HISTORICAL_V6_HARD32_HOLDOUT.resolve()),
        "holdout_sha256": HARD32_HOLDOUT_SHA256,
        "pair_manifest_sha256": HARD32_PAIR_MANIFEST_SHA256,
        "rows": 32,
        "split": "val",
        "authorization_scope": SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    mismatches = [
        name
        for name, expected in expected_protected.items()
        if not isinstance(protected, Mapping) or protected.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "V14 receipt frozen Hard32 binding differs: " + ", ".join(mismatches)
        )
    if selection_file is None:
        raise ValueError(
            f"{SCENE_V14_HARD32_CONTRACT} requires the frozen selection file"
        )

    hard32 = validate_historical_v6_hard32_artifacts(
        dataset_file=dataset_file,
        selection_file=selection_file,
    )
    receipt_file = validated.get("receipt_path")
    receipt_file_sha256 = validated.get("receipt_file_sha256")
    if not isinstance(receipt_file, str) or not isinstance(receipt_file_sha256, str):
        receipt_file = selected["diagnostic_receipt"]["path"]
        receipt_file_sha256 = selected["diagnostic_receipt"]["file_sha256"]
    return {
        "authorization_kind": SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "contract": v14.GATE_CONTRACT,
        "scope": SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "candidate_lock": dict(lock["binding"]),
        "diagnostic_receipt": {
            "path": receipt_file,
            "file_sha256": receipt_file_sha256,
            "payload_sha256": selected["diagnostic_receipt"]["payload_sha256"],
            "evaluation_fingerprint": validated["evaluation_fingerprint"],
            "status": "fail",
        },
        "base_model": str(validated_base_model),
        "evaluator_code_bridge": code_bridge,
        "hard32_output_dir": str(expected_output),
        "checkpoint": dict(checkpoint),
        "training_provenance": dict(provenance),
        "benchmark_selection_lock": dict(benchmark_lock),
        "frozen_hard32": hard32,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def _lexical_absolute(path: Path | str) -> Path:
    """Normalize a CLI path without resolving or probing its filesystem target."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def scene_hard_failure_consumption_marker_path(
    selection_receipt_path: Path,
) -> Path:
    selection_path = _lexical_absolute(selection_receipt_path)
    return (
        selection_path.parent
        / SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_FILENAME
    )


def claim_scene_hard_failure_hard32_authorization(
    *,
    selection_receipt_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Irrevocably consume one focused train-selection Hard32 authorization."""

    marker_path = scene_hard_failure_consumption_marker_path(
        selection_receipt_path
    )
    marker_payload = dict(payload)
    marker_payload["schema"] = (
        SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA
    )
    marker_payload["claim_sha256"] = fingerprint_payload_sha256(marker_payload)
    serialized = json.dumps(marker_payload, sort_keys=True, separators=(",", ":"))
    encoded = serialized.encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker_path, flags, 0o400)
    except FileExistsError as exc:
        raise ValueError(
            "Focused Hard32 authorization was already consumed; exactly-once "
            f"evaluation forbids replay or another output: {marker_path}"
        ) from exc
    try:
        offset = 0
        while offset < len(encoded):
            try:
                written = os.write(descriptor, encoded[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError(
                    "Focused Hard32 consumption marker write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(marker_path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {
        "path": str(marker_path),
        "bytes": len(encoded),
        "file_sha256": sha256_text(serialized),
        "claim_sha256": marker_payload["claim_sha256"],
        "schema": SCENE_HARD_FAILURE_HARD32_CONSUMPTION_MARKER_SCHEMA,
    }


def validate_focused_hard32_exactly_once_authorization(
    *,
    selection_receipt_path: Path,
    memory_dir: Path,
    output_dir: Path,
    overwrite: bool,
    dataset_file: Path,
    selection_file: Path | None,
    base_model: Path | str,
    device: str,
    dtype: str,
    attn_implementation: str,
    delta_mem_root: Path | str,
    conditions: list[str],
    donor_rule: str,
    max_new_tokens: int,
    normal_fusion_profile: str,
    expected_memory_layer_count: int | None,
    inline_row_indices: str | None,
    preflight_only: bool,
) -> dict[str, Any]:
    """Validate and exclusively claim focused Hard32 before protected access."""

    if preflight_only:
        raise ValueError(
            "Focused Hard32 forbids --preflight-only because protected access must "
            "consume its exactly-once authorization"
        )
    if overwrite:
        raise ValueError("Focused Hard32 evaluation forbids --overwrite")
    if inline_row_indices is not None or selection_file is None:
        raise ValueError(
            "Focused Hard32 requires only the frozen --row-indices-file; inline row "
            "indices are forbidden"
        )
    if conditions != list(SCENE_FOCUSED_CONDITIONS):
        raise ValueError(
            f"{SCENE_HARD_FAILURE_HARD32_CONTRACT} requires conditions in exact order: "
            + ",".join(SCENE_FOCUSED_CONDITIONS)
        )
    if donor_rule != DONOR_RULE_LENGTH_MATCHED:
        raise ValueError(
            f"Focused Hard32 requires donor_rule={DONOR_RULE_LENGTH_MATCHED}"
        )
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(
            f"Focused Hard32 requires max_new_tokens={DEFAULT_MAX_NEW_TOKENS}"
        )
    if normal_fusion_profile != "native":
        raise ValueError("Focused Hard32 requires normal_fusion_profile=native")
    if device != "cuda:0" or dtype != "bfloat16" or attn_implementation != "sdpa":
        raise ValueError("Focused Hard32 requires CUDA 0, bfloat16, and SDPA")

    requested_delta_mem_root = _lexical_absolute(delta_mem_root)
    if requested_delta_mem_root != _lexical_absolute(PROJECT_ROOT):
        raise ValueError("Focused Hard32 requires this Delta-Mem checkout")

    protected_dataset = _lexical_absolute(dataset_file)
    protected_selection = _lexical_absolute(selection_file)
    protected_base_model = _lexical_absolute(base_model)
    if protected_dataset != _lexical_absolute(HISTORICAL_V6_OFFICIAL_VAL):
        raise ValueError("Focused Hard32 requires the exact frozen official val path")
    if protected_selection != _lexical_absolute(HISTORICAL_V6_HARD32_SELECTION):
        raise ValueError("Focused Hard32 requires the exact frozen Hard32 selection path")
    if protected_base_model != _lexical_absolute(HISTORICAL_V6_BASE_MODEL):
        raise ValueError("Focused Hard32 requires the exact frozen base-model path")
    pair_list_sha256 = sha256_text(
        json.dumps(HARD32_FROZEN_DONOR_PAIRS, separators=(",", ":"))
    )
    if pair_list_sha256 != HARD32_FROZEN_DONOR_PAIRS_SHA256:
        raise RuntimeError("Focused Hard32 frozen donor pair-list hash differs")

    requested_output = _lexical_absolute(output_dir)
    _reject_symlink_components(
        requested_output,
        description="focused Hard32 output directory",
    )
    if requested_output.exists():
        raise ValueError(
            "Focused Hard32 requires a fresh, nonexistent output directory and "
            "cannot resume"
        )

    authorization = validate_focused_train_selection_authorization(
        selection_receipt_path,
        memory_dir=memory_dir,
    )
    resolved_memory = Path(
        str(authorization["selected_checkpoint"]["path"])
    ).expanduser().resolve()
    checkpoint_binding = dict(authorization["selected_checkpoint"])
    global_step = checkpoint_binding.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise ValueError("Focused Hard32 selected checkpoint step differs")
    run_root = resolved_memory.parent.parent
    expected_memory = run_root / "trainer" / f"checkpoint-{global_step}"
    canonical_receipt = (
        run_root
        / "train32_endpoint_screen"
        / "train32_checkpoint_selection_receipt.json"
    )
    canonical_output = run_root / "hard32_once"
    receipt_binding = dict(authorization["receipt"])
    requested_receipt = _lexical_absolute(selection_receipt_path)
    bound_receipt = _lexical_absolute(str(receipt_binding.get("path", "")))
    if (
        resolved_memory != expected_memory
        or requested_receipt != canonical_receipt
        or bound_receipt != canonical_receipt
    ):
        raise ValueError(
            "Focused Hard32 requires the canonical checkpoint and Train32 "
            "selection receipt layout"
        )
    if requested_output != canonical_output:
        raise ValueError(
            "Focused Hard32 permits only the canonical hard32_once output directory"
        )

    resolved_layer_count = resolved_memory_layer_count(
        resolved_memory,
        expected_memory_layer_count,
    )
    architecture = memory_architecture_contract(resolved_memory)
    if (
        resolved_layer_count != 42
        or architecture.get("target_layers") != list(range(42))
        or architecture.get("delta_heads") != ["q", "o"]
        or architecture.get("rank") != 4
        or architecture.get("rwkv_ms_semantics_version") != 2
        or architecture.get("memory_backend") != "rwkv_ms"
    ):
        raise ValueError(
            "Focused Hard32 requires a 42-layer q,o rank-4 RWKV-MS checkpoint"
        )

    consumption = claim_scene_hard_failure_hard32_authorization(
        selection_receipt_path=canonical_receipt,
        payload={
            "authorization_kind": authorization["authorization_kind"],
            "authorization_scope": authorization["scope"],
            "run_root": str(run_root),
            "selection_receipt": receipt_binding,
            "selected_checkpoint": checkpoint_binding,
            "selection_evaluation_fingerprint": authorization[
                "evaluation_fingerprint"
            ],
            "train_gate": dict(authorization["gate"]),
            "hard32_output_dir": str(requested_output),
            "protected_dataset_path": str(protected_dataset),
            "protected_selection_path": str(protected_selection),
            "base_model_path": str(protected_base_model),
            "runtime": {
                "device": device,
                "dtype": dtype,
                "attn_implementation": attn_implementation,
            },
            "delta_mem_root": str(requested_delta_mem_root),
            "frozen_donor_pairs_sha256": pair_list_sha256,
            "conditions": list(conditions),
            "donor_rule": donor_rule,
            "max_new_tokens": max_new_tokens,
            "normal_fusion_profile": normal_fusion_profile,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
            "retry_authorized": False,
        },
    )
    claimed_authorization = dict(authorization)
    claimed_authorization["hard32_output_dir"] = str(requested_output)
    claimed_authorization["authorization_consumption"] = consumption
    return claimed_authorization


def scene_v15_consumption_marker_path(selection_receipt_path: Path) -> Path:
    selection_path = _lexical_absolute(selection_receipt_path)
    return selection_path.parent / SCENE_V15_HARD32_CONSUMPTION_MARKER_FILENAME


def claim_scene_v15_hard32_authorization(
    *,
    selection_receipt_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Irrevocably consume the selected checkpoint's single Hard32 authorization."""

    marker_path = scene_v15_consumption_marker_path(selection_receipt_path)
    marker_payload = dict(payload)
    marker_payload["schema"] = SCENE_V15_HARD32_CONSUMPTION_MARKER_SCHEMA
    marker_payload["claim_sha256"] = fingerprint_payload_sha256(marker_payload)
    serialized = json.dumps(marker_payload, sort_keys=True, separators=(",", ":"))
    encoded = serialized.encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker_path, flags, 0o400)
    except FileExistsError as exc:
        raise ValueError(
            "V15 Hard32 authorization was already consumed; exactly-once "
            f"evaluation forbids replay: {marker_path}"
        ) from exc
    try:
        offset = 0
        while offset < len(encoded):
            try:
                written = os.write(descriptor, encoded[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError("V15 Hard32 consumption marker write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(marker_path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {
        "path": str(marker_path),
        "bytes": len(encoded),
        "file_sha256": sha256_text(serialized),
        "claim_sha256": marker_payload["claim_sha256"],
        "schema": SCENE_V15_HARD32_CONSUMPTION_MARKER_SCHEMA,
    }


def validate_scene_v15_candidate_hard32_authorization(
    *,
    selection_receipt_path: Path,
    candidate_lock_path: Path,
    launch_receipt: Path,
    completion_receipt: Path,
    base_model: Path | str,
    memory_dir: Path,
    output_dir: Path,
    overwrite: bool,
    dataset_file: Path,
    selection_file: Path | None,
    device: str,
    dtype: str,
    attn_implementation: str,
    delta_mem_root: Path | str,
    conditions: list[str],
    donor_rule: str,
    max_new_tokens: int,
    normal_fusion_profile: str,
    expected_memory_layer_count: int | None,
    inline_row_indices: str | None,
    preflight_only: bool,
    broader_authorization_supplied: bool,
) -> dict[str, Any]:
    """Authorize one frozen Hard32 run from the complete Train32-only selection."""

    if preflight_only:
        raise ValueError(
            "V15 Hard32 forbids --preflight-only because protected access must "
            "consume its exactly-once authorization"
        )
    if overwrite:
        raise ValueError("V15 selected-checkpoint Hard32 evaluation forbids --overwrite")
    if broader_authorization_supplied:
        raise ValueError("V15 Hard32 forbids mixed or broader benchmark authorization")
    if inline_row_indices is not None or selection_file is None:
        raise ValueError(
            "V15 Hard32 requires only the frozen --row-indices-file; inline row "
            "indices are forbidden"
        )
    if conditions != list(SCENE_V15_HARD32_CONDITIONS):
        raise ValueError(
            f"{SCENE_V15_HARD32_CONTRACT} requires conditions in exact order: "
            + ",".join(SCENE_V15_HARD32_CONDITIONS)
        )
    if donor_rule != DONOR_RULE_LENGTH_MATCHED:
        raise ValueError(
            f"V15 Hard32 requires donor_rule={DONOR_RULE_LENGTH_MATCHED}"
        )
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(
            f"V15 Hard32 requires max_new_tokens={DEFAULT_MAX_NEW_TOKENS}"
        )
    if normal_fusion_profile != "native":
        raise ValueError("V15 Hard32 requires normal_fusion_profile=native")
    if device != "cuda:0" or dtype != "bfloat16" or attn_implementation != "sdpa":
        raise ValueError("V15 Hard32 requires CUDA 0, bfloat16, and SDPA")
    if _lexical_absolute(delta_mem_root) != _lexical_absolute(PROJECT_ROOT):
        raise ValueError("V15 Hard32 requires this Delta-Mem checkout")

    protected_dataset = _lexical_absolute(dataset_file)
    protected_selection = _lexical_absolute(selection_file)
    if protected_dataset != _lexical_absolute(HISTORICAL_V6_OFFICIAL_VAL):
        raise ValueError("V15 Hard32 requires the exact frozen official val path")
    if protected_selection != _lexical_absolute(HISTORICAL_V6_HARD32_SELECTION):
        raise ValueError("V15 Hard32 requires the exact frozen Hard32 selection path")
    pair_list_sha256 = sha256_text(
        json.dumps(HARD32_FROZEN_DONOR_PAIRS, separators=(",", ":"))
    )
    if pair_list_sha256 != HARD32_FROZEN_DONOR_PAIRS_SHA256:
        raise RuntimeError("V15 Hard32 frozen donor pair-list hash differs")

    from experiments.rethinking_rwkv_ms_gemma import (  # lazy circular import
        scene_memory_v15_launch_contract as launch,
    )
    from experiments.rethinking_rwkv_ms_gemma import (  # lazy circular import
        select_scene_memory_v15_checkpoint as selector,
    )

    selection_path = _lexical_absolute(selection_receipt_path)
    lock_path = _lexical_absolute(candidate_lock_path)
    selection_root = _lexical_absolute(selector.SELECTION_ROOT)
    if (
        selection_path.name != selector.SELECTION_RECEIPT_FILENAME
        or selection_path.parent.parent != selection_root
        or lock_path
        != selection_path.parent / selector.CANDIDATE_LOCK_FILENAME
    ):
        raise ValueError(
            "V15 Hard32 requires the canonical candidate lock beside its "
            "canonical Train32 selection receipt"
        )
    for path, description in (
        (selection_path, "V15 selection receipt"),
        (lock_path, "V15 candidate lock"),
        (launch_receipt.expanduser().absolute(), "V15 launch receipt"),
        (completion_receipt.expanduser().absolute(), "V15 completion receipt"),
    ):
        _reject_symlink_components(path, description=description)

    # This validation reads only Train32 audit files and checkpoint artifacts.
    selected_receipt = selector.validate_selection_receipt(selection_path)
    candidate_lock = selector.validate_candidate_lock(
        lock_path,
        selection_receipt=selection_path,
    )
    if (
        candidate_lock.get("schema") != SCENE_V15_CANDIDATE_LOCK_SCHEMA
        or candidate_lock.get("authorization_kind")
        != SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND
        or candidate_lock.get("selection_policy")
        != SCENE_V15_CANDIDATE_SELECTION_POLICY
    ):
        raise ValueError("V15 candidate-lock evaluator contract differs")

    expected_output = _lexical_absolute(candidate_lock["hard32_output_dir"])
    requested_output = _lexical_absolute(output_dir)
    if requested_output != expected_output:
        raise ValueError("V15 candidate lock permits only its exact Hard32 output directory")
    _reject_symlink_components(requested_output, description="V15 Hard32 output directory")
    if requested_output.exists():
        raise ValueError(
            "V15 Hard32 requires a fresh, nonexistent output directory and cannot resume"
        )

    selected = candidate_lock["selected_candidate"]
    selected_checkpoint = selected_receipt["selected_checkpoint"]
    selector_manifest = selected_receipt.get("selector_manifest")
    selector_base_model = selected_receipt.get("base_model")
    if (
        not isinstance(selector_manifest, Mapping)
        or not isinstance(selector_base_model, Mapping)
        or selected.get("selector_manifest") != selector_manifest
        or selected.get("selector_manifest_fingerprint")
        != selected_receipt.get("fingerprint")
        or selected.get("base_model") != selector_base_model
    ):
        raise ValueError("V15 selector-manifest or base-model candidate chain differs")
    resolved_memory = _lexical_absolute(memory_dir)
    expected_memory = _lexical_absolute(str(selected_checkpoint.get("path", "")))
    if resolved_memory != expected_memory or selected.get("checkpoint") != selected_checkpoint:
        raise ValueError("V15 candidate lock rejects every unselected checkpoint")

    selector.validate_artifact_binding(
        selected_receipt["launch_receipt"],
        description="V15 bound launch receipt",
        expected_path=launch_receipt,
    )
    selector.validate_artifact_binding(
        selected_receipt["completion_receipt"],
        description="V15 bound completion receipt",
        expected_path=completion_receipt,
    )
    selector.validate_artifact_binding(
        selector_manifest,
        description="V15 bound selector manifest",
        expected_path=selection_path.parent / selector.SELECTOR_MANIFEST_FILENAME,
    )

    data = launch.validate_data_contract()
    warm = launch.validate_warm_start_contract()
    base_identity = launch.validate_base_model_contract(Path(base_model))
    if dict(base_identity) != dict(selector_base_model):
        raise ValueError("V15 selected checkpoint base-model identity differs")
    checkpoint_contract = launch.validate_checkpoint_contract(
        resolved_memory,
        data=data,
        warm=warm,
        smoke=False,
    )
    if checkpoint_contract != selected_checkpoint:
        raise ValueError("V15 selected checkpoint artifact contract differs")
    resolved_layer_count = resolved_memory_layer_count(
        resolved_memory,
        expected_memory_layer_count,
    )
    architecture = memory_architecture_contract(resolved_memory)
    if (
        resolved_layer_count != 42
        or architecture.get("target_layers") != list(range(42))
        or architecture.get("delta_heads") != ["q", "o"]
        or architecture.get("rank") != 4
        or architecture.get("rwkv_ms_semantics_version") != 2
        or architecture.get("memory_backend") != "rwkv_ms"
    ):
        raise ValueError("V15 Hard32 requires a 42-layer q,o rank-4 RWKV-MS checkpoint")
    launch_validation = launch.validate_launch_receipt(
        launch_receipt,
        checkpoint=resolved_memory,
        base_model_identity=base_identity,
        smoke=False,
        data=data,
        warm_start=warm,
    )
    completion_validation = launch.validate_completion_receipt(
        completion_receipt,
        checkpoint=resolved_memory,
        checkpoint_contract=checkpoint_contract,
        launch=launch_validation,
        smoke=False,
        data=data,
        warm_start=warm,
    )

    expected_checkpoints = {
        Path(str(summary["checkpoint"]["path"])).name: summary["checkpoint"]
        for summary in selected_receipt["candidates"]
    }
    completion_payload = completion_validation.get("payload", {})
    launch_payload = launch_validation.get("payload", {})
    if (
        completion_payload.get("checkpoints") != expected_checkpoints
        or launch_payload.get("checkpoints")
        != {
            name: str(contract["path"])
            for name, contract in expected_checkpoints.items()
        }
    ):
        raise ValueError(
            "V15 completion receipt does not bind all four selected post-update audits"
        )

    selection_binding = {
        "path": str(selection_path),
        "bytes": selection_path.stat().st_size,
        "file_sha256": sha256_file(selection_path),
        "payload_sha256": selected_receipt["validated_receipt_sha256"],
    }
    lock_binding = {
        "path": str(lock_path),
        "bytes": lock_path.stat().st_size,
        "file_sha256": sha256_file(lock_path),
        "payload_sha256": candidate_lock["validated_lock_sha256"],
        "selection_policy": SCENE_V15_CANDIDATE_SELECTION_POLICY,
    }
    training_provenance = {
        "launch": {
            "path": launch_validation["path"],
            "file_sha256": launch_validation["file_sha256"],
            "receipt_sha256": launch_validation["receipt_sha256"],
        },
        "completion": {
            "path": completion_validation["path"],
            "file_sha256": completion_validation["file_sha256"],
            "receipt_sha256": completion_validation["receipt_sha256"],
        },
    }
    authorization = {
        "authorization_kind": SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "scope": SCENE_V15_HARD32_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "candidate_lock": lock_binding,
        "selection_receipt": selection_binding,
        "selection_fingerprint": selected_receipt["fingerprint"],
        "selector_manifest": dict(selector_manifest),
        "ranked_checkpoint_steps": list(
            selected_receipt["ranked_checkpoint_steps"]
        ),
        "selected_checkpoint_step": selected_receipt["selected_checkpoint_step"],
        "post_save_audit": dict(selected["post_save_audit"]),
        "base_model": base_identity["path"],
        "base_model_identity": dict(base_identity),
        "hard32_output_dir": str(requested_output),
        "checkpoint": dict(checkpoint_contract),
        "training_provenance": training_provenance,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    # This duplicates the protected contract check using compiled-in identities only.
    # The real selection and dataset remain unopened until the exclusive claim below.
    validate_scene_v6_matched_donor_contract(
        contract=SCENE_V15_HARD32_CONTRACT,
        row_indices=list(HARD32_ROW_INDICES),
        expected_hashes=dict(HARD32_ROW_HASHES),
        selection_dataset_contract={
            "path": str(_lexical_absolute(HISTORICAL_V6_OFFICIAL_VAL)),
            "split": "val",
            "sha256": OFFICIAL_SCENE_V4_VAL_SHA256,
        },
        conditions=conditions,
        donor_rule=donor_rule,
        max_new_tokens=max_new_tokens,
        normal_fusion_profile=normal_fusion_profile,
        expected_memory_layer_count=resolved_layer_count,
        memory_target_layers=architecture["target_layers"],
        memory_delta_heads=architecture["delta_heads"],
        memory_rank=architecture["rank"],
        rwkv_ms_semantics_version=architecture["rwkv_ms_semantics_version"],
        memory_backend=architecture["memory_backend"],
        selection_manifest_sha256=HARD32_SELECTION_SHA256,
        scene_v15_candidate_authorization=authorization,
    )
    consumption = claim_scene_v15_hard32_authorization(
        selection_receipt_path=selection_path,
        payload={
            "authorization_kind": SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
            "authorization_scope": SCENE_V15_HARD32_AUTHORIZATION_SCOPE,
            "selection_receipt": selection_binding,
            "candidate_lock": lock_binding,
            "selection_fingerprint": selected_receipt["fingerprint"],
            "selector_manifest": dict(selector_manifest),
            "checkpoint": dict(checkpoint_contract),
            "training_provenance": training_provenance,
            "base_model": dict(base_identity),
            "hard32_output_dir": str(requested_output),
            "protected_dataset_path": str(protected_dataset),
            "protected_selection_path": str(protected_selection),
            "conditions": list(conditions),
            "donor_rule": donor_rule,
            "frozen_donor_pairs_sha256": pair_list_sha256,
            "max_new_tokens": max_new_tokens,
            "normal_fusion_profile": normal_fusion_profile,
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
            "retry_authorized": False,
        },
    )
    hard32 = validate_historical_v6_hard32_artifacts(
        dataset_file=dataset_file,
        selection_file=selection_file,
    )
    authorization["authorization_consumption"] = consumption
    authorization["frozen_hard32"] = hard32
    return authorization


def validate_hard32_pass_receipt(
    receipt_path: Path,
    *,
    memory_dir: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Hard32 receipt must contain an object")
    unsigned = dict(receipt)
    recorded_sha = unsigned.pop("receipt_sha256", None)
    if recorded_sha != fingerprint_payload_sha256(unsigned):
        raise ValueError("Hard32 receipt checksum differs")
    if receipt.get("schema") != HARD32_RECEIPT_SCHEMA:
        raise ValueError("Hard32 receipt schema differs")
    checkpoint = receipt.get("checkpoint")
    candidate_lineage = (
        checkpoint.get("candidate_lineage") if isinstance(checkpoint, dict) else None
    )
    if (
        receipt.get("authorization_scope")
        in {
            "fixed_hard32_only_no_full170",
            SCENE_V8_HARD32_AUTHORIZATION_SCOPE,
        }
        or (
            isinstance(candidate_lineage, dict)
            and candidate_lineage.get("lineage_kind")
            in {
                SCENE_V7_TRAIN32_AUTHORIZATION_KIND,
                SCENE_V8_TRAIN32_AUTHORIZATION_KIND,
            }
        )
    ):
        raise ValueError(
            "A Train32-authorized fixed-Hard32 receipt does not authorize full170"
        )
    if (
        receipt.get("objective_interpretation")
        != SCENE_V6_IDENTITY_OBJECTIVE_INTERPRETATION
    ):
        raise ValueError("Hard32 receipt objective interpretation differs")
    if receipt.get("status") != "pass":
        raise ValueError("Full170 requires a passed hard32 receipt")
    contract = receipt.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("name") != "scene_v6_identity_hard32"
        or contract.get("rows") != 32
        or contract.get("conditions") != list(CONDITIONS)
    ):
        raise ValueError("Hard32 receipt contract differs")
    resolved_memory_dir = memory_dir.expanduser().resolve()
    current_lineage = scene_v6_training_lineage(resolved_memory_dir)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("memory_dir") != str(resolved_memory_dir)
        or checkpoint.get("adapter_sha256")
        != sha256_file(resolved_memory_dir / "delta_mem_adapter.pt")
        or checkpoint.get("config_sha256")
        != sha256_file(resolved_memory_dir / "delta_mem_config.json")
        or checkpoint.get("candidate_lineage") != current_lineage
    ):
        raise ValueError("Hard32 receipt checkpoint binding differs")
    dataset = receipt.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("sha256") != OFFICIAL_SCENE_V4_VAL_SHA256
        or dataset.get("split") != "val"
    ):
        raise ValueError("Hard32 receipt dataset binding differs")
    selection = receipt.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("sha256") != HARD32_SELECTION_SHA256
        or selection.get("holdout_sha256") != HARD32_HOLDOUT_SHA256
        or selection.get("pair_manifest_sha256") != HARD32_PAIR_MANIFEST_SHA256
        or selection.get("rows")
        != [
            {"source_index": index, "row_sha256": HARD32_ROW_HASHES[index]}
            for index in HARD32_ROW_INDICES
        ]
    ):
        raise ValueError("Hard32 receipt selection binding differs")
    _validate_bound_file(selection, description="selection")
    donor_mapping = receipt.get("donor_mapping")
    if (
        not isinstance(donor_mapping, dict)
        or donor_mapping.get("rule") != DONOR_RULE_LENGTH_MATCHED
        or donor_mapping.get("protected_pairing")
        != "global_minimum_bottleneck_then_total_v1"
        or donor_mapping.get("frozen_pairs")
        != [list(pair) for pair in HARD32_FROZEN_DONOR_PAIRS]
        or donor_mapping.get("frozen_pairs_sha256")
        != HARD32_FROZEN_DONOR_PAIRS_SHA256
        or donor_mapping.get("maximum_write_token_difference")
        != HARD32_FROZEN_DONOR_MAX_WRITE_TOKEN_DIFFERENCE
        or donor_mapping.get("total_write_token_difference")
        != HARD32_FROZEN_DONOR_TOTAL_WRITE_TOKEN_DIFFERENCE
        or donor_mapping.get("cardinality_stratum_rows")
        != HARD32_FROZEN_DONOR_STRATUM_ROWS
        or donor_mapping.get("sha256")
        != HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256
        or sha256_text(
            json.dumps(
                donor_mapping.get("rows"),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        != HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256
    ):
        raise ValueError("Hard32 receipt donor mapping differs")
    code = receipt.get("code")
    if (
        not isinstance(code, dict)
        or code.get("evaluator_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("Hard32 receipt evaluator code binding differs")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Hard32 receipt output bindings are missing")
    _validate_bound_file(outputs.get("manifest"), description="manifest")
    _validate_bound_file(outputs.get("summary"), description="summary")
    condition_outputs = outputs.get("conditions")
    if not isinstance(condition_outputs, dict) or set(condition_outputs) != set(CONDITIONS):
        raise ValueError("Hard32 receipt condition output bindings differ")
    for condition in CONDITIONS:
        _validate_bound_file(
            condition_outputs[condition],
            description=f"{condition} output",
        )
    gate = receipt.get("gate")
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "pass"
        or gate.get("all_gates_passed") is not True
        or gate.get("full170_authorized_for_bound_checkpoint") is not True
    ):
        raise ValueError("Hard32 receipt gate differs")
    return {
        "path": str(receipt_path),
        "file_sha256": sha256_file(receipt_path),
        "payload_sha256": recorded_sha,
        "evaluation_fingerprint": receipt["evaluation_fingerprint"],
        "checkpoint_adapter_sha256": checkpoint["adapter_sha256"],
        "checkpoint_config_sha256": checkpoint["config_sha256"],
    }


def validate_existing_manifest(
    manifest: Any,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Existing output manifest must be an object")
    recorded_fingerprint = manifest.get("fingerprint")
    if (
        not isinstance(recorded_fingerprint, str)
        or SHA256_RE.fullmatch(recorded_fingerprint) is None
    ):
        raise ValueError("Existing output manifest has an invalid fingerprint")
    payload_fingerprint = fingerprint_payload_sha256(
        manifest.get("fingerprint_payload")
    )
    if payload_fingerprint != recorded_fingerprint:
        raise ValueError(
            "Existing output manifest fingerprint_payload does not hash to its fingerprint"
        )
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError("Existing output manifest fingerprint differs from this run")
    return manifest


def validate_resume_record_contract(
    record: dict[str, Any],
    *,
    condition: str,
    condition_protocol: dict[str, Any],
    selected_by_index: dict[int, dict[str, Any]],
    donor_by_index: dict[int, dict[str, Any]] | None,
    shuffled_by_index: dict[int, dict[str, Any]] | None = None,
    fingerprint: str,
    split: str = "val",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    require_semantic_nll: bool = False,
    candidate_lineage_record_binding: Mapping[str, Any] | None = None,
) -> int:
    source_index = record.get("source_index")
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ValueError("Resume record source_index must be an integer")
    sample = selected_by_index.get(source_index)
    if sample is None:
        raise ValueError(f"Resume output contains unselected source index {source_index}")
    donor_by_index = donor_by_index or {}
    pair_donor = donor_by_index.get(source_index)
    if condition == "state_only_donor" and pair_donor is None:
        raise ValueError(
            f"Resume donor mapping is missing source index {source_index}"
        )
    generation_donor = pair_donor if condition == "state_only_donor" else None
    shuffled_by_index = shuffled_by_index or {}
    generation_shuffled = shuffled_by_index.get(source_index)
    if condition == "state_only_shuffled" and generation_shuffled is None:
        raise ValueError(
            f"Resume shuffled mapping is missing source index {source_index}"
        )

    expected_literals = {
        "status": "ok",
        "condition": condition,
        "condition_protocol": condition_protocol,
        "task": TASK_NAME,
        "task_kind": "scene",
        "split": split,
        "key": f"{TASK_NAME}:{source_index}",
        "line_index": source_index,
        "selection_ordinal": list(selected_by_index).index(source_index),
        "row_sha256": sample["row_sha256"],
        "fingerprint": fingerprint,
        "gold": sample["gold"],
        "write_token_count": sample.get("write_token_count"),
        "donor_source_index": (
            None if generation_donor is None else int(generation_donor["source_index"])
        ),
        "donor_row_sha256": (
            None if generation_donor is None else generation_donor["row_sha256"]
        ),
    }
    for field, expected in expected_literals.items():
        if record.get(field) != expected:
            raise ValueError(
                f"Resume record {field} differs at source index {source_index}"
            )
    if condition == "state_only_shuffled" and generation_shuffled is not None:
        for field, expected in {
            "shuffled_source_index": int(generation_shuffled["source_index"]),
            "shuffled_row_sha256": generation_shuffled["row_sha256"],
        }.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"Resume record {field} differs at source index {source_index}"
                )
    if (
        candidate_lineage_record_binding is not None
        and record.get("candidate_lineage")
        != dict(candidate_lineage_record_binding)
    ):
        raise ValueError(
            "Resume record candidate lineage differs at source index "
            f"{source_index}"
        )

    if "parsed_json" not in record:
        raise ValueError(f"Resume record parsed_json is missing at source index {source_index}")
    raw_generation = record.get("raw_generation")
    if not isinstance(raw_generation, str):
        raise ValueError(
            f"Resume record raw_generation is invalid at source index {source_index}"
        )
    parsed_json = record["parsed_json"]
    if extract_json(raw_generation) != parsed_json:
        raise ValueError(
            "Resume record raw_generation does not reproduce parsed_json at "
            f"source index {source_index}"
        )

    expected_strict = score_prediction("scene", parsed_json, sample["gold"])
    if record.get("score_strict") != expected_strict:
        raise ValueError(
            f"Resume record score_strict differs at source index {source_index}"
        )
    expected_recovered = recovered_scene_score(parsed_json, sample["gold"])
    if record.get("score_recovered") != expected_recovered:
        raise ValueError(
            f"Resume record score_recovered differs at source index {source_index}"
        )
    input_tokens = record.get("input_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens <= 0
    ):
        raise ValueError(
            f"Resume record input_tokens is invalid at source index {source_index}"
        )
    output_tokens = record.get("output_tokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or not 0 <= output_tokens <= max_new_tokens
    ):
        raise ValueError(
            f"Resume record output_tokens is invalid at source index {source_index}"
        )
    hit_max_new_tokens = record.get("hit_max_new_tokens")
    if (
        not isinstance(hit_max_new_tokens, bool)
        or hit_max_new_tokens != (output_tokens >= max_new_tokens)
    ):
        raise ValueError(
            "Resume record hit_max_new_tokens is inconsistent at source index "
            f"{source_index}"
        )
    elapsed_seconds = record.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0.0
    ):
        raise ValueError(
            f"Resume record elapsed_seconds is invalid at source index {source_index}"
        )
    input_rendered_sha256 = record.get("input_rendered_sha256")
    if (
        not isinstance(input_rendered_sha256, str)
        or SHA256_RE.fullmatch(input_rendered_sha256) is None
    ):
        raise ValueError(
            "Resume record input_rendered_sha256 is invalid at source index "
            f"{source_index}"
        )
    peak_cuda_memory_bytes = record.get("peak_cuda_memory_bytes")
    if peak_cuda_memory_bytes is not None and (
        isinstance(peak_cuda_memory_bytes, bool)
        or not isinstance(peak_cuda_memory_bytes, int)
        or peak_cuda_memory_bytes < 0
    ):
        raise ValueError(
            "Resume record peak_cuda_memory_bytes is invalid at source index "
            f"{source_index}"
        )
    if not isinstance(record.get("memory_trace"), list):
        raise ValueError(
            f"Resume record memory_trace is invalid at source index {source_index}"
        )
    if not isinstance(record.get("online_state_after_generation"), dict):
        raise ValueError(
            "Resume record online_state_after_generation is invalid at source index "
            f"{source_index}"
        )
    prime = record.get("prime")
    condition_requires_prime = condition in {
        "state_only",
        "state_only_donor",
        "state_only_shuffled",
        "state_only_no_write",
    }
    if condition_requires_prime:
        if not isinstance(prime, dict):
            raise ValueError(
                f"Resume record prime is invalid at source index {source_index}"
            )
        prime_tokens = prime.get("tokens")
        if (
            isinstance(prime_tokens, bool)
            or not isinstance(prime_tokens, int)
            or prime_tokens <= 0
            or not isinstance(prime.get("rendered_sha256"), str)
            or SHA256_RE.fullmatch(prime["rendered_sha256"]) is None
            or prime.get("kv_cache_retained") is not False
            or not isinstance(prime.get("online_state"), dict)
        ):
            raise ValueError(
                f"Resume record prime is invalid at source index {source_index}"
            )
    elif prime is not None:
        raise ValueError(
            f"Resume record prime differs at source index {source_index}"
        )
    semantic_nll = record.get("semantic_decision_nll")
    semantic_condition = condition in {
        "state_only",
        "state_only_donor",
        "state_only_no_write",
    }
    if require_semantic_nll and semantic_condition:
        if not isinstance(semantic_nll, dict):
            raise ValueError(
                f"Resume record semantic_decision_nll is missing at source index {source_index}"
            )
        all_semantic = semantic_nll.get("all_semantic")
        pair_target = semantic_nll.get("pair_target")
        if not isinstance(all_semantic, dict) or not isinstance(pair_target, dict):
            raise ValueError(
                f"Resume record semantic NLL branches are missing at source index {source_index}"
            )

        def validate_nll_branch(
            branch: dict[str, Any],
            *,
            expected_mask_mode: str,
            expected_normalization: str,
            exact_token_count: int | None,
        ) -> tuple[list[int], list[int]]:
            if (
                branch.get("mask_mode") != expected_mask_mode
                or branch.get("normalization") != expected_normalization
            ):
                raise ValueError(
                    "Resume record semantic_decision_nll protocol differs at "
                    f"source index {source_index}"
                )
            positions = branch.get("selected_target_positions")
            token_ids = branch.get("selected_target_token_ids")
            token_count = branch.get("token_count")
            if (
                not isinstance(positions, list)
                or not positions
                or not all(
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in positions
                )
                or not isinstance(token_ids, list)
                or len(token_ids) != len(positions)
                or not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in token_ids
                )
                or token_count != len(positions)
                or (
                    exact_token_count is not None
                    and token_count != exact_token_count
                )
            ):
                raise ValueError(
                    "Resume record semantic_decision_nll targets are invalid at "
                    f"source index {source_index}"
                )
            for field in ("nll_sum", "mean_nll"):
                value = branch.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"Resume record semantic_decision_nll {field} is invalid "
                        f"at source index {source_index}"
                    )
            rendered_hash = branch.get("read_rendered_sha256")
            if (
                not isinstance(rendered_hash, str)
                or SHA256_RE.fullmatch(rendered_hash) is None
            ):
                raise ValueError(
                    "Resume record semantic_decision_nll render hash is invalid at "
                    f"source index {source_index}"
                )
            return positions, token_ids

        all_positions, all_token_ids = validate_nll_branch(
            all_semantic,
            expected_mask_mode=SEMANTIC_DECISION_MASK_MODE,
            expected_normalization=SEMANTIC_DECISION_NLL_NORMALIZATION,
            exact_token_count=None,
        )
        pair_positions, pair_token_ids = validate_nll_branch(
            pair_target,
            expected_mask_mode=PAIR_TARGET_DECISION_MASK_MODE,
            expected_normalization=PAIR_TARGET_DECISION_NLL_NORMALIZATION,
            exact_token_count=1,
        )
        if pair_donor is None:
            raise ValueError(
                f"Resume semantic pair donor is missing at source index {source_index}"
            )
        if (
            pair_target.get("target_mode") != PAIR_TARGET_DECISION_MASK_MODE
            or pair_target.get("donor_source_index")
            != int(pair_donor["source_index"])
            or pair_target.get("donor_row_sha256") != pair_donor["row_sha256"]
            or not isinstance(
                pair_target.get("first_differing_semantic_ordinal"), int
            )
            or isinstance(
                pair_target.get("first_differing_semantic_ordinal"), bool
            )
            or pair_target["first_differing_semantic_ordinal"] < 0
            or not isinstance(pair_target.get("donor_target_token_ids"), list)
            or len(pair_target["donor_target_token_ids"]) != 1
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in pair_target["donor_target_token_ids"]
            )
            or not isinstance(pair_target.get("causal_prefix_sha256"), str)
            or SHA256_RE.fullmatch(pair_target["causal_prefix_sha256"]) is None
        ):
            raise ValueError(
                f"Resume semantic pair identity differs at source index {source_index}"
            )
        all_token_by_position = dict(zip(all_positions, all_token_ids, strict=True))
        if (
            pair_positions[0] not in all_token_by_position
            or all_token_by_position[pair_positions[0]] != pair_token_ids[0]
            or pair_target["read_rendered_sha256"]
            != all_semantic["read_rendered_sha256"]
        ):
            raise ValueError(
                f"Resume semantic pair target differs at source index {source_index}"
            )
    elif require_semantic_nll and semantic_nll is not None:
        raise ValueError(
            f"Resume record semantic_decision_nll differs at source index {source_index}"
        )
    return source_index


def validate_resume_records(
    records: list[dict[str, Any]],
    *,
    condition: str,
    condition_protocol: dict[str, Any],
    selected_by_index: dict[int, dict[str, Any]],
    donor_by_index: dict[int, dict[str, Any]] | None = None,
    shuffled_by_index: dict[int, dict[str, Any]] | None = None,
    fingerprint: str,
    split: str = "val",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    require_semantic_nll: bool = False,
    candidate_lineage_record_binding: Mapping[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    validated: dict[int, dict[str, Any]] = {}
    for record in records:
        source_index = validate_resume_record_contract(
            record,
            condition=condition,
            condition_protocol=condition_protocol,
            selected_by_index=selected_by_index,
            donor_by_index=donor_by_index,
            shuffled_by_index=shuffled_by_index,
            fingerprint=fingerprint,
            split=split,
            max_new_tokens=max_new_tokens,
            require_semantic_nll=require_semantic_nll,
            candidate_lineage_record_binding=candidate_lineage_record_binding,
        )
        if source_index in validated:
            raise ValueError(f"Duplicate resume record for source index {source_index}")
        validated[source_index] = record
    return validated


def read_resume_records(path: Path) -> list[dict[str, Any]]:
    """Read append-only output, removing only a malformed final partial record."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("rb+") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        while True:
            record_offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                decoded = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if handle.tell() != file_size or raw_line.endswith(b"\n"):
                    raise
                handle.seek(record_offset)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            if not isinstance(decoded, dict):
                raise ValueError(f"Resume record at byte {record_offset} is not an object")
            records.append(decoded)
            if handle.tell() == file_size and not raw_line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return records


def clear_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_adapter_model(args: argparse.Namespace, expected_layer_count: int):
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=args.memory_dir,
        memory_repo=None,
    )
    runtime = apply_normal_fusion_profile(
        model,
        profile_name=args.normal_fusion_profile,
        expected_layer_count=expected_layer_count,
    )
    return model, tokenizer, runtime


def load_base_model(args: argparse.Namespace):
    return load_model_and_tokenizer(
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=None,
        memory_repo=None,
    )


def scene_state_code_fingerprint(delta_mem_root: Path) -> dict[str, str]:
    return {
        "evaluator_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(SCRIPT_DIR / "common.py"),
        "novel_agent_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_novel_agent_eval.py"
        ),
        "scene_recovery_sha256": sha256_file(
            SCRIPT_DIR / "analyze_novel_agent_eval.py"
        ),
        "chat_templates_sha256": sha256_file(
            delta_mem_root / "deltamem" / "chat_templates.py"
        ),
        "delta_api_sha256": sha256_file(
            delta_mem_root / "deltamem" / "core" / "delta.py"
        ),
        "delta_impl_sha256": sha256_file(
            delta_mem_root / "deltamem" / "core" / "delta_impl.py"
        ),
        "hrm_rwkv7_sha256": sha256_file(
            delta_mem_root / "deltamem" / "core" / "hrm_rwkv7.py"
        ),
        "backbone_compat_sha256": sha256_file(
            delta_mem_root / "deltamem" / "core" / "backbone_compat.py"
        ),
        "write_segmentation_sha256": sha256_file(
            delta_mem_root / "deltamem" / "core" / "write_segmentation.py"
        ),
        "runtime_session_sha256": sha256_file(
            delta_mem_root / "deltamem" / "runtime" / "session.py"
        ),
        "model_loading_sha256": sha256_file(
            delta_mem_root / "deltamem" / "model_loading.py"
        ),
    }


def historical_base_model_binding(base_model: Path) -> dict[str, Any]:
    resolved = _historical_exact_path(
        base_model,
        expected=HISTORICAL_V6_BASE_MODEL,
        description="base model",
        directory=True,
    )
    return {
        "path": str(resolved),
        "weights": base_model_weight_identity(resolved),
        "prompt_artifacts": base_model_prompt_identity(resolved),
    }


def main() -> None:
    args = parse_args()
    if (
        args.preflight_only
        and args.evaluation_contract == SCENE_V15_HARD32_CONTRACT
    ):
        raise ValueError(
            "V15 Hard32 forbids --preflight-only because its protected access "
            "authorization is exactly once"
        )
    if args.preflight_only and args.evaluation_contract not in {
        HISTORICAL_V6_HARD32_CONTRACT,
        SCENE_V14_HARD32_CONTRACT,
    }:
        raise ValueError(
            "--preflight-only is restricted to the historical V6 and locked "
            "V14 Hard32 contracts"
        )
    conditions = selected_conditions(args.conditions)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    historical_preflight = validate_historical_v6_run_preflight(
        args,
        conditions=conditions,
    )
    args.memory_dir = args.memory_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    v8_hard32_requested = args.evaluation_contract == SCENE_V8_HARD32_CONTRACT
    v14_hard32_requested = args.evaluation_contract == SCENE_V14_HARD32_CONTRACT
    v15_hard32_requested = args.evaluation_contract == SCENE_V15_HARD32_CONTRACT
    focused_train_requested = (
        args.evaluation_contract == SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT
    )
    focused_hard32_requested = (
        args.evaluation_contract == SCENE_HARD_FAILURE_HARD32_CONTRACT
    )
    if (
        args.row_indices_file is not None
        and not v8_hard32_requested
        and not v14_hard32_requested
        and not v15_hard32_requested
        and not focused_hard32_requested
    ):
        args.row_indices_file = args.row_indices_file.expanduser().resolve()
    if args.hard32_receipt is not None:
        args.hard32_receipt = args.hard32_receipt.expanduser().resolve()
    if args.scene_v7_train32_receipt is not None:
        args.scene_v7_train32_receipt = args.scene_v7_train32_receipt.expanduser()
    if args.scene_v8_train32_receipt is not None:
        args.scene_v8_train32_receipt = args.scene_v8_train32_receipt.expanduser()
    if args.scene_v14_value14_receipt is not None:
        args.scene_v14_value14_receipt = args.scene_v14_value14_receipt.expanduser()
    if args.scene_v14_candidate_lock is not None:
        args.scene_v14_candidate_lock = args.scene_v14_candidate_lock.expanduser()
    if args.scene_v14_launch_receipt is not None:
        args.scene_v14_launch_receipt = args.scene_v14_launch_receipt.expanduser()
    if args.scene_v14_completion_receipt is not None:
        args.scene_v14_completion_receipt = (
            args.scene_v14_completion_receipt.expanduser()
        )
    if args.scene_v15_selection_receipt is not None:
        args.scene_v15_selection_receipt = args.scene_v15_selection_receipt.expanduser()
    if args.scene_v15_candidate_lock is not None:
        args.scene_v15_candidate_lock = args.scene_v15_candidate_lock.expanduser()
    if args.scene_v15_launch_receipt is not None:
        args.scene_v15_launch_receipt = args.scene_v15_launch_receipt.expanduser()
    if args.scene_v15_completion_receipt is not None:
        args.scene_v15_completion_receipt = (
            args.scene_v15_completion_receipt.expanduser()
        )
    if args.focused_source_manifest is not None:
        args.focused_source_manifest = args.focused_source_manifest.expanduser()
    if args.focused_train_selection_receipt is not None:
        args.focused_train_selection_receipt = (
            args.focused_train_selection_receipt.expanduser()
        )
    validate_scene_focused_artifact_scope(
        evaluation_contract=args.evaluation_contract,
        source_manifest=args.focused_source_manifest,
        train_selection_receipt=args.focused_train_selection_receipt,
    )
    validate_scene_v7_train32_receipt_scope(
        evaluation_contract=args.evaluation_contract,
        receipt_path=args.scene_v7_train32_receipt,
    )
    validate_scene_v8_train32_receipt_scope(
        evaluation_contract=args.evaluation_contract,
        receipt_path=args.scene_v8_train32_receipt,
    )
    validate_scene_v14_receipt_scope(
        evaluation_contract=args.evaluation_contract,
        gate_receipt=args.scene_v14_value14_receipt,
        candidate_lock=args.scene_v14_candidate_lock,
        launch_receipt=args.scene_v14_launch_receipt,
        completion_receipt=args.scene_v14_completion_receipt,
    )
    validate_scene_v15_receipt_scope(
        evaluation_contract=args.evaluation_contract,
        selection_receipt=args.scene_v15_selection_receipt,
        candidate_lock=args.scene_v15_candidate_lock,
        launch_receipt=args.scene_v15_launch_receipt,
        completion_receipt=args.scene_v15_completion_receipt,
    )
    if (
        args.hard32_receipt is not None
        and args.evaluation_contract != "scene_v6_matched_donor_validation"
    ):
        raise ValueError(
            "--hard32-receipt is accepted only by scene_v6_matched_donor_validation"
        )
    scene_v8_train32_authorization = None
    if v8_hard32_requested:
        scene_v8_train32_authorization = (
            validate_scene_v8_train32_hard32_authorization(
                args.scene_v8_train32_receipt,
                memory_dir=args.memory_dir,
                dataset_file=args.dataset_file,
                selection_file=args.row_indices_file,
            )
        )
        if args.row_indices_file is not None:
            args.row_indices_file = args.row_indices_file.expanduser().resolve()
    scene_v14_candidate_authorization = None
    if v14_hard32_requested:
        scene_v14_candidate_authorization = (
            validate_scene_v14_candidate_hard32_authorization(
                args.scene_v14_candidate_lock,
                args.scene_v14_value14_receipt,
                launch_receipt=args.scene_v14_launch_receipt,
                completion_receipt=args.scene_v14_completion_receipt,
                base_model=args.base_model,
                memory_dir=args.memory_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                dataset_file=args.dataset_file,
                selection_file=args.row_indices_file,
            )
        )
        if args.row_indices_file is not None:
            args.row_indices_file = args.row_indices_file.expanduser().resolve()
    scene_v15_candidate_authorization = None
    if v15_hard32_requested:
        scene_v15_candidate_authorization = (
            validate_scene_v15_candidate_hard32_authorization(
                selection_receipt_path=args.scene_v15_selection_receipt,
                candidate_lock_path=args.scene_v15_candidate_lock,
                launch_receipt=args.scene_v15_launch_receipt,
                completion_receipt=args.scene_v15_completion_receipt,
                base_model=args.base_model,
                memory_dir=args.memory_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                dataset_file=args.dataset_file,
                selection_file=args.row_indices_file,
                device=args.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
                delta_mem_root=args.delta_mem_root,
                conditions=conditions,
                donor_rule=args.donor_rule,
                max_new_tokens=args.max_new_tokens,
                normal_fusion_profile=args.normal_fusion_profile,
                expected_memory_layer_count=args.expected_memory_layer_count,
                inline_row_indices=args.row_indices,
                preflight_only=args.preflight_only,
                broader_authorization_supplied=any(
                    receipt is not None
                    for receipt in (
                        args.hard32_receipt,
                        args.scene_v7_train32_receipt,
                        args.scene_v8_train32_receipt,
                        args.scene_v14_value14_receipt,
                        args.scene_v14_candidate_lock,
                        args.scene_v14_launch_receipt,
                        args.scene_v14_completion_receipt,
                    )
                ),
            )
        )
        if args.row_indices_file is not None:
            args.row_indices_file = args.row_indices_file.expanduser().resolve()
    focused_train_selection_authorization = None
    if focused_hard32_requested:
        focused_train_selection_authorization = (
            validate_focused_hard32_exactly_once_authorization(
                selection_receipt_path=args.focused_train_selection_receipt,
                memory_dir=args.memory_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                dataset_file=args.dataset_file,
                selection_file=args.row_indices_file,
                base_model=args.base_model,
                device=args.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
                delta_mem_root=args.delta_mem_root,
                conditions=conditions,
                donor_rule=args.donor_rule,
                max_new_tokens=args.max_new_tokens,
                normal_fusion_profile=args.normal_fusion_profile,
                expected_memory_layer_count=args.expected_memory_layer_count,
                inline_row_indices=args.row_indices,
                preflight_only=args.preflight_only,
            )
        )
        if args.row_indices_file is not None:
            args.row_indices_file = args.row_indices_file.expanduser().resolve()
    dataset_file = resolve_scene_dataset_file(
        args.dataset_file,
        evaluation_contract=args.evaluation_contract,
    )
    focused_train_source = None
    if focused_train_requested:
        focused_train_source = validate_focused_train_source_manifest(
            args.focused_source_manifest,
            dataset_file=dataset_file,
        )
    row_indices, expected_hashes, selection_dataset_contract = read_selection(args)
    if focused_train_source is not None:
        expected_hashes = dict(focused_train_source["expected_row_hashes"])
    validate_selection_dataset_contract(dataset_file, selection_dataset_contract)
    expected_layer_count = resolved_memory_layer_count(
        args.memory_dir, args.expected_memory_layer_count
    )
    memory_architecture = memory_architecture_contract(args.memory_dir)
    scene_v7_train32_authorization = None
    if focused_train_selection_authorization is not None:
        candidate_lineage = {
            "lineage_kind": "scene_hard_failure_train_overfit_selection",
            "authorization": focused_train_selection_authorization,
        }
    elif scene_v15_candidate_authorization is not None:
        candidate_lineage = {
            "lineage_kind": SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
            "authorization": scene_v15_candidate_authorization,
        }
    elif scene_v14_candidate_authorization is not None:
        candidate_lineage = {
            "lineage_kind": SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND,
            "authorization": scene_v14_candidate_authorization,
        }
    elif scene_v8_train32_authorization is not None:
        candidate_lineage = {
            "lineage_kind": SCENE_V8_TRAIN32_AUTHORIZATION_KIND,
            "authorization": scene_v8_train32_authorization,
        }
    elif args.scene_v7_train32_receipt is not None:
        scene_v7_train32_authorization = (
            validate_scene_v7_train32_hard32_authorization(
                args.scene_v7_train32_receipt,
                memory_dir=args.memory_dir,
            )
        )
        candidate_lineage = {
            "lineage_kind": SCENE_V7_TRAIN32_AUTHORIZATION_KIND,
            "authorization": scene_v7_train32_authorization,
        }
    elif args.evaluation_contract == HISTORICAL_V6_HARD32_CONTRACT:
        if historical_preflight is None:
            raise AssertionError("Historical V6 preflight result is missing")
        candidate_lineage = historical_preflight["checkpoint"]
    else:
        candidate_lineage = (
            None
            if args.evaluation_contract == "generic"
            else scene_v6_training_lineage(args.memory_dir)
        )
    if (
        args.evaluation_contract == "scene_v6_identity_hard32"
        and scene_v7_train32_authorization is None
        and candidate_lineage.get("lineage_kind")
        != "identity_checkpoint_receipt"
    ):
        raise ValueError(
            "scene_v6_identity_hard32 requires a complete checkpoint_receipt.json"
        )
    hard32_receipt_authorization = None
    if args.evaluation_contract == "scene_v6_matched_donor_validation":
        if args.hard32_receipt is None:
            raise ValueError(
                "scene_v6_matched_donor_validation requires --hard32-receipt"
            )
        hard32_receipt_authorization = validate_hard32_pass_receipt(
            args.hard32_receipt,
            memory_dir=args.memory_dir,
        )
    elif args.hard32_receipt is not None:
        raise ValueError(
            "--hard32-receipt is accepted only by scene_v6_matched_donor_validation"
        )
    selection_manifest_sha256 = (
        None
        if args.row_indices_file is None
        else sha256_file(args.row_indices_file)
    )
    evaluation_contract = validate_scene_v6_matched_donor_contract(
        contract=args.evaluation_contract,
        row_indices=row_indices,
        expected_hashes=expected_hashes,
        selection_dataset_contract=selection_dataset_contract,
        conditions=conditions,
        donor_rule=args.donor_rule,
        max_new_tokens=args.max_new_tokens,
        normal_fusion_profile=args.normal_fusion_profile,
        expected_memory_layer_count=expected_layer_count,
        memory_target_layers=memory_architecture["target_layers"],
        memory_delta_heads=memory_architecture["delta_heads"],
        memory_rank=memory_architecture["rank"],
        rwkv_ms_semantics_version=memory_architecture[
            "rwkv_ms_semantics_version"
        ],
        memory_backend=memory_architecture["memory_backend"],
        selection_manifest_sha256=selection_manifest_sha256,
        hard32_receipt_authorization=hard32_receipt_authorization,
        scene_v8_train32_authorization=scene_v8_train32_authorization,
        scene_v14_candidate_authorization=scene_v14_candidate_authorization,
        scene_v15_candidate_authorization=scene_v15_candidate_authorization,
        focused_train_source=focused_train_source,
        focused_train_selection_authorization=(
            focused_train_selection_authorization
        ),
        memory_dir=args.memory_dir,
    )
    evaluation_split = str(evaluation_contract.get("split", "val"))
    if evaluation_split not in {"train", "val"}:
        raise ValueError("Scene-state evaluation split is invalid")
    if args.evaluation_contract == "generic" and len(row_indices) > MAX_SELECTED_ROWS:
        raise ValueError(
            f"Focused validation is capped at {MAX_SELECTED_ROWS} selected rows; "
            f"received {len(row_indices)}"
        )
    samples = load_selected_rows(
        dataset_file,
        row_indices,
        expected_hashes=expected_hashes,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "model_loaded": False,
                    "output_created": False,
                    "selected_rows": len(samples),
                    "evaluation_contract": evaluation_contract,
                    "artifact_preflight": historical_preflight,
                    "repository_revision": git_revision(PROJECT_ROOT),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    base_model_path = Path(args.base_model).expanduser().resolve()
    args.base_model = str(base_model_path)
    selected_by_index = {sample["source_index"]: sample for sample in samples}
    donor_by_index: dict[int, dict[str, Any]] = {}
    if "state_only_donor" in conditions:
        if focused_train_source is not None:
            from transformers import AutoTokenizer

            donor_tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=True,
            )
            annotate_write_token_counts(samples, donor_tokenizer)
            del donor_tokenizer
            donor_by_index = materialize_focused_train_donor_mapping(
                samples,
                focused_train_source["donor_by_source_index"],
            )
        elif args.donor_rule == DONOR_RULE_CYCLIC:
            donor_by_index = build_cyclic_donor_mapping(samples)
        else:
            from transformers import AutoTokenizer

            donor_tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=True,
            )
            annotate_write_token_counts(samples, donor_tokenizer)
            del donor_tokenizer
            if is_identity_hard32_contract(args.evaluation_contract):
                donor_by_index = build_frozen_hard32_donor_mapping(samples)
            else:
                donor_by_index = build_length_matched_label_distinct_donor_mapping(
                    samples
                )
    donor_mapping = (
        donor_mapping_fingerprint_rows(samples, donor_by_index)
        if donor_by_index
        else []
    )
    shuffled_by_index: dict[int, dict[str, Any]] = {}
    if "state_only_shuffled" in conditions:
        if not donor_by_index:
            raise ValueError("state_only_shuffled requires the locked donor mapping")
        shuffled_by_index = build_deterministic_shuffled_mapping(
            samples,
            donor_by_index,
        )
    shuffled_mapping = (
        shuffled_mapping_fingerprint_rows(samples, shuffled_by_index)
        if shuffled_by_index
        else []
    )
    condition_protocols = resolved_condition_protocols(
        conditions,
        donor_rule=args.donor_rule,
    )
    profile_fields = normal_fusion_fingerprint_fields(
        args.normal_fusion_profile, expected_layer_count
    )

    delta_mem_root = Path(args.delta_mem_root).expanduser().resolve()
    if not (delta_mem_root / "deltamem").is_dir():
        raise ValueError(f"Delta-Mem source root is invalid: {delta_mem_root}")
    if delta_mem_root != PROJECT_ROOT:
        raise ValueError(
            "--delta-mem-root must match the checkout containing this evaluator: "
            f"expected={PROJECT_ROOT} actual={delta_mem_root}"
        )
    args.delta_mem_root = str(delta_mem_root)
    code_fingerprint = scene_state_code_fingerprint(delta_mem_root)
    base_model_binding = {
        "path": str(base_model_path),
        "weights": base_model_weight_identity(base_model_path),
        "prompt_artifacts": base_model_prompt_identity(base_model_path),
    }
    candidate_lineage_record_binding = build_candidate_lineage_record_binding(
        candidate_lineage
    )
    selection_source = (
        {"kind": "inline_indices"}
        if args.row_indices is not None
        else {
            "kind": "indices_file",
            "path": str(args.row_indices_file.expanduser().resolve()),
            "sha256": sha256_file(args.row_indices_file),
            "expected_row_hashes_provided": bool(expected_hashes),
        }
    )
    fingerprint_payload = {
        "schema_version": 1,
        "task": TASK_NAME,
        "split": evaluation_split,
        "code": code_fingerprint,
        "runtime_packages": runtime_package_versions(),
        "delta_mem_root": str(delta_mem_root),
        "base_model": str(base_model_path),
        "base_model_weights": base_model_binding["weights"],
        "base_model_prompt_artifacts": base_model_binding["prompt_artifacts"],
        "memory_dir": str(args.memory_dir),
        "memory_config_sha256": sha256_file(args.memory_dir / "delta_mem_config.json"),
        "memory_adapter_sha256": sha256_file(args.memory_dir / "delta_mem_adapter.pt"),
        "dataset_file": str(dataset_file),
        "dataset_sha256": sha256_file(dataset_file),
        "selection_source": selection_source,
        "selection": [
            {"source_index": sample["source_index"], "row_sha256": sample["row_sha256"]}
            for sample in samples
        ],
        "conditions": conditions,
        "condition_protocols": condition_protocols,
        "donor_rule": args.donor_rule,
        "state_only_donor_mapping": donor_mapping,
        **(
            {"state_only_shuffled_mapping": shuffled_mapping}
            if "state_only_shuffled" in conditions
            else {}
        ),
        "evaluation_contract": evaluation_contract,
        "candidate_lineage": candidate_lineage,
        "candidate_lineage_record_binding": candidate_lineage_record_binding,
        "hard32_receipt_authorization": hard32_receipt_authorization,
        "scene_v7_train32_authorization": scene_v7_train32_authorization,
        "scene_v14_candidate_authorization": scene_v14_candidate_authorization,
        "scene_v15_candidate_authorization": scene_v15_candidate_authorization,
        **(
            {
                "focused_train_source": focused_train_source,
                "focused_train_selection_authorization": (
                    focused_train_selection_authorization
                ),
            }
            if focused_train_requested or focused_hard32_requested
            else {}
        ),
        "historical_v6_preflight": historical_preflight,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        **profile_fields,
    }
    fingerprint = fingerprint_payload_sha256(fingerprint_payload)
    repository_revision = git_revision(PROJECT_ROOT)
    manifest = {
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "code": {
            "rwkv_repo": repository_revision,
            **code_fingerprint,
        },
        "evaluation_kind": "focused state-isolating scene-boundary diagnostic",
        "evaluation_contract": evaluation_contract,
        "candidate_lineage": candidate_lineage,
        "candidate_lineage_record_binding": candidate_lineage_record_binding,
        "warning": (
            "This exact train-only slice is an overfit diagnostic and cannot authorize "
            "validation without a separate selected-checkpoint receipt."
            if evaluation_split == "train"
            else (
                "This selected validation slice is diagnostic, not an unbiased estimate "
                "of the complete benchmark, and its rows must not be used for training."
            )
        ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=args.evaluation_contract
        not in {
            HISTORICAL_V6_HARD32_CONTRACT,
            SCENE_V15_HARD32_CONTRACT,
            SCENE_HARD_FAILURE_HARD32_CONTRACT,
        },
    )
    output_paths = [
        *(args.output_dir / f"{condition}.jsonl" for condition in conditions),
        args.output_dir / "manifest.json",
        args.output_dir / "progress.json",
        args.output_dir / "summary.json",
        args.output_dir / "hard32_receipt.json",
        args.output_dir / "historical_v6_hard32_receipt.json",
    ]
    if args.overwrite:
        for path in output_paths:
            path.unlink(missing_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if args.evaluation_contract == SCENE_V15_HARD32_CONTRACT:
        if any(path.exists() for path in output_paths):
            raise RuntimeError(
                "V15 Hard32 is a one-shot evaluation and cannot resume existing outputs"
            )
        write_json_atomic(manifest_path, manifest)
    elif args.evaluation_contract == SCENE_HARD_FAILURE_HARD32_CONTRACT:
        if any(path.exists() for path in output_paths):
            raise RuntimeError(
                "Focused Hard32 is a one-shot evaluation and cannot resume existing outputs"
            )
        write_json_atomic(manifest_path, manifest)
    elif manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            manifest = validate_existing_manifest(
                existing,
                expected_fingerprint=fingerprint,
            )
        except ValueError as exc:
            raise ValueError(
                f"Cannot resume from {manifest_path}: {exc}; "
                "use --overwrite or a new directory"
            ) from exc
    else:
        write_json_atomic(manifest_path, manifest)

    completed_by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in conditions:
        path = args.output_dir / f"{condition}.jsonl"
        if args.evaluation_contract == SCENE_V15_HARD32_CONTRACT:
            if path.exists():
                raise RuntimeError(
                    "V15 Hard32 condition output appeared after fresh initialization; "
                    "resume is forbidden"
                )
            completed_by_condition[condition] = {}
        elif args.evaluation_contract == SCENE_HARD_FAILURE_HARD32_CONTRACT:
            if path.exists():
                raise RuntimeError(
                    "Focused Hard32 condition output appeared after fresh initialization; "
                    "resume is forbidden"
                )
            completed_by_condition[condition] = {}
        else:
            completed_by_condition[condition] = validate_resume_records(
                read_resume_records(path),
                condition=condition,
                condition_protocol=condition_protocols[condition],
                selected_by_index=selected_by_index,
                donor_by_index=donor_by_index,
                shuffled_by_index=shuffled_by_index,
                fingerprint=fingerprint,
                split=evaluation_split,
                max_new_tokens=args.max_new_tokens,
                require_semantic_nll=(
                    is_identity_hard32_contract(args.evaluation_contract)
                ),
                candidate_lineage_record_binding=candidate_lineage_record_binding,
            )

    runtime_profile: dict[str, Any] | None = manifest.get("runtime_fusion_profile")
    for adapter_group in (False, True):
        group_conditions = [
            condition
            for condition in conditions
            if bool(condition_protocols[condition]["adapter"]) == adapter_group
            and len(completed_by_condition[condition]) < len(samples)
        ]
        if not group_conditions:
            continue
        if adapter_group:
            model, tokenizer, loaded_runtime_profile = load_adapter_model(
                args, expected_layer_count
            )
            if runtime_profile is not None and runtime_profile != loaded_runtime_profile:
                raise RuntimeError(
                    "Runtime fusion profile differs from the existing output manifest"
                )
            runtime_profile = loaded_runtime_profile
            manifest["runtime_fusion_profile"] = runtime_profile
            write_json_atomic(manifest_path, manifest)
        else:
            model, tokenizer = load_base_model(args)
        try:
            for condition in group_conditions:
                output_path = args.output_dir / f"{condition}.jsonl"
                for sample in samples:
                    source_index = int(sample["source_index"])
                    if source_index in completed_by_condition[condition]:
                        continue
                    result = evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor_by_index.get(source_index),
                        shuffled_sample=shuffled_by_index.get(source_index),
                        condition=condition,
                        max_new_tokens=args.max_new_tokens,
                        device=args.device,
                        collect_semantic_nll=(
                            is_identity_hard32_contract(args.evaluation_contract)
                        ),
                    )
                    record = {
                        "status": "ok",
                        "fingerprint": fingerprint,
                        "completed_at": utc_now(),
                        "condition": condition,
                        "condition_protocol": condition_protocols[condition],
                        "task": TASK_NAME,
                        "task_kind": "scene",
                        "split": evaluation_split,
                        "key": f"{TASK_NAME}:{source_index}",
                        "line_index": source_index,
                        "source_index": source_index,
                        "selection_ordinal": row_indices.index(source_index),
                        "row_sha256": sample["row_sha256"],
                        "write_token_count": sample.get("write_token_count"),
                        "gold": sample["gold"],
                        **(
                            {"candidate_lineage": candidate_lineage_record_binding}
                            if candidate_lineage_record_binding is not None
                            else {}
                        ),
                        **result,
                    }
                    append_record(output_path, record)
                    completed_by_condition[condition][source_index] = record
                    completed = sum(len(rows) for rows in completed_by_condition.values())
                    write_json_atomic(
                        args.output_dir / "progress.json",
                        {
                            "fingerprint": fingerprint,
                            "completed": completed,
                            "expected": len(samples) * len(conditions),
                            "last_key": f"{condition}:{source_index}",
                            "updated_at": utc_now(),
                        },
                    )
                    print(
                        progress_log_line(
                            condition=condition,
                            source_index=source_index,
                            record=record,
                        ),
                        flush=True,
                    )
        finally:
            del model
            del tokenizer
            clear_model_memory()

    ordered_records = {
        condition: [completed_by_condition[condition][index] for index in row_indices]
        for condition in conditions
    }
    summaries = {
        condition: summarize_records(records)
        for condition, records in ordered_records.items()
    }
    comparisons = build_comparisons(summaries)
    semantic_evidence = (
        build_semantic_decision_evidence(ordered_records)
        if is_identity_hard32_contract(args.evaluation_contract)
        else None
    )
    base_outcome_evidence = (
        build_base_outcome_evidence(ordered_records)
        if is_identity_hard32_contract(args.evaluation_contract)
        and args.evaluation_contract != SCENE_V15_HARD32_CONTRACT
        else None
    )
    scene_v15_hard32_gate = (
        build_scene_v15_hard32_gate(
            summaries=summaries,
            comparisons=comparisons,
            semantic_evidence=semantic_evidence,
            ordered_records=ordered_records,
            contract=evaluation_contract,
        )
        if args.evaluation_contract == SCENE_V15_HARD32_CONTRACT
        else {"status": "not_requested", "contract": evaluation_contract}
    )
    scene_v6_identity_hard32_gate = (
        build_scene_v6_identity_hard32_gate(
            summaries=summaries,
            comparisons=comparisons,
            semantic_evidence=semantic_evidence,
            contract=evaluation_contract,
        )
        if is_identity_hard32_contract(args.evaluation_contract)
        and args.evaluation_contract != SCENE_V15_HARD32_CONTRACT
        else {"status": "not_requested", "contract": evaluation_contract}
    )
    hard32_gate = (
        scene_v15_hard32_gate
        if args.evaluation_contract == SCENE_V15_HARD32_CONTRACT
        else scene_v6_identity_hard32_gate
    )
    historical_v6_evidence = (
        build_historical_v6_hard32_evidence(ordered_records)
        if args.evaluation_contract == HISTORICAL_V6_HARD32_CONTRACT
        else None
    )
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "complete": all(len(records) == len(samples) for records in ordered_records.values()),
        "task": TASK_NAME,
        "split": evaluation_split,
        "selected_source_indices": row_indices,
        "donor_rule": args.donor_rule,
        "evaluation_contract": evaluation_contract,
        "candidate_lineage": candidate_lineage,
        "candidate_lineage_record_binding": candidate_lineage_record_binding,
        "state_only_donor_mapping": donor_mapping,
        **(
            {"state_only_shuffled_mapping": shuffled_mapping}
            if "state_only_shuffled" in conditions
            else {}
        ),
        "selected_rows": len(samples),
        "conditions": summaries,
        "comparisons": comparisons,
        "semantic_decision_evidence": semantic_evidence,
        "base_outcome_evidence": base_outcome_evidence,
        "scene_v6_identity_hard32_gate": scene_v6_identity_hard32_gate,
        "scene_v15_hard32_gate": scene_v15_hard32_gate,
        "historical_v6_hard32_evidence": historical_v6_evidence,
        "scene_v6_matched_donor_gate": build_scene_v6_matched_donor_gate(
            comparisons,
            evaluation_contract,
        ),
        "runtime_fusion_profile": runtime_profile,
        "interpretation": {
            "normal_full_minus_base_full": "Structured-task delta with full prompt visible.",
            "normal_full_minus_no_write_full": (
                "Contribution of RWKV-MS state writes while the full prompt remains visible."
            ),
            "state_only_minus_state_only_no_write": (
                "Online-state causal delta after discarding the prime call's ordinary KV cache."
            ),
            "state_only_minus_state_only_donor": (
                "State-identity delta versus a donor selected by the recorded donor rule "
                "under the same system-only query protocol."
            ),
        },
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    write_json_atomic(
        args.output_dir / "progress.json",
        {
            "fingerprint": fingerprint,
            "completed": len(samples) * len(conditions),
            "expected": len(samples) * len(conditions),
            "complete": True,
            "updated_at": utc_now(),
        },
    )
    if is_identity_hard32_contract(args.evaluation_contract):
        if args.row_indices_file is None:
            raise AssertionError("Protected hard32 contract requires a selection file")
        receipt = build_hard32_receipt(
            output_dir=args.output_dir,
            fingerprint=fingerprint,
            contract=evaluation_contract,
            candidate_lineage=candidate_lineage,
            code_fingerprint=code_fingerprint,
            dataset_file=dataset_file,
            selection_file=args.row_indices_file,
            donor_mapping=donor_mapping,
            shuffled_mapping=shuffled_mapping,
            gate=hard32_gate,
            semantic_evidence=semantic_evidence,
            base_outcome_evidence=base_outcome_evidence,
            memory_dir=args.memory_dir,
            conditions=conditions,
        )
        write_json_atomic(args.output_dir / "hard32_receipt.json", receipt)
    elif args.evaluation_contract == HISTORICAL_V6_HARD32_CONTRACT:
        if args.row_indices_file is None or historical_v6_evidence is None:
            raise AssertionError("Historical V6 receipt inputs are incomplete")
        receipt_path = args.output_dir / "historical_v6_hard32_receipt.json"
        receipt = build_historical_v6_hard32_receipt(
            output_dir=args.output_dir,
            fingerprint=fingerprint,
            contract=evaluation_contract,
            candidate_lineage=candidate_lineage,
            code_fingerprint=code_fingerprint,
            repository_revision=repository_revision,
            base_model=base_model_path,
            base_model_binding=base_model_binding,
            dataset_file=dataset_file,
            selection_file=args.row_indices_file,
            memory_dir=args.memory_dir,
            conditions=conditions,
            evidence=historical_v6_evidence,
        )
        write_json_atomic(receipt_path, receipt)
        validate_historical_v6_hard32_receipt(
            receipt_path,
            fingerprint=fingerprint,
            memory_dir=args.memory_dir,
            base_model=base_model_path,
            base_model_binding=base_model_binding,
            dataset_file=dataset_file,
            selection_file=args.row_indices_file,
            code_fingerprint=code_fingerprint,
            repository_revision=repository_revision,
            evidence=historical_v6_evidence,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
