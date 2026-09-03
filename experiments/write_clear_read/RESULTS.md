# Write / clear / read results

Exact match (EM) and answer cross-entropy (CE) of the frozen base under four read conditions.
`correct` = state written from this row's passage; `donor` = state from another row's passage;
`zero` = no memory; `in_context` = passage in the prompt (upper bound). Passage is never in the
read context for the three memory conditions. Regenerate with `python report.py`.

| run | setup | step | eval set | correct EM | donor EM | zero EM | in-context EM | correct CE | donor CE | zero CE | mem mass |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4e4b_k8_kvbank_qread_ai | gemma-4-E4B-it | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.234 | 0.070 | 0.004 | 1.000 | 1.006 | 1.346 | 3.724 | 0.38 |
| gemma4e4b_k8_kvbank_qread_ai | gemma-4-E4B-it | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.324 | 0.055 | 0.016 | 1.000 | 0.911 | 1.509 | 3.671 | 0.38 |
| gemma4e4b_k8_kvbank_qread_ai | gemma-4-E4B-it | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.293 | 0.082 | 0.020 | 1.000 | 0.992 | 1.521 | 3.561 | 0.38 |
| gemma4e4b_k8_kvbank_qread_ai_sliding | gemma-4-E4B-it | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.414 | 0.062 | 0.004 | 1.000 | 0.803 | 2.094 | 3.724 | 0.37 |
| gemma4e4b_k8_kvbank_qread_ai_sliding | gemma-4-E4B-it | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.562 | 0.109 | 0.016 | 1.000 | 0.639 | 1.902 | 3.671 | 0.37 |
| gemma4e4b_k8_kvbank_qread_ai_sliding | gemma-4-E4B-it | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.508 | 0.059 | 0.020 | 1.000 | 0.743 | 2.136 | 3.561 | 0.37 |
| gemma4e4b_k8_single16 | gemma-4-E4B-it | delta | read=bank | routing=single | slots=16 | train=synthetic K=8 | 3000 | synthetic_k16 | 0.125 | 0.082 | 0.004 | 1.000 | 1.146 | 1.255 | 3.724 | 0.84 |
| gemma4e4b_k8_single16 | gemma-4-E4B-it | delta | read=bank | routing=single | slots=16 | train=synthetic K=8 | 3000 | synthetic_k4 | 0.195 | 0.062 | 0.016 | 1.000 | 1.034 | 1.461 | 3.671 | 0.84 |
| gemma4e4b_k8_single16 | gemma-4-E4B-it | delta | read=bank | routing=single | slots=16 | train=synthetic K=8 | 3000 | synthetic_k8 | 0.180 | 0.113 | 0.020 | 1.000 | 1.092 | 1.374 | 3.561 | 0.84 |
| gemma4e4b_k8_single_qread | gemma-4-E4B-it | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.109 | 0.070 | 0.004 | 1.000 | 1.127 | 1.133 | 3.724 | 0.27 |
| gemma4e4b_k8_single_qread | gemma-4-E4B-it | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.102 | 0.102 | 0.016 | 1.000 | 1.152 | 1.161 | 3.671 | 0.27 |
| gemma4e4b_k8_single_qread | gemma-4-E4B-it | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.074 | 0.051 | 0.020 | 1.000 | 1.176 | 1.186 | 3.561 | 0.27 |
| gemma4e4b_k8_single_qread_ai | gemma-4-E4B-it | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.176 | 0.074 | 0.004 | 1.000 | 1.094 | 1.231 | 3.724 | 0.21 |
| gemma4e4b_k8_single_qread_ai | gemma-4-E4B-it | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.203 | 0.117 | 0.016 | 1.000 | 1.041 | 1.302 | 3.671 | 0.20 |
| gemma4e4b_k8_single_qread_ai | gemma-4-E4B-it | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.230 | 0.066 | 0.020 | 1.000 | 1.051 | 1.359 | 3.561 | 0.21 |
| qwen1p7b_k2_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=2 | 3000 | synthetic_k2 | 0.793 | 0.039 | 0.039 | 1.000 | 0.222 | 2.090 | 4.706 | 0.13 |
| qwen1p7b_k2_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=2 | 3000 | synthetic_k4 | 0.715 | 0.074 | 0.043 | 1.000 | 0.390 | 2.309 | 4.985 | 0.13 |
| qwen1p7b_k2_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=2 | 3000 | synthetic_k8 | 0.418 | 0.074 | 0.023 | 1.000 | 0.895 | 2.194 | 4.665 | 0.14 |
| qwen1p7b_k8_chunk4x4 | Qwen3-1.7B | delta | read=bank | routing=chunk | slots=16 | train=synthetic K=8 | 3000 | synthetic_k16 | 0.168 | 0.086 | 0.035 | 1.000 | 0.998 | 1.114 | 4.725 | 0.17 |
| qwen1p7b_k8_chunk4x4 | Qwen3-1.7B | delta | read=bank | routing=chunk | slots=16 | train=synthetic K=8 | 3000 | synthetic_k4 | 0.371 | 0.102 | 0.031 | 1.000 | 0.878 | 1.159 | 4.660 | 0.17 |
| qwen1p7b_k8_chunk4x4 | Qwen3-1.7B | delta | read=bank | routing=chunk | slots=16 | train=synthetic K=8 | 3000 | synthetic_k8 | 0.297 | 0.082 | 0.031 | 1.000 | 0.981 | 1.197 | 4.801 | 0.17 |
| qwen1p7b_k8_cosine4x4 | Qwen3-1.7B | delta | read=bank | routing=cosine | slots=16 | train=synthetic K=8 | 3000 | synthetic_k16 | 0.379 | 0.059 | 0.035 | 1.000 | 0.819 | 1.328 | 4.725 | 0.17 |
| qwen1p7b_k8_cosine4x4 | Qwen3-1.7B | delta | read=bank | routing=cosine | slots=16 | train=synthetic K=8 | 3000 | synthetic_k4 | 0.586 | 0.082 | 0.031 | 1.000 | 0.630 | 1.364 | 4.660 | 0.16 |
| qwen1p7b_k8_cosine4x4 | Qwen3-1.7B | delta | read=bank | routing=cosine | slots=16 | train=synthetic K=8 | 3000 | synthetic_k8 | 0.449 | 0.086 | 0.031 | 1.000 | 0.775 | 1.324 | 4.801 | 0.16 |
| qwen1p7b_k8_kvbank_qread | Qwen3-1.7B | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.527 | 0.086 | 0.031 | 1.000 | 0.649 | 1.594 | 4.723 | 0.14 |
| qwen1p7b_k8_kvbank_qread | Qwen3-1.7B | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.695 | 0.094 | 0.031 | 1.000 | 0.451 | 1.646 | 4.659 | 0.14 |
| qwen1p7b_k8_kvbank_qread | Qwen3-1.7B | kvbank | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.602 | 0.074 | 0.031 | 1.000 | 0.579 | 1.617 | 4.803 | 0.14 |
| qwen1p7b_k8_single16 | Qwen3-1.7B | delta | read=bank | routing=single | slots=16 | train=synthetic K=8 | 3000 | synthetic_k16 | 0.289 | 0.070 | 0.031 | 1.000 | 0.875 | 1.248 | 4.723 | 0.17 |
| qwen1p7b_k8_single16 | Qwen3-1.7B | delta | read=bank | routing=single | slots=16 | train=synthetic K=8 | 3000 | synthetic_k4 | 0.543 | 0.090 | 0.031 | 1.000 | 0.694 | 1.279 | 4.659 | 0.17 |
| qwen1p7b_k8_single16 | Qwen3-1.7B | delta | read=bank | routing=single | slots=16 | train=synthetic K=8 | 3000 | synthetic_k8 | 0.438 | 0.094 | 0.031 | 1.000 | 0.817 | 1.314 | 4.803 | 0.17 |
| qwen1p7b_k8_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.469 | 0.102 | 0.031 | 1.000 | 0.706 | 1.512 | 4.723 | 0.13 |
| qwen1p7b_k8_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.695 | 0.074 | 0.031 | 1.000 | 0.458 | 1.539 | 4.659 | 0.13 |
| qwen1p7b_k8_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.551 | 0.066 | 0.031 | 1.000 | 0.648 | 1.531 | 4.803 | 0.13 |
| qwen1p7b_k8_single_qread_10k | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 10000 | synthetic_k16 | 0.504 | 0.078 | 0.031 | 1.000 | 0.653 | 1.810 | 4.723 | 0.12 |
| qwen1p7b_k8_single_qread_10k | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 10000 | synthetic_k4 | 0.711 | 0.055 | 0.031 | 1.000 | 0.394 | 1.867 | 4.659 | 0.12 |
| qwen1p7b_k8_single_qread_10k | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 10000 | synthetic_k8 | 0.586 | 0.070 | 0.031 | 1.000 | 0.539 | 1.932 | 4.803 | 0.12 |
| qwen1p7b_k8_single_qread_ai | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.492 | 0.090 | 0.031 | 1.000 | 0.710 | 1.568 | 4.723 | 0.13 |
| qwen1p7b_k8_single_qread_ai | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.680 | 0.090 | 0.031 | 1.000 | 0.461 | 1.659 | 4.659 | 0.13 |
| qwen1p7b_k8_single_qread_ai | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.586 | 0.074 | 0.031 | 1.000 | 0.615 | 1.706 | 4.803 | 0.13 |
| qwen1p7b_squad_single_qread | Qwen3-1.7B | delta | read=query | routing=single | slots=1/pos | train=squad | 3000 | squad_val | 0.086 | 0.074 | 0.059 | 0.523 | 2.468 | 2.469 | 5.774 | 0.19 |
| qwen4b_k8_single_qread | Qwen3-4B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k16 | 0.398 | 0.090 | 0.047 | 1.000 | 0.803 | 1.381 | 4.511 | 0.08 |
| qwen4b_k8_single_qread | Qwen3-4B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k4 | 0.633 | 0.086 | 0.035 | 1.000 | 0.590 | 1.467 | 4.599 | 0.07 |
| qwen4b_k8_single_qread | Qwen3-4B | delta | read=query | routing=single | slots=1/pos | train=synthetic K=8 | 3000 | synthetic_k8 | 0.492 | 0.051 | 0.047 | 0.996 | 0.734 | 1.488 | 4.481 | 0.07 |

## Learning curves (correct EM / donor EM on the training-distribution eval set)

- **gemma4e4b_k8_kvbank_qread_ai** (synthetic_k8): 0: 0.02/0.02, 500: 0.07/0.08, 1000: 0.10/0.11, 1500: 0.14/0.07, 2000: 0.23/0.11, 2500: 0.23/0.09, 3000: 0.29/0.08
- **gemma4e4b_k8_kvbank_qread_ai_sliding** (synthetic_k8): 0: 0.02/0.02, 500: 0.10/0.09, 1000: 0.13/0.10, 1500: 0.25/0.08, 2000: 0.35/0.09, 2500: 0.43/0.07, 3000: 0.51/0.06
- **gemma4e4b_k8_single16** (synthetic_k8): 0: 0.02/0.02, 500: 0.08/0.08, 1000: 0.11/0.10, 1500: 0.13/0.08, 2000: 0.17/0.09, 2500: 0.18/0.08, 3000: 0.18/0.11
- **gemma4e4b_k8_single_qread** (synthetic_k8): 0: 0.02/0.02, 500: 0.09/0.09, 1000: 0.09/0.09, 1500: 0.07/0.08, 2000: 0.10/0.09, 2500: 0.09/0.07, 3000: 0.07/0.05
- **gemma4e4b_k8_single_qread_ai** (synthetic_k8): 0: 0.02/0.02, 500: 0.07/0.08, 1000: 0.11/0.07, 1500: 0.13/0.07, 2000: 0.15/0.09, 2500: 0.16/0.09, 3000: 0.23/0.07
- **qwen1p7b_k2_single_qread** (synthetic_k2): 0: 0.04/0.03, 500: 0.44/0.07, 1000: 0.75/0.07, 1500: 0.78/0.06, 2000: 0.78/0.06, 2500: 0.80/0.05, 3000: 0.79/0.04
- **qwen1p7b_k8_chunk4x4** (synthetic_k8): 0: 0.03/0.03, 500: 0.07/0.08, 1000: 0.06/0.07, 1500: 0.14/0.09, 2000: 0.20/0.07, 2500: 0.27/0.07, 3000: 0.30/0.08
- **qwen1p7b_k8_cosine4x4** (synthetic_k8): 0: 0.03/0.03, 500: 0.07/0.06, 1000: 0.23/0.06, 1500: 0.27/0.08, 2000: 0.38/0.08, 2500: 0.44/0.10, 3000: 0.45/0.09
- **qwen1p7b_k8_kvbank_qread** (synthetic_k8): 0: 0.03/0.03, 500: 0.09/0.07, 1000: 0.24/0.06, 1500: 0.41/0.08, 2000: 0.54/0.06, 2500: 0.61/0.08, 3000: 0.60/0.07
- **qwen1p7b_k8_single16** (synthetic_k8): 0: 0.03/0.03, 500: 0.06/0.07, 1000: 0.14/0.10, 1500: 0.19/0.07, 2000: 0.30/0.08, 2500: 0.39/0.08, 3000: 0.44/0.09
- **qwen1p7b_k8_single_qread** (synthetic_k8): 0: 0.03/0.03, 500: 0.07/0.08, 1000: 0.23/0.08, 1500: 0.33/0.08, 2000: 0.50/0.07, 2500: 0.55/0.09, 3000: 0.55/0.07
- **qwen1p7b_k8_single_qread_10k** (synthetic_k8): 0: 0.03/0.03, 1000: 0.15/0.09, 2000: 0.36/0.11, 3000: 0.48/0.06, 4000: 0.53/0.06, 5000: 0.56/0.07, 6000: 0.59/0.05, 7000: 0.57/0.05, 8000: 0.59/0.06, 9000: 0.59/0.07, 10000: 0.59/0.07
- **qwen1p7b_k8_single_qread_ai** (synthetic_k8): 0: 0.03/0.03, 500: 0.09/0.06, 1000: 0.25/0.04, 1500: 0.44/0.09, 2000: 0.54/0.05, 2500: 0.57/0.04, 3000: 0.59/0.07
- **qwen1p7b_squad_single_qread** (squad_val): 0: 0.06/0.06, 500: 0.07/0.07, 1000: 0.07/0.07, 1500: 0.06/0.06, 2000: 0.06/0.07, 2500: 0.06/0.07, 3000: 0.09/0.07
- **qwen4b_k8_single_qread** (synthetic_k8): 0: 0.05/0.05, 500: 0.09/0.08, 1000: 0.21/0.10, 1500: 0.25/0.08, 2000: 0.38/0.10, 2500: 0.43/0.08, 3000: 0.49/0.05
