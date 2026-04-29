# SFT Reasoning Dataset

This folder contains datasets for SFT Reasoning training experiments on `Qwen/Qwen2.5-Math-1.5B`.

## Training & Evaluation Datasets
### Final SFT Training And Val Datasets
These are the datasets used for the SFT reasoning experiments:

- Training Datasets
| File | Examples | Description |
|------|----------|-------------|
| `sft_gpt-oss-120b.jsonl` | 4,836 | Full SFT dataset with reasoning traces (contains both correct and incorrect answers) |
| `sft_gpt-oss-120b_filtered.jsonl` | 3,496 | **Recommended** - Filtered dataset containing only reasoning traces that lead to correct answers |

  Each example contains: `problem`, `reasoning_trace`, `extracted_answer`, `expected_answer`

- Validation Dataset
| File | Examples | Description |
|------|----------|-------------|
| `val.jsonl` | ~5K | Validation dataset used for evaluation (extracted from CS336 Assignment 5) |

## Dataset Creation Pipeline

The training datasets were created through the following pipeline:

1) Source Data
| File | Description |
|------|-------------|
| `train.jsonl` | Training problems from `hiyouga/math12k` dataset, filtered to exclude validation set problems |
| `math_results.jsonl` | Original validation data from CS336 Assignment 5 |

2) Batch Inference using fireworks `gpt-oss-120b` api
| File | Description |
|------|-------------|
| `train_data_4_batchinference_gpt-oss-120b.jsonl` | Training problems formatted with prompts for batch inference |
| `batch-infer-math-train-outputs_gpt-oss-120b.jsonl` | Raw batch inference outputs from `gpt-oss-120b` model |

3) Final Processing
The final SFT datasets (`sft_gpt-oss-120b.jsonl` and filtered version) were created by:
1. Filtering for successfully completed responses (`finish_reason=stop`)
2. Extracting reasoning traces and answers using regex
3. Matching extracted answers with expected answers for the filtered version


## Baseline Results
| File | Description |
|------|-------------|
| `baseline_results.jsonl` | Baseline evaluation results for untrained `Qwen/Qwen2.5-Math-1.5B` on validation set |

## Reference
Part of CS336 Assignment 5 (SFT Reasoning). See [building-from-scratch/sft](https://github.com/garg-aayush/building-from-scratch/tree/main/sft) for training code and details.
