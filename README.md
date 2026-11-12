# ORIENT: Online Reinforcement Learning for Data Selection in MLLM Instruction Tuning

## Introduction

**ORIENT** is an online data selection method for multimodal instruction tuning. It selects training examples based on the current model state and updates the selection policy using a fixed feedback set, integrating data selection with model fine-tuning.

This repository provides a LLaVA-1.5 implementation with LoRA, distributed training, reference caching, and accelerated online scoring.

## Technical Supplement

[Download the technical supplement (PDF)](docs/ORIENT-supplement.pdf).

The supplement provides detailed methods, protocols, and additional analyses:

- Exact constrained subset inference, policy statistics, and numerical details.
- The complete 191-task Vision-Flan-to-LLaVA mapping used for frozen policy transfer, with assignment rules and boundary cases.
- Complete scores for all 15 signal-range rules, image diagnostics, and the exact enrichment denominator.
- Batch-grouping replay statistics, fixed-checkpoint gradient comparisons, and all 33 metric-omission results.

## Installation

### 1. Environment Setup

Follow the [LLaVA installation guide](https://github.com/haotian-liu/LLaVA#install) to set up the environment. This implementation uses LLaVA `1.2.2.post1`, Python 3.10, and PyTorch 2.1.2. Install ORIENT from the repository root:

```bash
conda activate llava
python -m pip install -e ".[training]" "numpy==1.26.4"
```

If LLaVA is available locally as source code only, set `export LLAVA_SOURCE=/path/to/LLaVA`. The launch scripts will add it to the Python path.

### 2. Model Preparation

- **Initial model**: a complete LLaVA-1.5 Llama multimodal checkpoint and tokenizer.
- **Reference model**: a frozen reference checkpoint or a complete LoRA export that includes projector weights.
- **Vision encoder**: the corresponding CLIP model and image preprocessing configuration.

We release our model trained with ORIENT using 20% of the LLaVA-665K dataset:
**Trained model**: [llava-v1.5-7b-lora-orient-top20](https://huggingface.co/JunmingCHEN123/llava-v1.5-7b-lora-orient-top20).

Set the model and data paths in [scripts/config.sh](scripts/config.sh), or override them with environment variables of the same names.

## Data Preparation

To support reproducibility, we release the subset selected during ORIENT's online training:
**ORIENT 133K subset**: [llava-orient-data-133k](https://huggingface.co/datasets/JunmingCHEN123/llava-orient-data-133k).

### 1. Data Layout

```text
data/
├── train.json         # Instruction data in LLaVA format
├── images/            # Image root directory
├── metadata.json      # Generated task labels and feedback split
└── reference.json     # Generated reference loss cache
```

`train.json` uses the native LLaVA JSON array format. Each example contains `conversations` and `task`. Image examples also include an `image` path relative to the image root; text-only examples omit this field. All examples within a task must use the same modality.

### 2. Prepare the Fixed Feedback Set

```bash
bash scripts/prepare.sh
```

By default, 20 examples per task are reserved for feedback, and the rest form the candidate pool. To use separate task files, append `--task-root data/tasks` and provide `file1_665.json` through `file10_665.json` in that directory.

### 3. Build the Reference Cache

```bash
bash scripts/build_reference.sh
```

Append `--gpus 0,1` to generate the cache on multiple GPUs. Rerun the same command to resume after an interruption. Generate a new cache whenever the data, model, or preprocessing settings change.

## Training

### Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 B=64 bash scripts/train.sh
```

`B` must equal the per-device batch size × gradient accumulation steps × number of processes. The script defaults to a per-device batch size of 8 and 4 gradient accumulation steps. Append `--dry-run` to check data alignment and selection budgets before training.

**Outputs**: `outputs/orient-run/training/` contains the LoRA model and projector, `policy.pt` stores the selection policy, and `selected_indices.json` records the selected examples' original row indices. Use a new output directory for each run.

### Policy Update Control

Each completed stage produces one PPO call with at most two Actor steps and one Critic step. The Actor steps share the original sampling policy and a fixed advantage. After each full Actor step, the exact KL divergence is checked against the original sampling policy with a limit of `0.03`. A step exceeding the limit is immediately reverted, restoring its Actor parameters, AdamW moments, and optimizer step counters. Any earlier accepted Actor step and the Critic update are retained, and the call stops further Actor updates. The `accepted_fraction` log field is `1.0` for an accepted step and `0.0` for a reverted step.

## Repository Structure

```text
orient/                    # Online selection, scoring, and training
scripts/
├── config.sh              # Shared paths and configuration
├── prepare.sh             # Data and feedback set preparation
├── build_reference.sh     # Reference cache generation
└── train.sh               # Single-GPU and multi-GPU training
tests/                     # Regression tests
```

## Features

- **Online selection**: selects training data using current model scores, task quotas, and semantic coverage.
- **Feedback learning**: updates the Actor/Critic using loss changes on a fixed feedback set.
- **Caching and acceleration**: supports resumable reference cache generation, input caching, frozen vision feature reuse, and candidate input prefetching.
- **Native training**: supports LLaVA argument passthrough, modality-length grouping, gradient accumulation, and continuous optimizer/scheduler state across stages.

## Configuration

| Option | Description |
| --- | --- |
| `scripts/config.sh` | Data, model, cache, and output paths |
| `RHO` | Data selection ratio; default: `0.2` |
| `B` / `N` | Global batch size / SFT updates per full stage; defaults: `32` / `2` |
| `NPROC_PER_NODE` | Number of training processes; default: `1` |
| `ATTN_IMPLEMENTATION` | Default: `eager`; use `flash_attention_2` if FlashAttention is installed |

Append native LLaVA arguments directly to the script, for example:

```bash
bash scripts/train.sh \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --group_by_modality_length True
```

Run `python -m orient.train --help` for additional method options.

## License

The project license is pending. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party license information.

## Acknowledgments

This project is based on [LLaVA](https://github.com/haotian-liu/LLaVA) and uses PyTorch, Transformers, PEFT, and Accelerate for training.
