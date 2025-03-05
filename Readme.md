# GPT-2 Training Script

## Overview
This script provides an implementation of a GPT-2 training pipeline using PyTorch. It allows training from scratch or fine-tuning a pretrained GPT-2 model using a tokenized dataset.

## Features
- Implements GPT-2 from scratch using PyTorch.
- Supports loading and fine-tuning OpenAI's pretrained GPT-2 models.
- Uses distributed training with PyTorch's DistributedDataParallel (DDP).
- Includes validation using HellaSwag for evaluation.
- Supports efficient tokenized dataset loading.

## Requirements
Before running the script, install the required dependencies:

```bash
pip install torch transformers numpy tiktoken
