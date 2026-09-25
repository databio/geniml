# Train Atacformer from scratch

This tutorial will guide you through the process of training an Atacformer model from scratch using genomic interval data. We will cover the necessary steps, including data preparation, model configuration, and training.

## Overview

The pre-training process uses a replaced token detection objective (similar to [ELECTRA](https://arxiv.org/abs/2003.10555)) to learn representations of genomic regions and their accessibility patterns.

## Prerequisites

Before starting, ensure you have:

- Access to pre-tokenized single-cell ATAC-seq data
- A universe file (`.bed.gz`) defining the genomic regions of interest
- GPU resources for training

## Model Configuration

You can configure the Atacformer model however you like. This is what was used in the original paper:

```python
config = AtacformerConfig(
    vocab_size=tokenizer.vocab_size,    # Based on universe file
    hidden_size=192,                    # Hidden dimension size
    num_hidden_layers=6,                # Number of transformer layers
    num_attention_heads=8,              # Number of attention heads
    intermediate_size=768,              # Feed-forward network size
    max_position_embeddings=8192,       # Maximum sequence length
    pad_token_id=tokenizer.pad_token_id,
)
```

## Module imports

First, import the necessary modules:

```python
import torch

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from atacformer import (
    AtacformerConfig,
    AtacformerForReplacedTokenDetection,
    DataCollatorForReplacedTokenDetection,
    TrainingTokenizer,
    get_decaying_cosine_with_hard_restarts_schedule_with_warmup,
)
```

## Training setup

Set up `torch`

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
```

Make sure your hyperparameters are set up correctly:

```python
MLM_PROBABILITY = 0.45
BATCH_SIZE = 16
MAX_LEARNING_RATE = 1.5e-4

RUN_NAME = "atacformer-pretraining"
```

## Data Preparation

Then load your dataset. The training expects a pre-tokenized dataset in Parquet format with the following columns:

- `input_ids`: Tokenized genomic regions
- Additional metadata columns (removed during training)

```python
dataset_path = "path/to/dataset.parquet"
tokenized_dataset = Dataset.from_parquet(dataset_path)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
```

## Tokenizer Setup

Next, set up the tokenizer:

```python
tokenizer = TrainingTokenizer("path/to/universe.bed.gz")
```

The tokenizer is created from a universe file that defines the genomic regions:

```python
tokenizer = TrainingTokenizer(universe_bed_file)
```

## Model and Training Arguments

Instantiate a new Atacformer model with the replaced token detection (RTD) objective. We move the model to `bfloat16` because its much faster when using "ampere" GPUs:

```python
config = AtacformerConfig(
    use_pos_embeddings=False,
    vocab_size=tokenizer.vocab_size,
    hidden_size=192,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=768,
    max_position_embeddings=8192,
    pad_token_id=tokenizer.pad_token_id,
)
model = AtacformerForReplacedTokenDetection(config)
model = model.to(torch.bfloat16)

print(f"Model size: {model.num_parameters()} parameters")
```

## Data Collation

The `DataCollatorForReplacedTokenDetection` handles:
- Random token replacement based on `mlm_probability`
- Proper masking and attention handling
- Batch preparation for training

Create the data collator:

```python
data_collator = DataCollatorForReplacedTokenDetection(
    tokenizer=tokenizer,
    mlm_probability=MLM_PROBABILITY,
)
```

## Setup the Trainer

Now, set up the `Trainer` with the model, training arguments, and data collator:

```python
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=25,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_strategy="steps",
    logging_steps=10,
    run_name=RUN_NAME,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=500,
    learning_rate=MAX_LEARNING_RATE,
    bf16=True,
    max_grad_norm=1.0,
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)
```

## Training

Finally, start the training process:

```python
trainer.train()
model.save_pretrained("output/atacformer-pretrained")
```