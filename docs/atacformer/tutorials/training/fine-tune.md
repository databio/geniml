# Fine-tune Atacformer on your data

Often, you will want to fine-tune a pre-trained Atacformer model on your own dataset. This is a common practice in transfer learning, where you take a model that has been pre-trained on a large dataset and adapt it to your specific dataset.

## Prerequisites
Before starting, ensure you have:

- A pre-trained Atacformer model (e.g., `databio/atacformer-base-hg38`)
- Pre-tokenized your dataset. If not, see the [pre-tokenize for training guide.](pretokenize-for-training.md)

## Setup training

We use a mixture of `geniml` and the `transformers` library to run training.

```python
import torch

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from atacformer import (
    AtacformerForReplacedTokenDetection,
    DataCollatorForReplacedTokenDetection,
    TrainingTokenizer,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
# optional for experiment tracking
# os.environ["WANDB_PROJECT"] = "atacformer-pretraining"

MLM_PROBABILITY = 0.45
BATCH_SIZE = 32
MAX_LEARNING_RATE = 1.5e-4
RUN_NAME = "atacformer-fine-tuning"

MODEL_TO_FINE_TUNE = "databio/atacformer-base-hg38"
```

## Load your dataset

Load your pre-tokenized dataset. The training expects a pre-tokenized dataset in Parquet format with the following columns:
- `input_ids`: Tokenized genomic regions

```python
dataset_path = "path/to/dataset.parquet"
tokenized_dataset = Dataset.from_parquet(dataset_path)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
```

## Create the tokenizer and data collator

Next, set up the tokenizer. The tokenizer is created from a universe file that defines the genomic regions:

```python
tokenizer = TrainingTokenizer.from_pretrained(MODEL_TO_FINE_TUNE)
data_collator = DataCollatorForReplacedTokenDetection(
    tokenizer=tokenizer,
    mlm_probability=MLM_PROBABILITY,
)
```

## Create the model

Grab the pre-trained model weights from the Hugging Face Hub:

```python
model = AtacformerForReplacedTokenDetection.from_pretrained(MODEL_TO_FINE_TUNE) 
model = model.to(torch.bfloat16)  # use bfloat16 for training (its faster on ampere GPUs)
```

## Training arguments
Set up the training arguments. You can adjust these based on your hardware and dataset size:

```python
training_args = TrainingArguments(
    output_dir="atacformer-fine-tuning-output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    run_name=RUN_NAME,
    warmup_steps=500,
    lr_scheduler_type="cosine_with_restarts",
    learning_rate=MAX_LEARNING_RATE,
    bf16=True,
    max_grad_norm=1.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)
```

## Train your model

Finally, start the training process:

```python
trainer.train()
model.save_pretrained("output/atacformer-fine-tuned")
```

## Evaluate your model

Your model can now be used like any other Atacformer model. You can evaluate it on your test set or use it for downstream tasks such as cell clustering, classification, or regression.