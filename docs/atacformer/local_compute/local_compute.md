Because of the nature of scATAC-seq data, Atacformer is trained using a large context size (8,192 tokens). This allows the model to capture long-range dependencies in the genomic data, which is crucial for understanding regulatory elements and their interactions.

However, this large context size also means that Atacformer requires significant computational resources for training and inference. While we recommend using a GPU when running Atacformer, it is possible to artificially limit the context size during inference to reduce memory usage.

You can do this in two ways:
1. Manually truncating the input sequences to a smaller size before passing them to the model.
2. Augmenting the model's configuration to set a smaller maximum sequence length.

## Manually Truncating Input Sequences
If you want to limit the context size during inference, you can manually truncate your input sequences. For example, if you want to limit the context size to 1,024 tokens, you can do this:
```python
from geniml.atacformer import AtacformerForCellClustering
from gtars.tokenizers import Tokenizer

model = AtacformerForCellClustering.from_pretrained("databio/atacformer-craft100k-hg38")
tokenizer = Tokenizer.from_pretrained("databio/atacformer-craft100k-hg38")

# Load your data and tokenize it
input_ids = ...  # your tokenized input ids

# Truncate to 1024 tokens
input_ids = [ids[:1024] for ids in input_ids]
```

## Augmenting the Model's Configuration
Manual truncation might not be desired as it biases the model towards tokens at the beginning of the sequence. A better approach would randomly sample a subset of the input ids to create a smaller context size. this is done automatically for you when your sequence exceeds the maximum sequence length defined in the model's configuration.

To that end, you can augment the model's configuration to set a smaller maximum sequence length. This way, the model will automatically handle the truncation for you.

```python
model.config.max_position_embeddings = 1024  # set to your desired context size
```