# Atacformer core concepts

Using an Atacformer model requires two steps: **1) tokenization** and **2) model inference**. Step one (tokenization) converts your data into a format that can be processed by the model. Step two actually performs the inference, using the processed data to make predictions and generate embeddings.

All tutorials and documentation related to Atacformer will follow this two-step process:
> Tokenize first, then infer.

Whether you are working with single-cell data, bulk data, ATAC-seq, ChIP-seq, or fragments files -- they all follow the same two-step process: *tokenize first, then infer*.
<p align="center">
    <img align="center" src="../img/core_concepts.svg" width="600">
</p>

## Infrastructure
The Atacformer models and training infrastructure are built on top of the [`transformers`](https://github.com/huggingface/transformers) library, which provides a robust framework for building and training transformer-based models. The Atacformer models are designed to be compatible with the Hugging Face ecosystem, allowing users to easily leverage pre-trained models, while customizing them for specific tasks.

## Tokenization
Atacformer is designed to handle genomic interval data. It can process _anything that can be represented by a chrom, start, and end_. To represent this data, each Atacformer model has a specified vocabulary of genomic intervals that it can process represented as a `.bed` file. Oftentimes, this is referred to as a consensus peak file. We call it a **universe**. When new data is to be processed, it is first "tokenized" into the model's universe, which is a process of mapping the genomic intervals in the data to the intervals in the model's universe.

<p align="center">
    <img align="center" src="../img/tokenization.svg" width="600">
</p>

Once tokenized, each token is mapped to an integer ID, which is subsequently converted to a dense embedding vector. This vector is then used as input to the Atacformer model.

## Pre-trained models and their tokenizers
Atacformer models are pre-trained on large datasets of genomic interval data. Each model has a corresponding tokenizer that is used to preprocess the data before it is fed into the model. The tokenizer is responsible for converting genomic intervals into a format that the model can understand. It also has knowledge of special tokens like `<pad>`, `<mask>`, and `<cls>` that are used in the model's architecture.

When you want to use a pre-trained Atacformer model, you should take care to use the corresponding tokenizer. For example, if you want to use the `atacformer-base-hg38` model, you should also use the `atacformer-base-hg38` tokenizer:

```python
from gtars.tokenizers import Tokenizer
from geniml.atacformer import AtacformerForCellClustering

tokenizer = Tokenizer.from_pretrained("databio/atacformer-base-hg38")
model = AtacformerForCellClustering.from_pretrained("databio/atacformer-base-hg38")
```

## Tokenizer API

We designed the tokenizer API to be as similar as possible to the Hugging Face tokenizer API, so you can use it in a similar way. If you're familiar with Hugging Face tokenizers, you can use the Atacformer tokenizer in a similar way:

```python
from gtars.models import RegionSet
from gtars.tokenizers import Tokenizer

rs = RegionSet.from_bed("path/to/your/regions.bed")
tokenizer = Tokenizer.from_pretrained("databio/atacformer-base-hg38")

# get the input IDs for the regions
tokens = tokenizer(rs)
input_ids = tokens["input_ids"]

# decode the input IDs back to regions
decoded_regions = tokenizer.decode(input_ids)

# get properties of the tokenizer
vocab_size = tokenizer.vocab_size
special_tokens = tokenizer.special_tokens_map
```
