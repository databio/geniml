# Pre-tokening training data
## Overview

Before we train a model, we must do what is called *pre-tokenization*. Pre-tokenziation is the process of converting raw genomic region data into lists of tokens and saved into a special file format we call `.gtok` (genomic token) files. These are binary files that contain the tokenized data in the form of integers. The `.gtok` files are used directly to train the model. There are several benefits to this, including:
1. **Speed**: The tokenization process can be slow, especially when you have a lot of data. By pre-tokenizing the data, you only have to do this once. Then, you can use the `.gtok` files to train the model as many times as you want.
2. **Memory**: The `.gtok` files are much smaller than the original data. This means that you can store more data in memory and train larger models. Moreover, this enables streaming the data from disk, which is useful when you have a lot of data.
3. **Reproducibility**: By saving the tokenized data, you can ensure that the same data is used to train the model every time. This is important for reproducibility.

## How to pretokenize data
Pretokenizing data is easy. You can use the built-in tokenizers and utilities in `geniml` to do this. Here is an example of how to pretokenize a bed file:

```python
from genimtools.utils import write_tokens_to_gtok
from geniml.tokenization import ITTokenizer

# instantiate a tokenizer
tokenizer = ITTokenizer("path/to/universe.bed")

# get tokens
tokens = tokenizer.tokenize("path/to/bedfile.bed")
write_tokens_to_gtok(tokens.ids, "path/to/bedfile.gtok")
```

Thats it! Now you can use the `.gtok` file to train a model.

## How to use the `.gtok` files

To facilitate working with `.gtok` files, we have some helper-classes that can be used to train a model directly from `.gtok` files. For example, you can use teh `Region2VecDataset` class to load the `.gtok` files and train a model. See the [training documentation](./train-region2vec.md) for more information.

```python
from geniml.region2vec.utils import Region2VecDataset

tokens_dir = "path/to/tokens"
dataset = Region2VecDataset(tokens_dir)

for tokens in dataset:
    # train the model
    print(tokens) # [42, 101, 99, ...]
```

## Caveats and considerations
When pretokenizing data, you should consider the following:
1. Tokens are specific to the universe that the tokenizer was trained on. If you use a different universe, you will get different tokens. This means that you should use the same universe to pretokenize the data as you will use to train the model.
2. The `.gtok` files are binary files. This means that they are not human-readable. You should keep the original bed files as well as the `.gtok` files. This is important for reproducibility and for debugging.