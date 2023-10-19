# How to use the tokenizers
## Overview
The `geniml` tokenizers are used to prepare data for training, evaluation, and inference of genomic machine learning models. Like tokenizers for natural langauge processing, the `geniml` tokenizers convert raw data into a format that can be used by our models. `geniml` has a few tokenizers, but they all follow the same principles.

All tokenizers require a *universe file* (or, vocab file). This is a bedfile that contains all possible regions that can be tokenized. It may also include special tokens like the start, end, unknown, and padding token.

In addition to tokenizers implemented here, we also have a standalone package called [`gtokenizers`](https://github.com/databio/gtokenizers) which provides tokenizer implementations in Rust with python bindings. The Rust implementations are much faster than the python implementations. We recommend using the Rust implementations whenever possible.

## Using the tokenizers
To start using a tokenizer, simply pass it an appropriate universe file:

```python
from geniml.tokenization import ITTokenizer # or any other tokenizer
from geniml.io import RegionSet

rs = RegionSet("/path/to/file.bed")
t = ITTokenizer("/path/to/universe.bed")

tokens = t.tokenize(rs)
for token in tokens:
    print(f"{t.chr}:{t.start}-{t.end}")
```

You can also get token ids for the tokens:

```python
from geniml.tokenization import ITTokenizer # or any other tokenizer
from geniml.io import RegionSet

rs = RegionSet("/path/to/file.bed")
t = ITTokenizer("/path/to/universe.bed")

model = Region2Vec(len(t), 100) # 100 dimensional embedding
tokens = t.tokenize(rs)

out = model(tokens.ids)
print(out.shape)
```

## Future work
Genomic region tokenization is an active area of research. We will implement new tokenizers as they are developed. If you have a tokenizer you'd like to see implemented, please open an issue or submit a pull request.

For core development of our tokenizers, see the [gtokenizers](https://github.com/databio/gtokenizers) repository.
