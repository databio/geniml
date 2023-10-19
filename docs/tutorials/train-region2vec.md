# How to train a new Region2Vec model
Region2Vec is an unsupervised method for creating embeddings of genomic regions and genomic region sets. This tutorial discusses how to train a new model. To learn how to use a pre-trained model see the [region2vec usage documentation](./use-pretrained-region2vec-model.md).

## Training data and tokenization
Training a model requires two things: 1) a set of genomic regions and 2) a tokenizer. The tokenizer is used to convert the genomic regions into tokens. The tokens are then used to train the model. A safe choice for the tokenizer is the tiled hg38 genome. However, you can define your own tokenizer if you want to use a different genome or if you want to use a different tokenization strategy.

You can download the 1000 tiles hg38 genome [here](https://big.databio.org/gitk/universes/tiles1000.hg38.bed).

## Training a model
### Instantiate a new model
To begin, create a new model from `Region2VecExModel`.

> Note: We use the `Region2VecExModel` because it is an extension of the `Region2Vec` class that comes with its own tokenizer. This ensures that the model and tokenizer are compatible.

```python
import logging
import os
from multiprocessing import cpu_count

from gitk.io import RegionSet
from gitk.tokenization import InMemTokenizer
from gitk.region2vec import Region2VecExModel
from tqdm.rich import tqdm


logging.basicConfig(level=logging.INFO)

# get the paths to data
universe_path = os.path.expandvars("$RESOURCES/regions/genome_tiles/tiles1000.hg38.bed")
data_path = os.path.expandvars("$DATA/ChIP-Atlas/hg38/ATAC_seq/")

model = Region2VecExModel(
    threads=cpu_count() - 2,
    tokenizer=InMemTokenizer(universe_path),
)
```

### Training data
Create a list of `RegionSet`s by supplying paths to bed files:
```python
# get list of all files in the data directory
# convert to backed RegionSet objects
files = os.listdir(data_path)
data = [
    RegionSet(os.path.join(data_path, f), backed=True)
    for f in tqdm(files, total=len(files))
]
```

> Note: Setting `backed=True` means that the data will be loaded into memory lazily. This is useful when you have a lot of data and you don't want to load it all into memory at once.

## Training
Now, simply give the model the list of `RegionSet`s and run the training:
```python
# train the model
model.train(data, epochs=100)
```

You can export your model using the `export` function:

```python
model.export("out")
```

These files are intended to be directly uploaded to huggingface. You can upload them using the `huggingface-cli` or the [huggingface website](https://huggingface.co/new).
