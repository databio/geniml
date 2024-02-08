# How to train a new Region2Vec model
Region2Vec is an unsupervised method for creating embeddings of genomic regions and genomic region sets. This tutorial discusses how to train a new model. To learn how to use a pre-trained model see the [region2vec usage documentation](./use-pretrained-region2vec-model.md).

## Training data and universe
Training a model requires two things: 1) a set of pre-tokenized data and 2) a universe. The universe is a set of regions that the model will be trained on. The universe is used to create the tokenizer, which is used to convert the raw data into tokens. The universe should be representative of the data that you will be training the model on. For example, if you are training a model on human data, you should use a universe that contains human regions. If you dont have a universe, a safe bet is to use the 1000 tiles hg38 genome.

You can download the 1000 tiles hg38 genome [here](https://big.databio.org/geniml/universes/tiles1000.hg38.bed).

The pre-tokenized data is a set of `.gtok` files. These are binary files that contain the tokenized data in the form of integers. The `.gtok` files are used directly to train the model. If you have not pre-tokenized your data, see the [pre-tokenization documentation](./pre-tokenization.md).

## Training a model
### Instantiate a new model
To begin, create a new model from `Region2VecExModel`.

> Note: We use the `Region2VecExModel` because it is an extension of the `Region2Vec` class that comes with its own tokenizer. This ensures that the model and tokenizer are compatible.

```python
import logging
import os
from multiprocessing import cpu_count

from geniml.io import RegionSet
from geniml.tokenization import ITTokenizer
from geniml.region2vec import Region2VecExModel
from rich.progress import track


logging.basicConfig(level=logging.INFO)

# get the paths to data
universe_path = os.path.expandvars("$RESOURCES/regions/genome_tiles/tiles1000.hg38.bed")
data_path = os.path.expandvars("$DATA/ChIP-Atlas/hg38/ATAC_seq/tokens")

model = Region2VecExModel(
    tokenizer=ITTokenizer(universe_path),
)
```

### Training data

The training data is a set of `.gtok` files. You can use the `Region2VecDataset` class to load the `.gtok` files and train the model.
```python
from geniml.region2vec.utils import Region2VecDataset

dataset = Region2VecDataset(data_path)
```

## Training

Now, simply give the model the list of `RegionSet`s and run the training:
```python
# train the model
model.train(dataset, epochs=100)
```

You can export your model using the `export` function:

```python
model.export("out")
```

These files are intended to be directly uploaded to huggingface. You can upload them using the `huggingface-cli` or the [huggingface website](https://huggingface.co/new).
