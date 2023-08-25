# How to use a pre-trained Region2Vec model
Region2Vec is an unsupervised method for creating embeddings of genomic regions and genomic region sets. This tutorial discusses how to use pre-trained models. To learn how to train a new model see the [region2vec training documentation](./train-region2vec.md) We make available several pre-trained models available on our [huggingface repo](https://huggingface.co/databio). These models can be used to create embeddings of genomic regions and region sets without having to train a new model.

## Get the model
To use one of our pre-trained models, simply import the `Region2VecExModel` and download the model from huggingface:

```python
from gitk.io import Region
from gitk.region2vec import Region2VecExModel

model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38")
```

> Note: We use the `Region2VecExModel` class to load the model because it is an extension of the `Region2Vec` class that comes with its own tokenizer. This ensures that the model and tokenizer are compatible.

## Encode regions
Now we can encode a genomic region or a list of regions:

```python
region = Region("chr1", 160100, 160200)
regions = [region] * 10

embedding = model.encode(region)
embeddings = model.encode(regions)
```

We can also encode an entire bed file:

```python
bed = "/path/to/bed/file.bed"

embeddings = model.encode(bed)
```

> Note: It is not uncommon for a region to not be tokenizable by the tokenizer. This is because the tokenizer was trained on a specific set of regions. If this is the case, the model simply returns `None` for the embedding of that region. If you want to override this behavior, you can set `return_none=False` in the `encode` function. This will return a zero vector for the embedding of the region. However, we do not recommend this as it removes the ability to distinguish between regions that are tokenizable and regions that are not.

## Region pooling
It is often the case that we want a single embedding that represents a set of regions. For example, we may want to encode a patient by taking the average embedding of all the SNPs in the patient's genome. We can do this by using the `pool` argument. Out of the box, we support `mean` and `max` pooling. For example:

```python
patient_snps = "/path/to/bed/file.bed"

embedding = model.encode(patient_snps, pool="mean") # or pool="max"
```

You can also supply a custom function to do pooling. For example, if you wanted to take the sum of the embeddings, you could do:

```python
def sum_pooling(embeddings: np.ndarray):
    return np.sum(embeddings, axis=0)

embedding = model.encode(patient_snps, pool=sum_pooling)
```

`pool` is `False` by default. Setting to `True` defaults to `mean` pooling.
