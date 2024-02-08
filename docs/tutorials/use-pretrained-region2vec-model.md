# How to use a pre-trained Region2Vec model
Region2Vec is an unsupervised method for creating embeddings of genomic regions and genomic region sets. This tutorial discusses how to use pre-trained models. To learn how to train a new model see the [region2vec training documentation](./train-region2vec.md) We make available several pre-trained models available on our [huggingface repo](https://huggingface.co/databio). These models can be used to create embeddings of genomic regions and region sets without having to train a new model.

## Get the model
To use one of our pre-trained models, simply import the `Region2VecExModel` and download the model from huggingface:

```python
from geniml.io import Region
from geniml.region2vec import Region2VecExModel

model = Region2VecExModel("databio/r2v-encode-hg38")
```

> Note: We use the `Region2VecExModel` class to load the model because it is an extension of the `Region2Vec` class that comes with its own tokenizer. This ensures that the model and tokenizer are compatible.

## Encode regions
Now we can encode a genomic region or a list of regions:

```python
region = Region("chr1", 160100, 160200)
regions = [region] * 10

embedding = model.encode(region) # get one region embedding
embeddings = model.encode(regions) # or, get many embeddings
```

We can also encode an entire bed file, which will return region embeddings for each region in the file:

```python
bed = "/path/to/bed/file.bed"

embeddings = model.encode(bed)
```

> Note: It is possible that a region can not be tokenized by the tokenizer. This is because the tokenizer was instantiated with a specific set of regions. If this is the case, the model simply returns the unknown token (`chrUNK-0:0`). If you find that this is happening often, you may want to ensure that your regions are a good fit for the universe of regions that the model was trained on. The unknown token will indeed have an embedding, but it will not be a meaningful representation of the region.

## Region pooling
It is often the case that we want a single embedding that represents a set of regions. For example, we may want to encode a patient by taking the average embedding of all the SNPs in the patient's genome. We can do this by simply averaging across the embeddings of the regions:

```python
patient_snps = "/path/to/bed/file.bed"

embeddings = model.encode(patient_snps) 
patient_embedding = np.mean(embeddings, axis=0)
```