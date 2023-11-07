# How to fine-tune a Region2Vec model (Very experimental)
## Overview
Fine-tuning a model is a way to adapt a pre-trained model to a new task. For example, we may want to fine-tune a model trained using unsupervised learning and ChIP-seq data to predict enhancers. This tutorial discusses how to fine-tune a pre-trained model. To learn how to train a new model see the [region2vec training documentation](./train-region2vec.md).

## Get a pretrained model
To begin, we need to get a pretrained model. We can get one from huggingface:
```python
from geniml.region2vec import Region2VecExModel

model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38-v2")
```
This will download the model from huggingface and load it into memory. The model is now ready to be fine-tuned. First we need to create a new classifier using the pretrained model:
```python
import torch
import torch.nn as nn
import torch.functional as F

# enhancer classifier
class EnhancerClassifier(nn.Module):
    def __init__(self, region2vec_model: torch.nn.Module):
        super().__init__()
        self.region2vec = region2vec_model
        self.classification = nn.Sequential(
            nn.Linear(region2vec_model.embedding_dim, 1),
        )
        
    def forward(self, x: torch.Tensor):
        x = self.region2vec(x)  # Get the embeddings from Region2Vec
        x = x.mean(dim=0)  # Average the embeddings (if multiple regions are passed in, this can occur due to tokenization)
        x = nn.ReLU()(x)  # Pass through a non-linearity
        x = self.classification(x)  # Pass through additional layers
        return x
```
After instantiating the tokenizer, we can can use the model like so:
```python
from geniml.io import Region
from geniml.tokenization import ITTokenizer

r = Region("chr1", 1_000_000, 1_000_500) # some enhancer region (maybe)

tokenizer = ITTokenizer.from_pretrained("databio/r2v-ChIP-atlas-hg38-v2")
classifier = EnhancerClassifier(model.model) # get the inner core of the model

x = tokenizer.tokenize(r)
x = torch.tensor([t.id for t in x], dtype=torch.long)
out = classifier(x)

out.shape # torch.Size([1])

# apply sigmoid
out = torch.sigmoid(out)

print("Enhancer probability:", round(out.item(), 3))
```

## Saving the fine-tuned embeddings
`torch`'s computational graph links the original region2vec model back to the `Region2VecExModel`. Therefore, if we want to save the fine-tuned embeddings, we simply ned to call `export` on the original model:
```python
model.export("my-fine-tuned-model")
```
