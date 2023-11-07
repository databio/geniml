# How to fine-tune a Region2Vec model
## Overview
Fine-tuning a model is a way to adapt a pre-trained model to a new task. For example, we may want to fine-tune a model trained on ChIP-seq data to predict enhancers. This tutorial discusses how to fine-tune a pre-trained model. To learn how to train a new model see the [region2vec training documentation](./train-region2vec.md)

## Get a pretrained model
To begin, we need to get a pretrained model. We can get one from huggingface:
```python
from geniml.region2vec.experimental import Region2VecExModel

model = Region2VecExModel("databio/r2v-ChIP-atlas-v2")
```
This will download the model from huggingface and load it into memory. The model is now ready to be fine-tuned. First we need to create a new classifier using the pretrained model:
```python
import torch.nn as nn
import torch.functional as F

# enhancer classifier
class Region2VecClassifier(nn.Module):
    def __init__(self, region2vec_model: torch.nn.Module):
        super().__init__()
        self.region2vec = region2vec_model
        self.classification = nn.Sequential(
            nn.Linear(region2vec_model.embedding_dim, 1),
        )
        
    def forward(self, x: torch.Tensor):
        x = self.region2vec(x)  # Get the embeddings from Region2Vec
        x = F.relu(x)  # Pass through a non-linear activation function
        x = self.classification(x)  # Pass through additional layers
        return x
```
After insantiating the tokenizer, we can can use the model like so:
```python
from geniml.io import Region
from geniml.tokenization import ITTokenizer

r = Region("chr1", 100_000, 100_500) # some enhancer region (maybe)

tokenizer = ITTokenizer.from_pretrained("databio/r2v-ChIP-atlas-v2")

classifier = Region2VecClassifier(model.model) # get the inner core of the model
out = classifier(torch.tensor([42])) # pass in a region_id tensor

out.shape # torch.Size([1])
```

