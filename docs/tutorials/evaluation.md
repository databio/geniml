# How to evaluate genomic region embeddings

## Preparation

### Create a Base Embedding Object

Given a set of genomic region embeddings `embeddings` and the corresponding regions `vocab`, use `BaseEmbeddings` to create an `base` embedding object.
```
from gitk.eval.utils import BaseEmbeddings
import pickle
base_obj =  BaseEmbeddings(embeddings, vocab)
with open("base_embed.pt", "wb") as f:
    pickle.dump(base_obj, f)
```

### Generate Binary Embeddings

```python
from gitk.eval.utils import get_bin_embeddings
universe_file = "/path/to/universe.bed"
token_files = ["file1.bed", "file2.bed"]
bin_embed = get_bin_embeddings(universe_file, token_files)
```

Or use command line:

```bash
gitk eval bin-gen --universe /path/to/universe.bed --token-folder /path/to/tokenized/folder --file-name bin_embed.pickle
```

## Statistical Tests

### Cluster Tendency Test (CTT)

CTT analyzes how well a set of region embeddings can be clustered.  CTT score lies between 0 and 1. A larger CTT score indicates a greater tendency for the embeddings being evaluated to have clusters. When the embeddings are uniformly distributed, the score is 0.5. For evenly spaced embeddings, the score approaches 0.

```python
from gitk.eval.ctt import get_ctt_score, ctt_eval
path = "/path/to/a/region2vec/model/"
embed_type = "region2vec"
ctt_score = get_ctt_score(path, embed_type, seed=42, num_data=10000, num_workers=10)
print(ctt_score)

# evaluate a batch of models and run CTT for 5 times with different random seeds
batch = [(path, embed_type)]
ctt_score_arr = ctt_eval(batch, num_runs=5, num_data=10000,num_workers=10)
print(f"Model: {ctt_score_arr[0][0]}\n CTT scores:{ctt_score_arr[0][1]}") # CTT scores for the 1st model in the batch
```

Or use the command line
```bash
gitk eval ctt --model-path /path/to/a/region2vec/model/ --embed-type region2vec
```
### Reconstruction Test (RCT)
RCT evaluates how well an embedding of a region preserves the region’s occurrence information in the training data. The best RCT score is 1.

```python
from gitk.eval.rct import get_rct_score, rct_eval
path = "/path/to/a/region2vec/model/"
embed_type = "region2vec"
bin_path = "/path/to/a/binary/embedding/for/the/same/tokenized/files/"
# set out_dim to -1 use all the dimensions of the binary embeddings. Set out_dim to a small positive number to reduce computational complexity.
rct_score = get_rct_score(path, embed_type, bin_path, out_dim=-1, cv_num=5, seed=42, num_workers=10)
print(rct_score)

# evaluate a batch of models and run RCT for 5 times with different random seeds
batch = [(path, embed_type, bin_path)]
rct_score_arr = rct_eval(batch, num_runs=5, cv_num=5, out_dim=-1, num_workers=10) 
print(f"Model: {rct_score_arr[0][0]}\n RCT scores:{rct_score_arr[0][1]}") # RCT scores for the 1st model in the batch
```

Or use the command line 
```bash
gitk eval rct --model-path /path/to/a/region2vec/model/ --embed-type region2vec
```
To change the learning setting, go to the definition of `get_rct_score` in `gitk/eval/rct.py` and change the constructor of `MLPRegressor`.


## Biological Tests

### Genome Distance Scaling Test (GDST)

GDST calculates a score measuring how much the embedding distance between two regions scales the corresponding genome distance.

```python
from gitk.eval.gdst import get_gdst_score, gdst_eval
path = "/path/to/a/region2vec/model/"
embed_type = "region2vec"
gdst_score = get_gdst_score(path, embed_type, num_samples=10000,seed=42)
print(gdst_score)

# evaluate a batch of models and run GDST for 5 times with different random seeds
batch = [(path,embed_type)] 
gdst_score_arr = gdst_eval(batch, num_runs=5, num_samples=10000)
```

Or use the command line 
```bash
gitk eval gdst --model-path /path/to/a/region2vec/model/ --embed-type region2vec
```

### Neighborhood Preserving Test (NPT)

NPT evaluates how significant genomic region embeddings preserve their neighboring regions on the genome against random embeddings. The code output the NPT score for a set of region embeddings. 

```python
from gitk.eval.npt import get_npt_score, npt_eval
path = "/path/to/a/region2vec/model/"
embed_type = "region2vec"
K = 10
# If resolution = K gives NPT for K neighbors
# If resolution < K, gives NPT for [resolution, resolution*2, ...] neighbors
resolution = K 
npt_score = get_npt_score(path, embed_type, K, num_samples=100, seed=0, resolution=resolution,num_workers=10)
print(npt_score['SNPR'])

# evaluate a batch of models and run NPT for 5 times with different random seeds
batch = [(path, embed_type)]
npt_score_arr = npt_eval(batch, K, num_samples=100, num_workers=10, num_runs=5, resolution=resolution)
print(f"Model: {npt_score_arr[0][0]}\n NPT scores: {npt_score_arr[0][1]}") # NPT scores for the 1st model in the batch
```

Or use the command line (the output will be the result when resolution=K)
```bash
gitk eval npt --model-path /path/to/a/region2vec/model/ --embed-type region2vec --K 50 --num-samples 1000
```
