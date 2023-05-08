# Evaluation of Genomic Region Embeddings

## Create a Base Embedding Object
Given a set of genomic region embeddings `embeddings` and the corresponding regions `vocab`, use `BaseEmbeddings` to create an `base` embedding object.
```
from gitk.eval.utils import BaseEmbeddings
import pickle
base_obj =  BaseEmbeddings(embeddings, vocab)
with open("base_embed.pt", "wb") as f:
    pickle.dump(base_obj, f)
```
## Genome Distance Scaling Test (GDST)
GDST evaluates how significant a set of genomic region embeddings preserves the biological knowledge that close regions on the genome tend to have similar biological functions and distant regions on the genome tend to have different biological functions. We assume that embedding distance reflects the function similarity between two genomic regions. The code uses a set of embeddings as input and outputs a GDS value showing how much the embedding distance scales the corresponding genome distance.

```
from gitk.eval.gdst import *
import numpy as np

# evaluate a single model
model_path = "/path/to/a/region2vec/model/"
gds = get_gds(model_path, embed_type="region2vec", num_samples=1000, seed=42)
print("Genome distance scaling: ", gds)
```
Or use the command line 
```
gitk eval gdst --model_path /path/to/a/region2vec/model/ --embed_type region2vec
```

Process a batch of two models (can be more than two models)
```
model_path1 = '/path/to/the/region2vec/model1/' 
model_path2 = '/path/to/the/region2vec/model2/' 
batch = [(model_path1, 'region2vec'), (model_path2, 'base')] # (model_path, embed_type)
gds_arr = get_gds_batch(batch, num_samples=1000, seed=42, num_workers=2) # set num_workers > 1 to enable multiprocessing
print("Model1: {}, GDS:{:.4f}".format(gds_arr[0][0],gds_arr[0][1]))
```

Run GDST 20 times for the two models
```
row_labels = ['model1-region2vec', 'model2-base']
gds_arr = gds_eval(batch, num_runs=20, num_samples=1000, num_workers=10)
print(gds_arr[0])
```


## Neighborhood Preserving Test (NPT)
Evaluate how significant genomic region embeddings preserve their neighboring regions on the genome against random embeddings. The code output the significance of neighborhood preserving ratio (SNPR) for a set of region embeddings. 

```
from gitk.eval.npt import *
model_path = '/path/to/a/region2vec/model/'
embed_type = 'region2vec'
K = 50
resolution = 10
result = get_snpr(
    model_path, embed_type, K, num_samples=1000, seed=0, resolution=resolution, num_workers=10
)
print(result["SNPR"]) # an array of SNPRs when the number of neighbors is 10, 20, 30, 40, 50, respectively
```

Or use the command line (the output will be the result when resolution=K)
```
gitk eval npt --model_path /path/to/a/region2vec/model/ --embed_type region2vec --K 50 --num_samples 1000
```

Process a batch of models
```
model_path1 = '/path/to/the/region2vec/model1/' 
model_path2 = '/path/to/the/region2vec/model2/' 
batch = [(model_path1, 'region2vec'), (model_path2, 'base')] # (model_path, embed_type)
result_list = get_snpr_batch(batch, K, num_samples=1000, seed=0)
print(result_list[0]["SNPR"][0]) # SNPRs for model1
print(result_list[1]["SNPR"][0]) # SNPRs for model2

# Run the genome distance test 20 times for the two models, setting save_folder will save the result for each run
K = 1000
resolution = 100 # increase the number of neighboring regions by resolution every time
npr_results = npt_eval(batch, K, num_samples=1000, num_workers=10, num_runs=20, resolution=resolution)
print("Model: {}".format(snpr_results[0][0]))
print(snpr_results[0][1].shape) # (20,10)
print(snpr_results[0][1][:,0]) # results from 20 runs when num_neighborhs=100
print(snpr_results[0][1][:,1]) # results from 20 runs when num_neighborhs=200
```

## Constrastive Clusters Test - Transcription Start Sites (CCT-TSS)
CCT-TSS evaluates how well a set of genomic region embeddings can separate regions relating to transcription start sites from those are not. This test involves clustering. 
Since we do not know a priori the true number of clusters for a set of region embeddings, we specify `K_arr` to test clusters in different sizes. The code gives the significance of contrastive clusters relating to transcription start sites (SCC-TSS) for a set of region embeddings.

The test requires running R scripts with the `GenomicDistributions`, `doParallel` and `optparse` packages.
```
from gitk.eval.cct import *

# process a single model (a set of genomic region embeddings)
model_path = '/path/to/a/region2vec/model/'
embed_type = 'region2vec'
save_folder = '/path/to/cst/results/'
Rscript_path = '/path/to/Rscript/'
assembly = 'hg19'
num_samples = 1000
K_arr = [5, 20, 40, 60, 100]
threshold = 0.0001 # significance threshold, the smaller the more significant
scores = get_scctss(model_path, embed_type, save_folder, Rscript_path, assembly, K_arr, num_samples, threshold)
print(scores) # scroes for each K in K_arr

# process a batch of two models
model_path1 = '/path/to/the/region2vec/model1/' 
model_path2 = '/path/to/the/region2vec/model2/' 
batch = [(model_path1, 'region2vec'), (model_path2, 'base')] # (model_path, embed_type)

# since we have more than one models, we can rank them based on the average scores over different Ks
scores_batch, avg_ranks = get_scctss_batch(batch, save_folder, Rscript_path, assembly, K_arr, num_samples, threshold)

# average ranks after running clustering_significance_test_batch num_runs times
avg_ranks_arr = cct_tss_eval(batch, save_folder, Rscript_path, assembly, K_arr, num_samples, threshold, num_runs=20)

```