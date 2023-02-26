# Evaluation of Genomic Region Embeddings

## Genome Distance Test
Evaluate how well genomic region embeddings preserve the structure (relative closeness) of genomic regions on the genome.

```
from gitk.eval.genome_distance_test import *
import numpy as np

model_path = '/path/to/a/region2vec/model/'
boundaries = np.linspace(1e3, 1e8, 5) # four bins
result = genome_distance_test(model_path, 'region2vec', boundaries, num_samples=1000, seed=0)

avgGD = result['AvgGD']
avgED = result['AvgED']
gdt_plot_fitted(avgGD, avgED, 'result.png')


# process a batch of two models
model_path1 = '/path/to/the/region2vec/model1/' 
model_path2 = '/path/to/the/region2vec/model2/' 
batch = [(model_path1, 'region2vec'), (model_path2, 'base')] # (model_path, embed_type)
result_list = genome_distance_test_batch(batch, boundaries, num_samples=1000, seed=0)

slope1 = result_list[0]['Slope']
error1 = result_list[0]['AvgED']
AvgGD1 = result_list[0]['AvgGD']
AvgED1 = result_list[0]['AvgED']
model_path1 = result_list[0]['Path']

slope2 = result_list[1]['Slope']
error2 = result_list[1]['Error']
AvgGD2 = result_list[1]['AvgGD']
AvgED2 = result_list[1]['AvgED']
model_path2 = result_list[1]['Path']

# Run the genome distance test 20 times for the two models
row_labels = ['model1-region2vec', 'model2-base']
slope_list, approx_err_list = gdt_eval(batch, boundaries, num_runs=20, num_samples=1000, save_folder=None)

# plot the genome distance test figure
gdt_box_plot(slope_list, approx_err_list, row_labels, filename='gdt_result.png')
```

## Neighborhood Preserving Test
Evaluate how significant genomic region embeddings preserve their neighboring regions on the genome against random embeddings.

```
from gitk.eval.neighborhood_preserving_test import *
model_path = '/path/to/a/region2vec/model/'
embed_type = 'region2vec'
K = 50
result = neighborhood_preserving_test(model_path, embed_type, K, num_samples=1000, seed=0)
print(result['SNPR'][0])

# process a batch of two models
model_path1 = '/path/to/the/region2vec/model1/' 
model_path2 = '/path/to/the/region2vec/model2/' 
batch = [(model_path1, 'region2vec'), (model_path2, 'base')] # (model_path, embed_type)
result_list = neighborhood_preserving_test_batch(batch, K, num_samples=1000, seed=0)
print(result_list[0]['SNPR'][0]) # SNPR for model1
print(result_list[1]['SNPR'][0]) # SNPR for model2

# Run the genome distance test 20 times for the two models, setting save_folder will save the result for each run
snpr_results = npt_eval(batch, K, num_samples=1000, num_runs=20, save_folder=None)

# plot the neighborhood preserving test figure
row_labels = ['model1-region2vec', 'model2-base']
snpr_plot(snpr_results, row_labels, filename='snpr_result.png')
```

## Clustering Significance Test
Evaluate how well the genomic region embeddings can form biologically meaningful clusters. The metric we use for a set of genomic region embeddings reflects how well these embeddings can separate clusters that are related to transcription start sites from those are not related to transcription start sites.
Since we do not know a priori the true number of clusters for a set of region embeddings, we specify `K_arr` to include several possible numbers of clusters.

The functions require running R scripts with the `GenomicDistributions` and `optparse` packages.
```
from gitk.eval.clustering_significance_test import *

# process a single model (a set of genomic region embeddings)
model_path = '/path/to/a/region2vec/model/'
embed_type = 'region2vec'
save_folder = '/path/to/cst/results/'
Rscript_path = '/path/to/Rscript/'
assembly = 'hg19'
num_samples = 1000
K_arr = [5, 20, 40, 60, 100]
threshold = 0.0001 # significance threshold
scores = clustering_significance_test(model_path, embed_type, save_folder, Rscript_path, assembly, K_arr, num_samples, threshold)

# process a batch of two models
model_path1 = '/path/to/the/region2vec/model1/' 
model_path2 = '/path/to/the/region2vec/model2/' 
batch = [(model_path1, 'region2vec'), (model_path2, 'base')] # (model_path, embed_type)

# since we have more than one models, we can rank them based on the average scores over different Ks
scores_batch, avg_ranks = clustering_significance_test_batch(batch, save_folder, Rscript_path, assembly, K_arr, num_samples, threshold)

# average ranks after running clustering_significance_test_batch num_runs times
avg_ranks_arr = cst_eval(batch, K, save_folder, Rscript_path, assembly, K_arr, num_samples, threshold, num_runs=20)

# plot the average ranks for the two models
row_labels = ['model1-region2vec', 'model2-base']
cst_plot(avg_ranks_arr, row_labels, filename='cst_result.png')
```