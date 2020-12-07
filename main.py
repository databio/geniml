<<<<<<< HEAD

from singlecellEmbedding import singlecellEmbedding
import sys



if __name__ == "__main__":
   
    args = sys.argv
    singlecellEmbeddingmodel = singlecellEmbedding()
    
    path_file = args[1]
    nocells = int(args[2])
    noreads = int(args[3])
    dimension = int(args[4])
    min_count = int(args[5])
    shuffle_repeat = int(args[6])
    umap_nneighbours = int(args[7])
    window_size = int(args[8])
      
    model_filename = './word2vecmodels/word2vec_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}.model'.format(nocells, noreads, dimension, window_size , min_count, shuffle_repeat, umap_nneighbours)
    plot_filename = './figs/umapplot_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}.model.svg'.format(nocells, noreads, dimension, window_size , min_count, shuffle_repeat, umap_nneighbours)

    singlecellEmbeddingmodel.main(path_file, nocells, noreads, shuffle_repeat, window_size, dimension, min_count, umap_nneighbours, model_filename, plot_filename)
=======
from singlecellEmbedding import singlecellEmbedding

singlecellEmbeddingmodel = singlecellEmbedding()

# input file name
path_data = './data/'
file_name = 'GSE74310_scATACseq_All_Counts.txt'
path_file = path_data + file_name

# hyperparameters 
nocells = 5
noreads = 2
dimension = 10
min_count = 100
shuffle_repeat = 5
umap_nneighbours = 100
window_size = 100

#output file names
model_filename = './word2vecmodels/word2vec_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}.model'.format(nocells, noreads, dimension, window_size , min_count, shuffle_repeat, umap_nneighbours)
plot_filename = './figs/umapplot_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}.model.svg'.format(nocells, noreads, dimension, window_size , min_count, shuffle_repeat, umap_nneighbours)

singlecellEmbeddingmodel.main(path_file, nocells, noreads, shuffle_repeat, window_size, dimension, min_count, umap_nneighbours, model_filename, plot_filename)
>>>>>>> 2f6087fff37cb274e4275eea6ec764adbfa95026
