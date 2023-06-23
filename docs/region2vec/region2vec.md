# Region2Vec
`Region2Vec` will generate embedding vectors for a given region set (universe) from a set of raw bed files. The program will first map all raw regions to the given region set. Then, it will concatenate all regions in a bed file in random orders into a sentence. The generated sentences will be used for Region2Vec training.


## Usage
1. Prepare a set of bed files in `src_folder`. [Optional] If only a subset of files will be used, specify a list of those files as `file_list`. By default, the program will use all the files in the folder to train a Region2Vec model.
2. Prepare a universe file `universe_file`.
3. Create a token folder which will be used to store tokenized files `dst_folder`.
5. Run the following command
``` 
from gitk.tokenization import hard_tokenization
from gitk.region2vec import region2vec

src_folder = '/path/to/raw/bed/files'
dst_folder = '/path/to/tokenized_files'
universe_file = '/path/to/universe_file'

# must run tokenization first
status = hard_tokenization(src_folder, dst_folder, universe_file, 1e-9)

if status: # if hard_tokenization is successful, then run Region2Vec training
    save_dir = '/path/to/training/results'
    region2vec(dst_folder, save_dir, num_shufflings=1000)

```
For customized settings, please go and check the parameters used in `main.py`. 
For training a Region2Vec model, the parameters, `init_lr`, `window_size`, `num_shufflings`, `embedding_dim`, are frequently tuned in experiments.

For command line usage, type `gitk region2vec --help` for details. We give a simple usage below
```bash
gitk region2vec --token-folder /path/to/token/folder --save-dir ./region2vec_model --num-shuffle 10 --embed-dim 100 --context-len 50 
```


