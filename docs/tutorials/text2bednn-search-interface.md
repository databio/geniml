# How to create a natural language search backend for BED files
The metadata of each BED file / region set is needed to build a natural language search backend. Embedding vectors of BED
files are created by `Region2Vec`, and embedding vectors of metadata are created by [`SentenceTransformers`](https://www.sbert.net/). `Embed2EmbedNN`,
a feedforward neural network (FNN), is trained to learn the embedding vectors of metadata from the embedding vectors of BED
files. When a natural language query string is given, it will first be encoded to a vector by `SentenceTransformers`, and that
vector will be encoded to a query vector by the FNN. `search` backend can perform k-nearest neighbors (KNN) search among the
stored embedding vectors of BED files, and the BED files whose embedding vectors are closest to that query vector are the
search results.

## Upload metadata and BED files
`RegionSetInfo` is a [`dataclass`](https://docs.python.org/3/library/dataclasses.html) that can store information about a BED file, which includes the file name, metadata, and the
embedding vectors of region set and metadata. A list of RegionSetInfo can be created with a folder of BED files and a file of their
metadata by `SentenceTransformers` and `Region2VecExModel`. The first column of metadata file must match the BED file names
(the first column contains BED file names, or strings which BED file names start with), and is sorted by the first column. It can be
sorted by a terminal command:
```
sort -k1 1 metadata_file >  new_metadata_file
```
Example code to build a list of RegionSetInfo
```python
from gitk.text2bednn.utils import build_regionset_info_list
from gitk.region2vec.main import Region2VecExModel
from sentence_transformers import SentenceTransformer

# load Region2Vec from hugging face
r2v_model = Region2VecExModel("databio/r2v-ChIP-atlas")
# load SentenceTransformers
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# folder of bed file
bed_folder = "path/to/folders/of/bed/files"
# path for metadata file
metadata_path = "path/to/file/of/metadata"

# list of RegionSetInfo
ri_list = build_regionset_info_list(bed_folder, metadata_path, r2v_model, st_model)
```

## Train the Embed2EmbedNN
The list of RegionSetInfo can be split into 3 lists, which represent the training set, validating set, and testing set. The embedding
vectors of metadata will be X, and the embedding vectors of the region set will be Y.
```python
from gitk.text2bednn.utils import data_split, region_info_list_to_vectors
from gitk.text2bednn.text2bednn import Vec2VecFNN

# split the list of RegionInfoSet into different data set
train_list, validate_list, test_list = data_split(ri_list)

# get the embedding vectors
train_X, train_Y = region_info_list_to_vectors(train_list)
validate_X, validate_Y = region_info_list_to_vectors(validate_list)

# train the neural network
e2enn = Vec2VecFNN()
e2enn.train(train_X, train_Y, validate_X, validate_Y)
```

## Load the vectors and information to search backend
[`qdrant-client`](https://github.com/qdrant/qdrant-client) and [`hnswlib`](https://github.com/nmslib/hnswlib) can store vectors and perform k-nearest neighbors (KNN) search with a given query vector, so we
created one database backend (`QdrantBackend`) and one local file backend (`HNSWBackend`) that can store the embedding
vectors for KNN search. `HNSWBackend` will create a .bin file with given path, which saves the searching index.
```python
from gitk.text2bednn.utils import prepare_vectors_for_database

# loading data to search backend
embeddings, labels = prepare_vectors_for_database(ri_list)

# search backend
hnsw_backend = HNSWBackend(local_index_path="path/to/local/index.bin")
hnsw_backend.load(embeddings, labels)
```

## text2bednn search interface
The `TextToBedNNSearchInterface` includes model that encode natural language to vectors (default: `SentenceTransformers`), a
model that encode natural language embedding vectors to BED file embedding vectors (`Embed2EmbedNN`), and a `search` backend.

```python
from gitk.text2bednn.text2bednn import Text2BEDSearchInterface

# initiate the search interface
file_interface = Text2BEDSearchInterface(st_model, e2enn, hnsw_backend)

# natural language query string
query_term = "human, kidney, blood"
# perform KNN search with K = 5, the id of stored vectors and the distance / similarity score will be returned
ids, scores = file_interface.nl_vec_search(query_term, 5)
```
