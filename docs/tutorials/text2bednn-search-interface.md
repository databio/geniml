# How to create a natural language search backend for BED files
The metadata of each BED file / region set is needed to build a natural language search backend. Embedding vectors of BED
files are created by `Region2Vec`, and embedding vectors of metadata are created by [`SentenceTransformers`](https://www.sbert.net/). `Embed2EmbedNN`,
a feedforward neural network (FNN), is trained to learn the embedding vectors of metadata from the embedding vectors of BED
files. When a natural language query string is given, it will first be encoded to a vector by `SentenceTransformers`, and that
vector will be encoded to a query vector by the FNN. `search` backend can perform k-nearest neighbors (KNN) search among the
stored embedding vectors of BED files, and the BED files whose embedding vectors are closest to that query vector are the
search results.

## Upload metadata and regions from files
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
from geniml.text2bednn.utils import build_regionset_info_list_from_files
from geniml.region2vec.main import Region2VecExModel
from fastembed.embedding import FlagEmbedding

# load Region2Vec from hugging face
r2v_model = Region2VecExModel("databio/r2v-ChIP-atlas")
# load natural language embedding model
nl_model = FlagEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# folder of bed file
bed_folder = "path/to/folders/of/bed/files"
# path for metadata file
metadata_path = "path/to/file/of/metadata"

# list of RegionSetInfo
ri_list = build_regionset_info_list_from_files(bed_folder, metadata_path, r2v_model, nl_model)
```

## Upload metadata and regions from PEP
A list of RegionSetInfo can also be created with a [`PEP`](https://pep.databio.org/en/latest/), which includes a `.csv` that stores metadata, and a `.yaml` as a a metadata validation
framework.

Example code to build a list of RegionSetInfo from a PEP:

```python
from geniml.text2bednn.utils import build_regionset_info_list_from_PEP

# columns in the csv of PEP that contains metadata information
columns = return [
        "tissue",
        "cell_line",
        "tissue_lineage",
        "tissue_description",
        "diagnosis",
        "sample_name",
        "antibody",
    ]

# path to the yaml file
yaml_path = "path/to/framework/yaml/file"

ri_list_PEP = build_regionset_info_list_from_PEP(
        yaml_path,
        col_names,
        r2v_model,
        nl_model,
    )
```

## Train the model
The list of RegionSetInfo can be split into 3 lists, which represent the training set, validating set, and testing set. The embedding
vectors of metadata will be X, and the embedding vectors of the region set will be Y.

```python
from sklearn.model_selection import train_test_split
from geniml.text2bednn.utils region_info_list_to_vectors
from geniml.text2bednn.text2bednn import Vec2VecFNN

# split the list of RegionInfoSet into different data set
train_list, validate_list = train_test_split(ri_list, test_size=0.2)

# get the embedding vectors
train_X, train_Y = region_info_list_to_vectors(train_list)
validate_X, validate_Y = region_info_list_to_vectors(validate_list)

# train the neural network
v2vnn = Vec2VecFNN()
v2vnn.train(train_X, train_Y, validating_data=(validate_X, validate_Y), num_epochs=50)
```

## Load the vectors and information to search backend
[`qdrant-client`](https://github.com/qdrant/qdrant-client) and [`hnswlib`](https://github.com/nmslib/hnswlib) can store vectors and perform k-nearest neighbors (KNN) search with a given query vector, so we
created one database backend (`QdrantBackend`) and one local file backend (`HNSWBackend`) that can store the embedding
vectors for KNN search. `HNSWBackend` will create a .bin file with given path, which saves the searching index.

```python
from geniml.text2bednn.utils import prepare_vectors_for_database

# loading data to search backend
embeddings, labels = prepare_vectors_for_database(ri_list)

# search backend
hnsw_backend = HNSWBackend(local_index_path="path/to/local/index.bin")
hnsw_backend.load(embeddings, labels)
```

## text2bednn search interface
The `TextToBedNNSearchInterface` includes model that encode natural language to vectors (default: `FlagEmbedding`), a
model that encode natural language embedding vectors to BED file embedding vectors (`Embed2EmbedNN`), and a `search` backend.

```python
from geniml.text2bednn.text2bednn import Text2BEDSearchInterface

# initiate the search interface
file_interface = Text2BEDSearchInterface(nl_model, e2enn, hnsw_backend)

# natural language query string
query_term = "human, kidney, blood"
# perform KNN search with K = 5, the id of stored vectors and the distance / similarity score will be returned
ids, scores = file_interface.nl_vec_search(query_term, 5)
```
