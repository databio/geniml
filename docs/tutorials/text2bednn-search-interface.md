# How to create a natural language search backend for BED files
The metadata of each BED file is needed to build a natural language search backend. BED files embedding vectors are created by
`Region2Vec` model, and metadata embedding vectors are created by [`FastEmbed`](https://github.com/qdrant/fastembed), [`SentenceTransformers`](https://www.sbert.net/), or other text embedding models.

`Vec2VecFNN`, a feedforward neural network (FNN), is trained to maps vectors from the embedding space of natural language to the embedding
space of BED files. When a natural language query string is given, it will first be encoded to a vector by the text embedding model, and then created 
vector will be encoded to a query vector by the FNN. `search` backend can perform k-nearest neighbors (KNN) search among the stored BED
file embedding vectors, and the BED files whose embedding vectors are closest to that query vector are the search results.

## Search distance metrics

The default distance metrics for KNN search in `geniml` is [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). Which is bounded in [0,1]. The smaller the value is, the higher similarity between the query vector and returned search results. [HNSWBackend](https://github.com/nmslib/hnswlib?tab=readme-ov-file#python-bindings) and [QdrantBackend](https://qdrant.tech/documentation/concepts/search/#metrics) also have other options of distance metrics.

## Store embedding vectors
It is recommended to use `geniml.search.backend.HNSWBackend` to store embedding vectors. In the `HNSWBackend` that stores each BED file embedding
vector, the `payload` should contain the name or identifier of BED file. In the `HNSWBackend` that stores the embedding vectors of each 
metadata string, the `payload` should contain the original string text and the names of BED files that have that string in metadata.

## Train the model
Training a `Vec2VecFNN` needs x-y pairs of vectors (x: metadata embedding vector; y: BED embedding vector). A pair of a metadata embedding
vector with the embedding vectors of BED files in its payload is a target pair, otherwise a non-target pair. Non-target pairs are sampled for
contrastive loss. Here is sample code to generate pairs from storage backend and train the model:

```python
# target is an array of 1 (target) and -1 (non-target) 
X, Y, target = vec_pairs(
    nl_backend,  # HNSWBackend that store metadata embedding vectors
    bed_backend,  # HNSWBackend that store BED embedding vectors
    "name",  # key to file name in BED backend payloads
    "files",  # key to matching files in metadata backend payloads
    True,  # sample non-target pairs
    1.0  # number of non-target pairs /number of target pairs = 1
)

# train without validate data
v2v_torch_contrast.train(
    X,
    Y,
    folder_path="path/to/folder/for/checkpoint",
    loss_func="cosine_embedding_loss",  # right now "cosine_embedding_loss" is the only contrastive loss function available
    training_target=target,
)

```

## Search interface
A search interface consists of a storage backend where vectors are stored, and a module (`geniml.search.query2vec`) that embed the query.
`geniml.search` supports two types of queries: region set query and text query. 

### Region set query

`BED2Vec` embed the query region set with a `Region2VecExModel`, and the embedding vector is used to perform KNN search within the backend.

```python
from geniml.search import BED2BEDSearchInterface, BED2Vec

# init BED2Vec with a hugging face repo of a Region2VecExModel
bed2vec = BED2Vec("databio/r2v-ChIP-atlas-hg38-v2")

# the search_backend can be QdrantBackend or HNSWBackend
search_interface = BED2BEDSearchInterface(search_backend, bed2vec)

# the query cam be a RegionSet object (see geniml.io) or path to a BED file in disk
file_search_result = search_interface.query_search("path/to/a/bed/file.bed", 5)
```

### Text query

`Text2Vec` embed the query string with a with a natural language embedding model first (default: `FlagEmbedding`), and then maps the text embedding vector into the embedding space of region sets through a trained `Vec2VecFNN`.

```
from geniml.search import Text2BEDSearchInterface, Text2Vec

text2vec = Text2Vec(
    "sentence-transformers/all-MiniLM-L6-v2",  # either a hugging face repo or an object from geniml.text2bednn.embedder
    "databio/v2v-geo-hg38"  # either a hugging face repo or a Vec2VecFNN
)

search_interface = Text2BEDSearchInterface(search_backend, text2vec)
text_search_result = search_interface.query_search("cancer cells", 5)
```

With a dictionary that contains query strings and id of relevant query results in search backend in this format:

```
{
    <query string>: [
        <id of relevant result in backend>,
        ...    
    ],
    ...
}
```

`Text2BEDSearchInterface` can return [mean average precision](https://www.youtube.com/watch?v=pM6DJ0ZZee0&t=157s), [average AUC-ROC](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf), and [average R-Precision](https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_491), here is example code:

```python
query_dict = {
    "metadata string 1": [2, 3],
    "metadata string 12": [1],
    "metadata string 3": [2, 4, 5],
    "metadata string 1": [0]
}

MAP, AUC, RP = search_interface.eval(query_dict)
```

## Hugging Face

### Model

`Vec2VecFNN` can be innitiated from a Hugging Face repository:

```python
model = Vec2VecFNN("databio/v2v-sentencetransformers-encode")
```

To upload the model onto huggingface, you can use `export` function to download the files of model(checkpoint.pt) and config(config.yaml).

```
v2v_torch1.export("path/totarget/folder", "checkpoint.pt")
```

Then upload both files with correct names onto the Hugging Face repository

### Dataset

`geniml.search.anecdotal_search_from_hf_data` can allow users to query with free-form natural language strings to search in a Hugging Face dataset. The dataset must have:

* hnsw index file of BED file embeddings (index.bin)
* dictionary file of payloads (payloads.pkl). It must have file name stored with the key of "file". For example:

```
# key of the payload is the storage index in the hnsw index
{
    0: {
        "file": "Example.bed",
        ...
    },
    ...
}
```

* metadata file (metadata.json) in this format: `{<metadata attribute>: {<annotation text>: [<files>]}}`. For example:

```
{
    "tissue": {
        "kidney": [
            "Example.bed",
            ...
        ],
        ...
    },
    ...
}
```

With the repo name of dataset, `Vec2VecFNN`, and repo name of the model that was used to encode training metadata, you can search through the dataset with any free-form query you type:

```python
from geniml.search import anecdotal_search_from_hf_data
import pprint

# vec2vec model
search_repo = "databio/v2v-sentencetransformers-encode"
# text encoder model
text_repo = "sentence-transformers/all-MiniLM-L6-v2"
# dataset
data_repo = "databio/geo-hg38-search-test"
result = anecdotal_search_from_hf_data(
    "glioblastoma",
    data_repo,
    search_repo,
    text_repo,
    10
)

pprint.pprint(result)

```
