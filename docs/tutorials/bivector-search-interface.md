# Region set (BED) search combining both metadata and genomic regions

## Metadata embedding vector backend

This vector beckend stored the embedding vectors of region set metadata annotations, which are encoded by  open-source text model ([`SentenceTransformers`](https://www.sbert.net/), etc.). The payload of each metadata annotation vector must contain the storage ids of region set that the annotation matches.

## Example code

```python
from geniml.search.backends import BiVectorBackend, QdrantBackend
from geniml.search.interfaces import BiVectorSearchInterface

# 2 required backends
text_backend = QdrantBackend(dim=384)
bed_backend = QdrantBackend()

# load vectors and payloads
bed_backend.load(vectors=np.array(bed_vecs), payloads=bed_payloads)
text_backend.load(vectors=np.array(text_embeddings), payloads=text_payloads)

# the search backend
search_backend = BiVectorBackend(text_backend, bed_backend)


# the search interface
search_interface = BiVectorSearchInterface(
    backend=search_backend, query2vec="sentence-transformers/all-MiniLM-L6-v2"
)


# actual search
result = search_interface.query_search(
    query="lung cancer cell lines",
    limit=10,
    with_payload=True,
    with_vectors=False,
    p=1.0,
    q=1.0,
    distance = False # QdrantBackend returns similarity as the score, not distance
)
```
