from time import time

from geniml.search.backends import BiVectorBackend, QdrantBackend
from geniml.search.interfaces import BiVectorSearchInterface


def test_bivec():
    # backend for text embeddings and bed embeddings
    text_backend = QdrantBackend(
        dim=384,
        collection="bed_text",
        qdrant_host="***REMOVED***",
        qdrant_api_key="***REMOVED***",
    )

    bed_backend = QdrantBackend(
        collection="bedbase2",
        qdrant_host="***REMOVED***",
        qdrant_api_key="***REMOVED***",
    )

    # the search backend

    search_backend = BiVectorBackend(text_backend, bed_backend)

    # the search interface
    search_interface = BiVectorSearchInterface(
        backend=search_backend, query2vec="sentence-transformers/all-MiniLM-L6-v2"
    )
    time1 = time()
    # actual search
    result = search_interface.query_search(
        query="lung",
        limit=10,
        with_payload=True,
        with_vectors=False,
        p=1.0,
        q=1.0,
        distance=False,  # QdrantBackend returns similarity as the score, not distance
    )
    assert result
    time2 = time()
    print(time2 - time1)
