from qdrant_client.models import VectorParams, Distance

DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333

DEFAULT_COLLECTION = "embeddings"

DEFAULT_CONFIG = VectorParams(size=100, distance=Distance.COSINE)

DEFAULT_INDEX_PATH = "./current_index.bin"

DEFAULT_HNSW_SPACE = "cosine"

DEFAULT_DIM = 100

DEFAULT_EF = 200

DEFAULT_M = 16
