from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333

DEFAULT_COLLECTION_NAME = "embeddings"

DEFAULT_QDRANT_CONFIG = VectorParams(size=100, distance=Distance.COSINE)

DEFAULT_INDEX_PATH = "./current_index.bin"

DEFAULT_HNSW_SPACE = "cosine"

DEFAULT_DIM = 100

DEFAULT_EF = 200

DEFAULT_M = 64

DEFAULT_QUANTIZATION_CONFIG = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    ),
)


# for evaluation dataset from huggingface
HF_INDEX = "index.bin"
HF_PAYLOADS = "payloads.pkl"
HF_METADATA = "metadata.json"
