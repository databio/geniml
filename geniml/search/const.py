from qdrant_client.http import models
from qdrant_client.models import Distance

DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333

DEFAULT_COLLECTION_NAME = "embeddings"

DEFAULT_QDRANT_DIST = Distance.COSINE

DEFAULT_INDEX_PATH = "./current_index.bin"

DEFAULT_HNSW_SPACE = "cosine"

DEFAULT_DIM = 100


# the size of the dynamic list for the nearest neighbors
# Higher ef leads to more accurate but slower search
# cannot be set lower than the number of queried nearest neighbors k
DEFAULT_EF = 200

# the number of bi-directional links created for every new element during construction
# Higher M work better on datasets with high intrinsic dimensionality and/or high recall
# low M work better for datasets with low intrinsic dimensionality and/or low recalls.
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
