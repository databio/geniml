DEFAULT_TRAIN_P = 0.85 * 0.9
DEFAULT_VALIDATE_P = 0.85 * 0.1

# sentence transformer model from hugging face
DEFAULT_HF_ST_MODEL: str = "sentence-transformers/all-MiniLM-L12-v2"

DEFAULT_NUM_EPOCHS = 1000
DEFAULT_NUM_UNITS = 256
DEFAULT_NUM_EXTRA_HIDDEN_LAYERS = 0
DEFAULT_BATCH_SIZE = 1
DEFAULT_OPTIMIZER_NAME = "Adam"
DEFAULT_LOSS_NAME = "mean_squared_error"
# embedding dimension of Region2Vec: https://huggingface.co/databio/r2v-ChIP-atlas-hg38
DEFAULT_EMBEDDING_DIM = (100,)
# default learning rate of Adam optimizer
DEFAULT_LEARNING_RATE = 0.001

# if validation loss does not improve after patience*epoches, training stops
DEFAULT_PATIENCE = 0.2

DEFAULT_VEC2VEC_MODEL_FILE_NAME = "vec2vec.h5"
DEFAULT_PAYLOAD_KEY = "payload"
DEFAULT_VECTOR_KEY = "vector"
DEFAULT_METADATA_KEY = "metadata"

MODULE_NAME = "text2bednn"
CONFIG_FILE_NAME = "config.yaml"
TORCH_MODEL_FILE_NAME_PATTERN = "v2c2v2c_{callback}_{checkpoint}.pt"
DEFAULT_MUST_TRAINED = True
DEFAULT_PLOT_FILE_NAME = "training_history"
DEFAULT_PLOT_TITLE = "Diagram of loss and epochs"
