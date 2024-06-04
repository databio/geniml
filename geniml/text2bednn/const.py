# metadata from csv
DEFAULT_GENOME_KEY = "sample_genome"
DEFAULT_SERIES_KEY = "gse"
DEFAULT_FILE_KEY = "file"
BIO_GPT_REPO = "microsoft/biogpt"
BIO_BERT_REPO = "dmis-lab/biobert-v1.1"


DEFAULT_TRAIN_P = 0.85 * 0.9
DEFAULT_VALIDATE_P = 0.85 * 0.1

# sentence transformer model from hugging face
DEFAULT_NL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_MAX_SEQ_LENGTH = 1000

DEFAULT_NUM_EPOCHS = 1000
DEFAULT_NUM_UNITS = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_OPTIMIZER_NAME = "Adam"
DEFAULT_LOSS_NAME = "cosine_embedding_loss"
DEFAULT_MARGIN = 0.0
# embedding dimension of Region2Vec: https://huggingface.co/databio/r2v-ChIP-atlas-hg38
DEFAULT_EMBEDDING_DIM = (100,)
# default learning rate of Adam optimizer
DEFAULT_LEARNING_RATE = 0.001

# if validation loss does not improve after patience*epoches, training stops
DEFAULT_PATIENCE = 0.2


DEFAULT_DATALOADER_SHUFFLE = True
MODULE_NAME = "text2bednn"
CONFIG_FILE_NAME = "config.yaml"
TORCH_MODEL_FILE_NAME_PATTERN = "v2v_{callback}_{checkpoint}.pt"
DEFAULT_MUST_TRAINED = True
DEFAULT_PLOT_FILE_NAME = "training_history"
DEFAULT_PLOT_TITLE = "Diagram of loss and epochs"
DEFAULT_HUGGINGFACE_MODEL_NAME = "checkpoint.pt"
