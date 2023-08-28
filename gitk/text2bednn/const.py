DEFAULT_TRAIN_P = 0.85*0.9
DEFAULT_VALIDATE_P = 0.85*0.1

#sentence transformer model from hugging face
DEFAULT_HF_ST_MODEL: str = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_NUM_UNITS = 256
DEFAULT_NUM_EXTRA_HIDDEN_LAYERS = 0
DEFAULT_BATCH_SIZE = 1
DEFAULT_OPTIMIZER_NAME = "adam"
DEFAULT_LOSS_NAME = "cosine_similarity"