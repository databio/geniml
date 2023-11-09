import torch

MODULE_NAME = "geniml.classification"

VOCAB_SIZE_KEY = "vocab_size"
NUM_CLASSES_KEY = "num_classes"
EMBEDDING_DIM_KEY = "embedding_dim"

MODEL_FILE_NAME = "checkpoint.pt"
UNIVERSE_FILE_NAME = "universe.bed"
CONFIG_FILE_NAME = "config.yaml"

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = torch.optim.Adam
DEFAULT_LOSS_FN = torch.nn.CrossEntropyLoss
DEFAULT_LABEL_KEY = "cell_type"
DEFAULT_TEST_TRAIN_SPLIT = 0.8
