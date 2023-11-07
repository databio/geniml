from typing import Literal

MODULE_NAME = "region2vec"
LR_TYPES = Literal["constant", "exponential", "step"]
POOLING_TYPES = Literal["mean", "max"]
MAX_WAIT_TIME = 10800

DEFAULT_EPOCHS = 100
DEFAULT_GENSIM_EPOCHS = 1
DEFAULT_MIN_COUNT = 10
DEFAULT_N_SHUFFLES = 1  # 1 is sufficient for most cases
DEFAULT_WINDOW_SIZE = 5
DEFAULT_EMBEDDING_SIZE = 100
DEFAULT_EPOCHS = 10
DEFAULT_INIT_LR = 0.1  # https://github.com/databio/gitk/issues/6#issuecomment-1476273162
DEFAULT_MIN_LR = 0.0001  # gensim default

CONFIG_FILE_NAME = "config.yaml"
MODEL_FILE_NAME = "checkpoint.pt"
UNIVERSE_FILE_NAME = "universe.bed"
