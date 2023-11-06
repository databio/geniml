from typing import Literal

import torch

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
DEFAULT_DECAY_RATE = 0.95
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = torch.optim.SGD
DEFAULT_LOSS_FN = torch.nn.CrossEntropyLoss
DEFAULT_PADDING_CHR = "chrPAD"
DEFAULT_PADDING_START = 0
DEFAULT_PADDING_END = 0
DEFAULT_NS_POWER = 0.75  # from Mikolov et al. 2013
DEFAULT_NS_K = 5  # from Mikolov et al. 2013

CONFIG_FILE_NAME = "config.yaml"
MODEL_FILE_NAME = "checkpoint.pt"
UNIVERSE_FILE_NAME = "universe.bed"
