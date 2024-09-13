from typing import Literal

PAD_CHR = "chrPAD"
PAD_START = 0
PAD_END = 0

MASK_CHR = "chrMASK"
MASK_START = 0
MASK_END = 0

DEFAULT_EMBEDDING_DIM = 768
DEFAULT_CONTEXT_SIZE = 2048

MASK_RATE = 0.15
REPLACE_WITH_MASK_RATE = 0.8
REPLACE_WITH_RANDOM_RATE = 0.1
KEEP_RATE = 0.1

CONFIG_FILE_NAME = "config.yaml"
MODEL_FILE_NAME = "checkpoint.pt"
UNIVERSE_FILE_NAME = "universe.toml"

POOLING_TYPES = Literal["mean", "max"]
POOLING_METHOD_KEY = "pooling_method"
D_MODEL_KEY = "embedding_dim"
VOCAB_SIZE_KEY = "vocab_size"
N_HEADS_KEY = "n_heads"
N_LAYERS_KEY = "n_layers"
CONTEXT_SIZE_KEY = "context_size"
D_FF_KEY = "d_ff"
