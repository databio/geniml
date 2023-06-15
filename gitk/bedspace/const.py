from enum import Enum
from pathlib import Path

PKG_NAME = "bedspace"

CACHE_DIR = MODEL_CACHE_DIR = str(Path.home() / f".{PKG_NAME}")

PREPROCESS_CMD = "preprocess"
TRAIN_CMD = "train"
DISTANCES_CMD = "distances"
SEARCH_CMD = "search"


DEFAULT_NUM_SEARCH_RESULTS = 10
DEFAULT_NUM_EPOCHS = 50
DEFAULT_DIM = 100
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_THRESHOLD = 0.5


# search type enum. Can be "l2r", "r2l", or "r2r"
class SearchType(Enum):
    l2r = "l2r"
    r2l = "r2l"
    r2r = "r2r"
