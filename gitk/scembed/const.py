from pathlib import Path

__author__ = ["Nathan LeRoy", "Jason Smith", "Erfaneh Gharavi"]
__email__ = "nleroy@virginia.edu"

LOGGING_LEVEL = "INFO"
MODULE_NAME = "scembed"

DEFAULT_EPOCHS = 100
DEFAULT_GENSIM_EPOCHS = 1
DEFAULT_MIN_COUNT = 10
DEAFULT_N_SHUFFLES = 10
DEFAULT_WINDOW_SIZE = 5
DEFAULT_EMBEDDING_SIZE = 100
DEFAULT_EPOCHS = 10
DEFAULT_INIT_LR = (
    0.1  # https://github.com/databio/gitk/issues/6#issuecomment-1476273162
)
DEFAULT_MIN_LR = 0.0001  # gensim default
DEFAULT_DECAY_RATE = 0.95

DEFAULT_CHUNK_SIZE = 10000

CHR_KEY = "chr"
START_KEY = "start"
END_KEY = "end"

MODEL_CACHE_DIR = str(Path.home() / ".scembed")
MODEL_HUB_URL = "http://big.databio.org/scembed/models"
MODEL_CONFIG_FILE_NAME = "model.yaml"

DEFAULT_BEDTOOLS_PATH = "bedtools"
