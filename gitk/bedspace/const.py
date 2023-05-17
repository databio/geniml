from pathlib import Path

PKG_NAME = "bedspace"

CACHE_DIR = MODEL_CACHE_DIR = str(Path.home() / f".{PKG_NAME}")

PREPROCESS_CMD = "preprocess"
TRAIN_CMD = "train"
DISTANCES_CMD = "distances"
SEARCH_CMD = "search"
