import logging
import os
import select
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from typing import Dict, List, Union, Tuple

import numpy as np
import torch

from yaml import safe_dump, safe_load

from ..tokenization.main import Tokenizer, ITTokenizer
from .const import (
    LR_TYPES,
    DEFAULT_INIT_LR,
    DEFAULT_MIN_LR,
    MODULE_NAME,
    CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
    VOCAB_SIZE_KEY,
    EMBEDDING_DIM_KEY,
    EMBEDDING_DIM_KEY_OLD,
)
from .models import Region2Vec

_LOGGER = logging.getLogger(MODULE_NAME)


def prRed(skk: str) -> None:
    """Prints the input string skk in the red font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[91m{skk}\033[00m")


def prGreen(skk: str) -> None:
    """Prints the input string skk in the green font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[92m{skk}\033[00m")


def prYellow(skk: str) -> None:
    """Prints the input string skk in the yellow font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[93m{skk}\033[00m")


def prLightPurple(skk: str) -> None:
    """Prints the input string skk in the light purple font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[94m{skk}\033[00m")


def prPurple(skk: str) -> None:
    """Prints the input string skk in the purple font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[95m{skk}\033[00m")


def prCyan(skk: str) -> None:
    """Prints the input string skk in the cyan font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[96m{skk}\033[00m")


def prLightGray(skk: str) -> None:
    """Prints the input string skk in the gray font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[97m{skk}\033[00m")


def prBlack(skk: str) -> None:
    """Prints the input string skk in the black font.

    Args:
        skk (str): The string to print.
    """
    print(f"\033[98m{skk}\033[00m")


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


class Timer:
    """Records the running time.

    Uses Timer.s() or Timer() to record the start time. Then, calls Timer.t() to get the
    elapsed time in seconds.
    """

    def __init__(self):
        """Initializes a Timer object and starts the timer."""
        self.v = time.time()

    def s(self):
        """Restarts the timer."""
        self.v = time.time()

    def t(self):
        """Gives the elapsed time.

        Returns:
            float: The elapsed time in seconds.
        """
        return time.time() - self.v


def time_str(t: float) -> str:
    """Converts time in float to a readable format.

    Converts time in float to hours, minutes, or seconds based on the value of
    t.

    Args:
        t (float): Time in seconds.

    Returns:
        str: Time in readable time.
    """
    if t >= 3600:
        return f"{t / 3600:.2f}h"
    if t >= 60:
        return f"{t / 60:.2f}m"
    return f"{t:.2f}s"


def timed_response(prompt: str, wait_time: int, default: str):
    """Prints prompt and waits for response.

    Args:
        prompt (str): The question asks for a response.
        wait_time (int): The number of seconds for waiting.
        default (str): If no response received, uses default as the response.

    Returns:
        str: a response given by the user or the default one.
    """
    print(prompt, end="", flush=True)
    i, o, e = select.select([sys.stdin], [], [], wait_time)
    if i:
        ans = sys.stdin.readline().strip()
        if ans not in ["y", "n"]:
            print(f"\033[91m{default}\033[00m")
            return default
        else:
            return ans
    else:
        print(f"\033[91m{default}\033[00m")
        return default


def log(obj: str, filename: str = "log.txt") -> None:
    """Adds information in obj to a file specified by filename.

    Adds information in obj to a file (default: log.txt) and prints obj.

    Args:
        obj (str): A string.
        filename (str, optional): The log file name. Defaults to "log.txt".
    """
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            f.write(obj)
            f.write("\n")


class lr_scheduler:
    """Changes the learning rate.

    Changes the learning rate using the mode of linear or milestones.
    If mode = "linear", then the learning rate linearly decreases after certain
    epochs.
    If mode = "milestones", then the learning rate decreases at specified
    epochs.
    """

    def __init__(
        self,
        init_lr: float,
        end_lr: float,
        epochs: int,
        lr_info: Dict[str, Union[int, float, list]],
        mode: str = "linear",
    ):
        """Initializes the learning rate scheduler.

        Args:
            init_lr (float): The initial learning rate.
            end_lr (float): The last learning rate.
            epochs (int): The number of training epochs.
            lr_info (dict[str,Union[int,list]]): Dictionary storing information
                for learning rate scheduling.
            mode (str, optional): The mode of learning rate scheduling.
                Defaults to "linear".
        """
        self.lr = init_lr
        self.end_lr = end_lr
        self.init_lr = init_lr
        self.mode = mode
        self.epochs = epochs
        self.lr_info = lr_info
        self.count = 0
        if mode == "linear":
            self.freq = lr_info["freq"]

    def step(self):
        """Updates the learning rate.

        Returns:
            float: Current learning rate.
        """
        self.count += 1
        if self.mode == "linear":
            if self.count % self.freq == 0:
                self.lr = self.init_lr - (self.init_lr - self.end_lr) / self.epochs * self.count
        elif self.mode == "milestone":
            milestones = np.array(self.lr_info["milestones"])
            power = (milestones <= self.count).sum()
            self.lr = self.init_lr * np.power(self.lr_info["ratio"], float(power))
            if self.lr < self.end_lr:
                self.lr = self.end_lr
        return self.lr


def ensure_dir(folder: str, default: str = "y") -> None:
    """Makes sure the folder exists.

    Makes sure the folder exists. If the folder exists, then asks the user to
    keep [n] or delete [y] it. If no response received after 5 secs, then
    deletes the folder and create a new one.

    Args:
        folder (str): The folder to be created.
        default (str, optional): Choose whether to delete [y] or keep [n] the
          folder. Defaults to y.
    """
    if os.path.exists(folder):
        if default == "y":
            prompt = f"\033[91m{folder} exists,remove?([y]/n):\033[00m "
        else:
            prompt = f"\033[91m{folder} exists,remove?(y/[n]):\033[00m "
        ans = timed_response(prompt, 5, default)
        if ans != "n":
            shutil.rmtree(folder)
        else:
            return
    os.makedirs(folder, exist_ok=True)


class LearningRateScheduler:
    """
    Simple class to track learning rates of the training procedure

    Based off of: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
    """

    def __init__(
        self,
        init_lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        type: LR_TYPES = "exponential",
        decay: float = None,
        n_epochs: int = None,
    ):
        """
        :param float init_lr: The initial learning rate
        :param float min_lr: The minimum learning rate
        :param str type: The type of learning rate schedule to use. Must be one of ['linear', 'exponential'].
        :param float decay: The decay rate to use. If None, this will be calculated from init_lr and n_epochs.
        :param int n_epochs: The number of epochs to train for. Only used if decay is None.
        """
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs

        # convert type to learning rate if necessary
        if type not in ["constant", "linear", "exponential"]:
            raise ValueError(
                f"Unknown schedule type: {type}. Must be one of ['constant', 'linear', 'exponential']."
            )

        self.type = type

        # init the current lr and iteration
        self._current_lr = init_lr
        self._iter = 1

        # init decay rate
        if decay is None:
            _LOGGER.warning(
                "No decay rate provided. Calculating decay rate from init_lr and n_epochs."
            )
            self.decay = init_lr / n_epochs
        else:
            self.decay = decay

    def _update_linear(self, epoch: int):
        """
        Update the learning rate using a linear schedule.

        :param int epoch: The current epoch
        """

        lr = self.init_lr - (self.decay * epoch)
        return max(lr, self.min_lr)

    def _update_exponential(self, epoch: int):
        """
        Update the learning rate using an exponential schedule.

        :param int epoch: The current epoch
        """
        lr = self.get_lr() * (1 / (1 + self.decay * epoch))
        return max(lr, self.min_lr)

    def update(self):
        # update the learning rate according to the type
        if self.type == "linear":
            self._current_lr = self._update_linear(self._iter)
            self._iter += 1
        elif self.type == "exponential":
            self._current_lr = self._update_exponential(self._iter)
            self._iter += 1
        elif self.type == "constant":
            pass  # do nothing
        else:
            raise ValueError(f"Unknown schedule type: {self.type}")

    def get_lr(self):
        return self._current_lr


def shuffle_documents(
    documents: List[List[any]],
    n_shuffles: int = 1,
    threads: int = None,
) -> List[List[any]]:
    """
    Shuffle around the genomic regions for each cell to generate a "context".

    :param List[List[str]] documents: the document list to shuffle.
    :param int n_shuffles: The number of shuffles to conduct.
    """

    def shuffle_list(list: List[any], n: int) -> List[any]:
        for _ in range(n):
            shuffle(list)
        return list

    _LOGGER.debug(f"Shuffling documents {n_shuffles} times.")
    shuffled_documents = documents.copy()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        shuffled_documents = list(
            executor.map(
                shuffle_list,
                shuffled_documents,
                [n_shuffles] * len(documents),
            ),
        )
    return shuffled_documents


def export_region2vec_model(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    path: str,
    checkpoint_file: str = MODEL_FILE_NAME,
    universe_file: str = UNIVERSE_FILE_NAME,
    config_file: str = CONFIG_FILE_NAME,
    **kwargs: Dict[str, any],
):
    """
    Export the region2vec model to a folder

    :param torch.nn.Module model: The model to export
    :param Tokenizer tokenizer: The tokenizer to export
    :param str path: The path to export the model to
    :param str checkpoint_file: The name of the checkpoint file to export
    :param str universe_file: The name of the universe file to export
    :param str config_file: The name of the config file to export
    :param Dict[str, any] kwargs: Any additional arguments to pass to the config file
    """
    # make sure the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    # export the model weights
    torch.save(model.state_dict(), os.path.join(path, checkpoint_file))

    # export the vocabulary
    with open(os.path.join(path, universe_file), "a") as f:
        for region in tokenizer.universe.regions:
            f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

    # export the config (vocab size, embedding size)
    config = {
        VOCAB_SIZE_KEY: len(tokenizer),
        EMBEDDING_DIM_KEY: model.embedding_dim,
    }
    if kwargs:
        config.update(kwargs)

    with open(os.path.join(path, config_file), "w") as f:
        safe_dump(config, f)


def load_local_region2vec_model(
    model_path: str,
    vocab_path: str,
    config_path: str,
) -> Tuple[Region2Vec, ITTokenizer, dict]:
    """
    Load a region2vec model from a local directory

    :param str model_path: The path to the model checkpoint file
    :param str config_path: The path to the model config file
    :param str vocab_path: The path to the model vocabulary file
    """
    # init the tokenizer - only one option for now
    tokenizer = ITTokenizer(vocab_path)

    # load the model state dict (weights)
    params = torch.load(model_path)

    # get the model config (vocab size, embedding size)
    with open(config_path, "r") as f:
        config = safe_load(f)

    # try with new key first, then old key for backwards compatibility
    embedding_dim = config.get(EMBEDDING_DIM_KEY, config.get(EMBEDDING_DIM_KEY_OLD))
    if embedding_dim is None:
        raise KeyError(
            f"Could not find embedding dimension in config file. Expected key {EMBEDDING_DIM_KEY} or {EMBEDDING_DIM_KEY_OLD}."
        )
    else:
        if EMBEDDING_DIM_KEY_OLD in config:
            _LOGGER.warning(
                f"Found old key {EMBEDDING_DIM_KEY_OLD} in config file. This key will be deprecated in future versions. Please notify this models maintainer."
            )

    model = Region2Vec(
        config[VOCAB_SIZE_KEY],
        embedding_dim=embedding_dim,
    )

    model.load_state_dict(params)

    return model, tokenizer, config
