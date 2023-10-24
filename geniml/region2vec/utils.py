import logging
import os
import select
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from random import shuffle
from typing import Dict, List, Union, Tuple, Any

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from ..io import Region, RegionSet
from .const import (
    DEFAULT_INIT_LR,
    DEFAULT_MIN_LR,
    MODULE_NAME,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_N_SHUFFLES,
)

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


class ScheduleType(Enum):
    """Learning rate schedule types"""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class LearningRateScheduler:
    """
    Simple class to track learning rates of the training procedure

    Based off of: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
    """

    def __init__(
        self,
        init_lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        type: Union[str, ScheduleType] = ScheduleType.EXPONENTIAL,
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
        if isinstance(type, str):
            try:
                self.type = ScheduleType[type.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown schedule type: {type}. Must be one of ['linear', 'exponential']."
                )

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
        if self.type == ScheduleType.LINEAR:
            self._current_lr = self._update_linear(self._iter)
            self._iter += 1
        elif self.type == ScheduleType.EXPONENTIAL:
            self._current_lr = self._update_exponential(self._iter)
            self._iter += 1
        else:
            raise ValueError(f"Unknown schedule type: {self.type}")

    def get_lr(self):
        return self._current_lr


def shuffle_documents(
    documents: List[List[any]],
    n_shuffles: int,
    threads: int = None,
) -> List[List[any]]:
    """
    Shuffle around the genomic regions for each cell to generate a "context".

    :param List[List[str]] documents: the document list to shuffle.
    :param int n_shuffles: The number of shuffles to conduct.
    :param int threads: The number of threads to use for shuffling.
    """

    def shuffle_list(l: List[any], n: int) -> List[any]:
        for _ in range(n):
            shuffle(l)
        return l

    _LOGGER.debug(f"Shuffling documents {n_shuffles} times.")
    shuffled_documents = documents.copy()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        shuffled_documents = list(
            track(
                executor.map(
                    shuffle_list,
                    shuffled_documents,
                    [n_shuffles] * len(documents),
                ),
                total=len(documents),
                description="Shuffling documents",
            )
        )
    return shuffled_documents


def make_syn1neg_file_name(model_file_name: str) -> str:
    """
    Make the syn1neg file name from the model file name.

    :param str model_file_name: The model file name.
    :return str: The syn1neg file name.
    """
    return f"{model_file_name}.syn1neg.npy"


def make_wv_file_name(model_file_name: str) -> str:
    """
    Make the wv file name from the model file name.

    :param str model_file_name: The model file name.
    :return str: The wv file name.
    """
    return f"{model_file_name}.wv.vectors.npy"


class Region2VecDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[any], any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[List[any], any]:
        # we need to return things as a tensor for proper batching
        return self.samples[idx]


def generate_window_training_data(
    data: List[List[any]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    threads: int = None,
    padding_value: any = 0,
    return_tensor: bool = True,
) -> List[Tuple[List[any], any]]:
    """
    Generates the windowed training data by sliding across the region sets. This is for the CBOW model.

    :param List[any] data: The data to generate the training data from.
    :param int window_size: The window size to use.
    :param int n_shuffles: The number of shuffles to perform.
    :param int threads: The number of threads to use.
    :param any padding_value: The padding value to use.
    :param bool return_tensor: Whether or not to return the data as a tensor.

    :return Tuple[List[List[any]], List[any]]: The contexts and targets.
    """
    _LOGGER.info("Generating windowed training data.")

    # shuffle the documents
    documents = shuffle_documents(
        [[t for t in tokens] for tokens in data], n_shuffles=n_shuffles, threads=threads
    )

    # compute the context length (inputs)
    context_len_req = 2 * window_size
    # contexts = []
    # targets = []
    samples = []
    for document in documents:
        for i, target in enumerate(document):
            context = document[max(0, i - window_size) : i] + document[i + 1 : i + window_size + 1]

            # pad the context if necessary
            if len(context) < context_len_req:
                context = context + [padding_value] * (context_len_req - len(context))

            # contexts.append(context)
            # targets.append(target)
            if return_tensor:
                samples.append(
                    (
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                    )
                )
            else:
                samples.append((context, target))

    # return contexts, targets
    return samples


def remove_below_min_count(l: List[Any]) -> List[Any]:
    """
    Remove elements from a list that fall below the minimum count.
    """
    counts = {}
    for e in l:
        counts[e] = counts.get(e, 0) + 1
    return [e for e in l if counts[e] > 1]
