import os
import select
import shutil
import sys
import time
from typing import Union
import numpy as np


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
        lr_info: dict[str, Union[int, float, list]],
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
