import os
import select
import shutil
import sys
import time

import numpy as np


def prRed(skk):
    print(f"\033[91m{skk}\033[00m")


def prGreen(skk):
    print(f"\033[92m{skk}\033[00m")


def prYellow(skk):
    print(f"\033[93m{skk}\033[00m")


def prLightPurple(skk):
    print(f"\033[94m{skk}\033[00m")


def prPurple(skk):
    print(f"\033[95m{skk}\033[00m")


def prCyan(skk):
    print(f"\033[96m{skk}\033[00m")


def prLightGray(skk):
    print(f"\033[97m{skk}\033[00m")


def prBlack(skk):
    print(f"\033[98m{skk}\033[00m")


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return f"{t / 3600:.2f}h"
    if t >= 60:
        return f"{t / 60:.2f}m"
    return f"{t:.2f}s"


def timed_response(prompt, wait_time, default):
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


def log(obj, filename="log.txt"):
    print(obj, flush=True)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            print(obj, file=f)


class lr_scheduler:
    def __init__(self, init_lr, end_lr, epochs, lr_info=None, mode="linear"):
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
        self.count += 1
        if self.mode == "linear":
            if self.count % self.freq == 0:
                self.lr = (
                    self.init_lr
                    - (self.init_lr - self.end_lr) / self.epochs * self.count
                )
        elif self.mode == "milestone":
            milestones = np.array(self.lr_info["milestones"])
            power = (milestones <= self.count).sum()
            self.lr = self.init_lr * np.power(self.lr_info["ratio"], float(power))
            if self.lr < self.end_lr:
                self.lr = self.end_lr
        return self.lr


def ensure_dir(path, default="y"):
    if os.path.exists(path):
        if default == "y":
            prompt = f"\033[91m{path} exists,remove?([y]/n):\033[00m "
        else:
            prompt = f"\033[91m{path} exists,remove?(y/[n]):\033[00m "
        ans = timed_response(prompt, 5, default)
        if ans != "n":
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path, exist_ok=True)
