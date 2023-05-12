from enum import Enum
from logging import getLogger
from typing import Union

import scanpy as sc

from .const import *

_LOGGER = getLogger(PKG_NAME)


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
        lr = self.init_lr - (self.decay * epoch)
        return max(lr, self.min_lr)

    def _update_exponential(self, epoch: int):
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


class AnnDataChunker:
    def __init__(self, adata: sc.AnnData, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Simple class to chunk an AnnData object into smaller pieces. Useful for
        training on large datasets.

        :param adata: AnnData object to chunk. Must be in backed mode. See: https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html
        :param chunk_size: Number of cells to include in each chunk
        """
        self.adata = adata
        self.chunk_size = chunk_size
        self.n_chunks = len(adata) // chunk_size + 1

    def __iter__(self):
        for i in range(self.n_chunks):
            chunk = self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :]
            yield chunk.to_memory()

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, item):
        return self.adata[item * self.chunk_size : (item + 1) * self.chunk_size, :]

    def __repr__(self):
        return f"<AnnDataChunker: {self.n_chunks} chunks of size {self.chunk_size}>"
