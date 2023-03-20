from enum import Enum
from logging import getLogger

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
        type: ScheduleType = ScheduleType.EXPONENTIAL,
        decay: float = None,
        n_epochs: int = None,
    ):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.type = type
        self.n_epochs = n_epochs

        # init the current lr and iteration
        self._current_lr = init_lr
        self._iter = 1

        # init decay rate
        if decay is None:
            _LOGGER.warning(
                "No decay rate provided. Calculating decay rate from init_lr and n_epochs."
            )
            self.decay = init_lr / n_epochs

    def _update_linear(self, epoch: int):
        lr = self.init_lr - (self.decay * epoch)
        return max(lr, self.min_lr)

    def _update_exponential(self, epoch: int):
        lr = self.get_lr() * 1 / (1 + (self.decay * epoch))
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
