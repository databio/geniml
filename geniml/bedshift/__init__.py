# Project configuration, particularly for logging.

import logmuse
from .bedshift import Bedshift
from ._version import __version__

__classes__ = ["Bedshift"]
__all__ = __classes__ + []

logmuse.init_logger("bedshift")
