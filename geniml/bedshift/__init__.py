# Project configuration, particularly for logging.

import logmuse

from ._version import __version__
from .bedshift import Bedshift
from .yaml_handler import BedshiftYAMLHandler

__classes__ = ["Bedshift"]
__all__ = __classes__ + []

logmuse.init_logger("bedshift")
