# Project configuration, particularly for logging.

import logmuse

from .bedshift import Bedshift
from .yaml_handler import BedshiftYAMLHandler

__all__ = ["Bedshift"]

logmuse.init_logger("bedshift")
