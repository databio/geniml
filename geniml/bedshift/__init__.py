# Project configuration, particularly for logging.

import logmuse

from .bedshift import Bedshift  # noqa: F401
from .yaml_handler import BedshiftYAMLHandler  # noqa: F401

__classes__ = ["Bedshift"]
__all__ = __classes__ + []

logmuse.init_logger("bedshift")
