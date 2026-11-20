# Project configuration, particularly for logging.

import logmuse

from .bedshift import Bedshift  # noqa: F401
from .yaml_handler import BedshiftYAMLHandler  # noqa: F401

__all__ = ["Bedshift"]

logmuse.init_logger("bedshift")
