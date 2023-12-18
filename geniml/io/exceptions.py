from ..exceptions import GenimlBaseError
from typing import Optional


class BackedFileNotAvailableError(GenimlBaseError):
    default_message = "File from url is not available in backed mode."

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.default_message)
