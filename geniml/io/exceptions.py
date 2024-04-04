from typing import Optional

from ..exceptions import GenimlBaseError


class BackedFileNotAvailableError(GenimlBaseError):
    default_message = "File from url is not available in backed mode."

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.default_message)


class BEDFileReadError(GenimlBaseError):
    default_message = "Error reading BED file."

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.default_message)
