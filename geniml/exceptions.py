class GenimlBaseError(Exception):
    """Base error type for peppy custom errors."""

    def __init__(self, msg):
        super(GenimlBaseError, self).__init__(msg)


class BBClientError(GenimlBaseError):
    """Base error type for BBClient errors."""

    def __init__(self, msg):
        super(BBClientError, self).__init__(msg)


class TokenizedFileNotFoundError(BBClientError):
    """Error raised when a tokenized file is not found."""

    def __init__(self, msg):
        super(TokenizedFileNotFoundError, self).__init__(msg)
