class GenimlBaseError(Exception):
    """Base error type for peppy custom errors."""

    def __init__(self, msg):
        super(GenimlBaseError, self).__init__(msg)
