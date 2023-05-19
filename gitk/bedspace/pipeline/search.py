import logging

from ..const import DEFAULT_NUM_SEARCH_RESULTS, PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def main(
    type: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """
    Run the search command.

    :param type: The type of search to run.
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    """
    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE
