import logging

from ..const import DEFAULT_NUM_SEARCH_RESULTS, PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def run_scenario1(
    query: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """
    Run the search command. This is for scenario 1: Give me a label, I'll return region sets.

    :param query: The query string (a label).
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    """
    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE


def run_scenario2(
    query: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """
    Run the search command. This is for scenario 2: Give me a region set, I'll return labels.

    :param query: The query string (a path to a file).
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    """

    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE


def run_scenario3(
    query: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """
    Run the search command. This is for scenario 3: Give me a region set, I'll return region sets.

    :param query: The query string (a path to a file).
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    """

    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE
