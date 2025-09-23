import logging
import sys
from typing import Dict

import logmuse
from ubiquerg import VersionInHelpParser

from ._version import __version__
from .argparsers import build_distance_argparser as distance_subparser
from .argparsers import build_preprocess_argparser as preprocess_subparser
from .argparsers import build_search_argparser as search_subparser
from .argparsers import build_train_argparser as train_subparser
from .const import *

global _LOGGER
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(PKG_NAME)


def build_argparser(parser: VersionInHelpParser = None):
    """
    Parse command-line arguments passed to the pipeline.
    """
    # Argument Parsing
    ###########################################################################
    if parser is None:
        parser = VersionInHelpParser(
            prog=PKG_NAME,
            version=__version__,
            description="%(prog)s - co-embed region sets and labels.",
        )

    # Individual subcommands
    msg_by_cmd = {
        PREPROCESS_CMD: "Preprocess data for training",
        TRAIN_CMD: "Train the StarSpace model",
        DISTANCES_CMD: "Compute distances between region sets and labels",
        SEARCH_CMD: "Search for regionsets similar to a query",
    }

    sp = parser.add_subparsers(dest="subcommand")
    subparsers: Dict[str, VersionInHelpParser] = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)

    # build up subparsers for commands
    subparsers[PREPROCESS_CMD] = preprocess_subparser(subparsers[PREPROCESS_CMD])
    subparsers[TRAIN_CMD] = train_subparser(subparsers[TRAIN_CMD])
    subparsers[DISTANCES_CMD] = distance_subparser(subparsers[DISTANCES_CMD])
    subparsers[SEARCH_CMD] = search_subparser(subparsers[SEARCH_CMD])

    return parser
