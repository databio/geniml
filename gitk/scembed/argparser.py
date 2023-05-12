from ubiquerg import VersionInHelpParser

from ._version import __version__
from .const import *


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
            description="%(prog)s - embed single-cell data as region vectors",
        )

    # Pipeline-specific arguments
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        required=True,
        help="Path to MarketMatrix format count matrix.",
    )

    parser.add_argument(
        "--nothreads",
        dest="nothreads",
        default=1,
        help="Number of available processors for  " "Word2Vec training.",
    )

    parser.add_argument(
        "--noreads",
        dest="noreads",
        default=2,
        help="Minimum number of reads that overlap a region "
        "for that region to be included.",
    )

    parser.add_argument(
        "--window-size",
        dest="window_size",
        default=100,
        help="Word2Vec window size.",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=30,
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--dimension",
        dest="dimension",
        default=100,
        help="Number of dimensions to train the word2vec " "model.",
    )

    parser.add_argument(
        "--min-count",
        dest="min_count",
        default=10,
        help="Minimum count for Word2Vec model.",
    )

    return parser
