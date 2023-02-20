from ubiquerg import VersionInHelpParser

from ._version import __version__
from .const import *


def build_argparser():
    """
    Parse command-line arguments passed to the pipeline.
    """
    # Argument Parsing
    ###########################################################################
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
        "-n",
        "--names",
        default=None,
        type=str,
        required=False,
        help="Path to sample/barcodes names in a single "
        "column tab-delimited format.",
    )

    parser.add_argument(
        "-c",
        "--coords",
        default=None,
        type=str,
        required=False,
        help="Path to sample/barcodes coordinates in a "
        "chr, start, end tab-delimited format.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        required=True,
        help="Path to output directory to store results.",
    )

    parser.add_argument(
        "-t",
        "--title",
        default="scembed",
        type=str,
        required=False,
        help="Project/run title for naming output.",
    )

    parser.add_argument(
        "--docs",
        dest="docs",
        default=None,
        help="Path to documents dictionary.",
    )

    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Path to Word2Vec model.",
    )

    parser.add_argument(
        "--embed",
        dest="embed_file",
        default=None,
        help="Path to document embeddings dictionary.",
    )

    parser.add_argument(
        "--label-delimiter",
        dest="label_delimiter",
        default="_",
        help="Delimiter used to split cell names.",
    )

    parser.add_argument(
        "--nothreads",
        dest="nothreads",
        default=1,
        help="Number of available processors for  " "Word2Vec training.",
    )

    parser.add_argument(
        "--nocells",
        dest="nocells",
        default=5,
        help="Minimum number of cells with a shared region "
        "for that region to be included.",
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

    parser.add_argument(
        "--shuffle-repeat",
        dest="shuffle_repeat",
        default=5,
        help="Number of times to shuffle the document to "
        "generate date for Word2Vec.",
    )

    parser.add_argument(
        "--umap-nneighbors",
        dest="umap_nneighbours",
        default=100,
        help="Number of neighbors for UMAP plot.",
    )

    parser.add_argument(
        "--umap-metric",
        dest="umap_metric",
        default="euclidean",
        help="UMAP distance metric.",
    )

    parser.add_argument(
        "--rasterize",
        dest="rasterize",
        default=False,
        action="store_true",
        help="Rasterize the UMAP scatter plot to reduce "
        "space and plot generation time.",
    )
    return parser
