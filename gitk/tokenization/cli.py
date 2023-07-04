def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """
    parser.add_argument(
        "--data-folder", type=str, help="Path to the folder that stores BED files"
    )
    parser.add_argument(
        "--token-folder", type=str, help="Folder that stores tokenized files"
    )
    # parameters for hard tokenization
    parser.add_argument("--universe", type=str, help="Path to a universe file")
    parser.add_argument("--nworkers", type=int, default=10, help="number of workers")
    parser.add_argument(
        "--bedtools-path",
        type=str,
        default="bedtools",
        help="Path to the bedtools binary. Default: bedtools. If bedtools does not exists, an exception will be raised",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0e-9,
        help="A parameter for bedtools.intersect",
    )

    return parser
