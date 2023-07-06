def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        "--model-file", help="path to file with lh model", required=True, type=str
    )
    parser.add_argument(
        "--file-no", help="number of files used to make the model", type=int
    )
    parser.add_argument(
        "--coverage-folder", help="path to coverage folder", required=True, type=str
    )
    parser.add_argument(
        "--coverage-prefix",
        help="prefix used when making coverage files",
        default="all",
        type=str,
    )
    parser.add_argument(
        "--force",
        help="if overwrite existing model",
        action="store_true",
    )
    return parser
