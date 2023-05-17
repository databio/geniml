def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        "--out-file", type=str, help="path to result file", required=True
    )
    parser.add_argument(
        "--cov-folder", type=str, help="path to coverage folder", required=True
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--save-max-cove",
        help="if present saves maximum coverage for each peak",
        action="store_true",
    )
    parser.add_argument(
        "--lambdas", type=str, help="lambdas matrix used to set emissions"
    )
    parser.add_argument("--coverage-prefix", default="all", type=str)

    return parser
