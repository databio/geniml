def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        "--out_file", type=str, help="path to result file", required=True
    )
    parser.add_argument(
        "--cov_folder", type=str, help="path to coverage folder", required=True
    )
    parser.add_argument(
        "--normalize",
        help="if normalize coverage before using HMM",
        action="store_true",
    )
    parser.add_argument(
        "--save_max_cove",
        help="if present saves maximum coverage for each peak",
        action="store_true",
    )
    parser.add_argument(
        "--lambdas", type=str, help="lambdas matrix use to set emissions"
    )
    parser.add_argument(
        "--coverage_prefix",
        help="prefixed used for making coverage files",
        default="all",
        type=str,
    )

    return parser
