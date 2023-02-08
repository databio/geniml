def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        "--out_file",
        type=str,
        help="path to result file",
        required=True)
    parser.add_argument(
        "--cov_folder",
        type=str,
        help="path to coverage folder",
        required=True)
    parser.add_argument(
        "--normalize",
        action='store_true')
    parser.add_argument(
        "--save_max_cove",
        help="if present saves maximum coverage for each peak",
        action='store_true')
    parser.add_argument(
        "--use_npy",
        action='store_true')
    parser.add_argument(
        "--lambdas",
        type=str,
        help="lambdas matrix used to set emissions")
    parser.add_argument(
        '--coverage_body',
        default='all_core',
        type=str)
    parser.add_argument(
        '--coverage_starts',
        default='all_start',
        type=str)
    parser.add_argument(
        '--coverage_ends',
        default='all_end',
        type=str)

    return parser
