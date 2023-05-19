def build_subparser_model(parser):
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

    return parser


def build_subparser_universe_hard(parser):
    parser.add_argument(
        "--merge",
        help="distance between output peaks that should be merged into one in output universe",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--filter-size",
        help="minimal siez of the region in the universe",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--fout", help="output file with the universe", required=True, type=str
    )
    parser.add_argument(
        "--coverage-file", help="path to core coverage file", required=True, type=str
    )
    parser.add_argument(
        "--cutoff", help="cutoff value used for making universe", type=int
    )

    return parser


def build_subparser_universe_flexible(parser):
    parser.add_argument(
        "--output-file", help="path to output, universe file", required=True, type=str
    )
    parser.add_argument(
        "--model-file", help="path to lh model file", required=True, type=str
    )
    parser.add_argument(
        "--coverage-prefix",
        help="prefixed used for making coverage files",
        default="all",
        type=str,
    )
    parser.add_argument(
        "--cov-folder", type=str, help="path to coverage folder", required=True
    )
    return parser


def build_subparser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "build_model": "Assess based on distance",
        "universe_hard": "Making cut-off universe",
        "universe_flexible": "Making ML flexible universe",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["build_model"] = build_subparser_model(subparsers["build_model"])
    subparsers["universe_hard"] = build_subparser_universe_hard(
        subparsers["universe_hard"]
    )
    subparsers["universe_flexible"] = build_subparser_universe_flexible(
        subparsers["universe_flexible"]
    )

    return parser
