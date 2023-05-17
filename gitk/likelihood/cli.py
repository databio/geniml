def build_subparser_model(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument("--model-folder", required=True, type=str)
    parser.add_argument("--file-no", type=int)
    parser.add_argument("--coverage-folder", required=True, type=str)
    parser.add_argument("--coverage-prefix", default="all", type=str)

    return parser


def build_subparser_universe_hard(parser):
    parser.add_argument("--merge", default=0, type=int)
    parser.add_argument("--filter-size", default=0, type=int)
    parser.add_argument("--fout", required=True, type=str)
    parser.add_argument("--coverage-file", required=True, type=str)
    parser.add_argument("--cut-off", type=int)

    return parser


def build_subparser_universe_flexible(parser):
    parser.add_argument("--output-file", required=True, type=str)
    parser.add_argument("--model-folder", required=True, type=str)

    return parser


def build_subparser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "build_model": "Asses based on distance",
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
