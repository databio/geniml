def build_subparser_model(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        '--model_folder',
        required=True,
        type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--file_list',
        type=str)
    group.add_argument(
        '--file_no',
        type=int)
    parser.add_argument(
        '--coverage_folder',
        required=True,
        type=str)
    parser.add_argument(
        '--coverage_core',
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


def build_subparser_universe_hard(parser):
    parser.add_argument(
        '--merge',
        default=0,
        type=int)
    parser.add_argument(
        '--filter_size',
        default=0,
        type=int)
    parser.add_argument(
        '--fout',
        required=True,
        type=str)
    parser.add_argument(
        '--coverage_file',
        required=True,
        type=str)
    parser.add_argument(
        '--cut_off',
        type=int)

    return parser


def build_subparser_universe_flexible(parser):
    parser.add_argument(
        '--output_file',
        required=True,
        type=str)
    parser.add_argument(
        '--model_folder',
        required=True,
        type=str)

    return parser


def build_subparser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "model": "Asses based on distance",
        "universe_hard": "Making cut-off universe",
        "universe_flexible": "Making ML flexible universe",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["model"] = build_subparser_model(subparsers["model"])
    subparsers["universe_hard"] = build_subparser_universe_hard(subparsers["universe_hard"])
    subparsers["universe_flexible"] = build_subparser_universe_flexible(subparsers["universe_flexible"])

    return parser
