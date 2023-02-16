def build_subparser_model(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        '--coverage_folder',
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
        '--model_folder',
        required=True,
        type=str)
    parser.add_argument(
        '--coverage_body',
        default='all_core',
        type=str)
    parser.add_argument(
        '--coverage_starts',
        default='all_end',
        type=str)
    parser.add_argument(
        '--coverage_ends',
        default='all_start',
        type=str)
    parser.add_argument(
        '--model_body',
        default='core',
        type=str)
    parser.add_argument(
        '--model_starts',
        default='starts',
        type=str)
    parser.add_argument(
        '--model_ends',
        default='ends',
        type=str)

    return parser


def build_subparser_universe(parser):
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


def build_subparser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "model": "Asses based on distance",
        "universe": "Asses based on coverage",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["model"] = build_subparser_model(subparsers["model"])
    subparsers["universe"] = build_subparser_universe(subparsers["universe"])

    return parser
