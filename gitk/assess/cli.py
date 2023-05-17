def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument("--raw-data-folder", type=str, required=True)
    parser.add_argument("--file-list", type=str, required=True)
    parser.add_argument("--universe", type=str, required=True)
    parser.add_argument("--npool", default=4, type=int)
    parser.add_argument("--save-to-file", action="store_true")
    parser.add_argument("--folder_out", type=str)
    parser.add_argument("--pref", type=str)

    return parser


def build_subparser_distance(parser):
    parser = build_subparser(parser)
    parser.add_argument("--flexible", action="store_true")
    parser.add_argument("--save-each", action="store_true")

    return parser


def build_mode_parser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "distance": "Asses based on distance",
        "intersection": "Asses based on coverage",
        "recovered": "Asses based on percent of recovered starts, ends",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["distance"] = build_subparser_distance(subparsers["distance"])
    subparsers["intersection"] = build_subparser(subparsers["intersection"])
    subparsers["recovered"] = build_subparser(subparsers["recovered"])

    return parser
