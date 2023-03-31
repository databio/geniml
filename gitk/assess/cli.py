def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        "--raw_data_folder", help="folder with raw data", type=str, required=True
    )
    parser.add_argument(
        "--file_list",
        help="list of files that need to be assessed",
        type=str,
        required=True,
    )
    parser.add_argument("--universe", help="universe file", type=str, required=True)
    parser.add_argument(
        "--npool", help="number of core that should be used", default=4, type=int
    )
    parser.add_argument(
        "--save_to_file",
        help="if save statistics for each BED file to a file",
        action="store_true",
    )
    parser.add_argument(
        "--folder_out", help="folder to which save the statistic", type=str
    )
    parser.add_argument("--pref", help="statistic file prefix", type=str)

    return parser


def build_subparser_distance(parser):
    parser = build_subparser(parser)
    parser.add_argument(
        "--flexible",
        help="if calculate the distance taking into account that the universe is flexible",
        action="store_true",
    )
    parser.add_argument(
        "--save_each",
        help="if save distance for each peak in each file ",
        action="store_true",
    )

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
