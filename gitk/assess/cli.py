def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """
    parser.add_argument(
        "--overlap",
        help="if calculate base-level overlap score",
        action="store_true",
    )
    parser.add_argument(
        "--distance",
        help="if calculate distance from region in query to nearest region in the universe",
        action="store_true",
    )
    parser.add_argument(
        "--distance-universe-to-file",
        help="if calculate distance from region in the universe to nearest region query",
        action="store_true",
    )
    parser.add_argument(
        "--distance-flexible",
        help="if calculate distance from region in query to nearest region in the universe taking into account "
             "universe flexibility ",
        action="store_true",
    )
    parser.add_argument(
        "--distance-flexible-universe-to-file",
        help="if calculate distance from region in the universe to nearest region in query taking into account "
             "universe flexibility ",
        action="store_true",
    )
    parser.add_argument(
        "--raw-data-folder", help="folder with raw data", type=str, required=True
    )
    parser.add_argument(
        "--file-list",
        help="list of files that need to be assessed",
        type=str,
        required=True,
    )
    parser.add_argument("--universe", help="universe file", type=str, required=True)
    parser.add_argument(
        "--no-workers", help="number of core that should be used", default=4, type=int
    )
    parser.add_argument(
        "--save-to-file",
        help="if save statistics for each BED file to a file",
        action="store_true",
    )
    parser.add_argument(
        "--folder-out", help="folder to which save the statistic", type=str
    )
    parser.add_argument("--pref", help="statistic file prefix", type=str)

    parser.add_argument(
        "--save-each",
        help="if save distance for each peak in each file ",
        action="store_true",
    )

    return parser


def build_subparser_distance(parser):
    parser = build_subparser(parser)
    parser.add_argument(
        "--flexible",
        help="if calculate the distance taking into account that the universe is flexible",
        action="store_true",
    )
    parser.add_argument(
        "--save-each",
        help="if save distance for each peak in each file ",
        action="store_true",
    )
    parser.add_argument(
        "--universe-to-file",
        help="if calculate distance from universe to file",
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

    return parser
