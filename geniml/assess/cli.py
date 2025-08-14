def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser: Argument parser
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
        "--raw-data-folder",
        help="folder with raw data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--file-list",
        help="list of files that need to be assessed",
        type=str,
        required=True,
    )
    parser.add_argument("--universe", help="universe file", type=str, required=True)
    parser.add_argument(
        "--no-workers",
        help="number of core that should be used",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--save-to-file",
        help="if save statistics for each BED file to a file",
        action="store_true",
    )
    parser.add_argument("--folder-out", help="folder to which save the statistic", type=str)
    parser.add_argument("--pref", help="statistic file prefix", type=str, required=True)

    parser.add_argument(
        "--save-each",
        help="if save distance for each peak in each file ",
        action="store_true",
    )

    return parser
