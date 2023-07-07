def build_subparser_hmm(parser):
    parser = build_subparser(parser)

    parser.add_argument(
        "--not-normalize",
        help="if not to normalize coverage signal before using HMM",
        action="store_false",
    )
    parser.add_argument(
        "--save-max-cove",
        help="if present saves maximum coverage for each peak",
        action="store_true",
    )

    return parser


def build_subparser_ml(parser):
    parser = build_subparser(parser)
    parser.add_argument("--model-file", help="path to lh model file", required=True, type=str)
    return parser


def build_subparser_cc(parser):
    parser = build_subparser(parser)
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
    parser.add_argument("--cutoff", help="cutoff value used for making universe", type=int)

    return parser


def build_subparser(parser):
    parser.add_argument(
        "--output-file",
        help="path to output, universe file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--coverage-folder",
        help="path to core coverage folder",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--coverage-prefix",
        help="prefixed used for making coverage files",
        default="all",
        type=str,
    )

    return parser


def build_mode_parser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "cc": "Making coverage cut-off universe",
        "ccf": "Making coverage cut-off flexible universe",
        "ml": "Making ML universe",
        "hmm": "Making HMM universe",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["cc"] = build_subparser_cc(subparsers["cc"])
    subparsers["ccf"] = build_subparser(subparsers["ccf"])
    subparsers["ml"] = build_subparser_ml(subparsers["ml"])
    subparsers["hmm"] = build_subparser_hmm(subparsers["hmm"])
    return parser
