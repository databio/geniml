from logging import getLogger

from .const import MODULE_NAME

_LOGGER = getLogger(MODULE_NAME)


def build_subparser(parser):
    """Build subparser for bbclient"""
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", description="Choose a subcommand"
    )
    subparsers.required = True

    # download BED sets from BEDbase
    parser_bedset = subparsers.add_parser("bedset", help="Download a bedset")
    parser_bedset.add_argument("--bedset", help="Bedset identifier")
    parser_bedset.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    # download BED files from BED base
    parser_ident = subparsers.add_parser("bed", help="Process identifiers")
    parser_ident.add_argument("--input-identifier", help="BED file identifier")
    parser_ident.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    # cache local BED files / BED sets
    parser_local = subparsers.add_parser("local", help="Cache a local bed file")
    parser_local.add_argument("--input-identifier", help="Local BED file/folder path")
    parser_local.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    return parser
