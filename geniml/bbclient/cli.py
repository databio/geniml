from logging import getLogger

from .const import MODULE_NAME

_LOGGER = getLogger(MODULE_NAME)


def build_subparser(parser):
    """Build subparser for bbclient"""
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", description="Choose a subcommand"
    )
    subparsers.required = True

    # cache BED file
    parser_bedset = subparsers.add_parser("cache-bed", help="Cache a BED file from local file or BEDbase")
    parser_bedset.add_argument("--input-identifier", help="BED file identifier, url, or file path")
    parser_bedset.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    # cache BED set
    parser_bedset = subparsers.add_parser("cache-bedset", help="Cache a BED set from local file or BEDbase")
    parser_bedset.add_argument("--input-identifier", help="Bedset identifier")
    parser_bedset.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    # seek the path of BED file or BED set
    parser_ident = subparsers.add_parser("seek", help="Return the path in cache folder")
    parser_ident.add_argument("--input-identifier", help="BED file identifier")
    parser_ident.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    # seek the path of BED file or BED set
    parser_ident = subparsers.add_parser("tree", help="Return the path in cache folder")
    parser_ident.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    return parser
