from logging import getLogger

from .const import DEFAULT_CACHE_FOLDER, MODULE_NAME

_LOGGER = getLogger(MODULE_NAME)


def build_subparser(parser):
    """Build subparser for bbclient"""
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", description="Choose a subcommand"
    )
    subparsers.required = True

    # cache BED file
    parser_bedset = subparsers.add_parser(
        "cache-bed", help="Cache a BED file from local file or BEDbase"
    )
    parser_bedset.add_argument("--input-identifier", help="BED file identifier, url, or file path")
    parser_bedset.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    # cache BED set
    parser_bedset = subparsers.add_parser(
        "cache-bedset", help="Cache a BED set from local file or BEDbase"
    )
    parser_bedset.add_argument("--input-identifier", help="Bedset identifier, or folder path")
    parser_bedset.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    # seek the path of BED file or BED set
    parser_ident = subparsers.add_parser(
        "seek", help="Seek the BED / BEDset path by giving identifier"
    )
    parser_ident.add_argument("--input-identifier", help="BED file / BEDset identifier")
    parser_ident.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    # list and count files and subdirectories in the subdirectory bedfiles and bedsets
    parser_ident = subparsers.add_parser("inspect", help="Inspect the contents of cache folder")
    parser_ident.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    # remove bed files or bedsets from the cache folder
    parser_ident = subparsers.add_parser(
        "rm", help="Remove the BED/BEDset from cache with given identifier"
    )
    parser_ident.add_argument("--input-identifier", help="BED file / BEDset identifier")
    parser_ident.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser
