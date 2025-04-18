from logging import getLogger

from .const import DEFAULT_CACHE_FOLDER, MODULE_NAME

_LOGGER = getLogger(MODULE_NAME)


def build_subparser_cache_bed(parser):
    """
    Builds argument parser to support to cache a BED file from local file or BEDbase.
    """
    parser.add_argument("identifier", nargs=1, help="BED file identifier, url, or file path")
    parser.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser


def build_subparser_cache_bedset(parser):
    """
    Builds argument parser to support to cache a BEDset from local folder or BEDbase.
    """
    parser.add_argument("identifier", nargs=1, help="BED file identifier, url, or file path")
    parser.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser


def build_subparser_seek(parser):
    """
    Builds argument parser to support to seek the path of BED file or BEDset.
    """
    parser.add_argument("identifier", nargs=1, help="BED file identifier, url, or file path")
    parser.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser


def build_subparser_inspect(parser):
    """
    Builds argument parser to support to list and count files and subdirectories in the subdirectory bedfiles and bedsets.
    """
    parser.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser


def build_subparser_cache_tokens(parser):
    """
    Builds argument parser to support to cache tokens from local file or BEDbase.
    """
    parser.add_argument(
        "--bed-id", dest="bed_id", nargs=1, help="Token file identifier, url, or file path"
    )
    parser.add_argument(
        "--universe-id",
        dest="universe_id",
        nargs=1,
        help="Unique identifier for the universe of the token file",
    )
    parser.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser


def build_subparser_remove(parser):
    """
    Builds argument parser to support to remove bed files or bedsets from the cache folder
    """
    parser.add_argument("identifier", nargs=1, help="BED file identifier, url, or file path")
    parser.add_argument(
        "--cache-folder",
        default=DEFAULT_CACHE_FOLDER,
        help="Cache folder path (default: bed_cache)",
    )

    return parser


def build_subparser(parser):
    """
    Builds argument parser to support the eval command line interface.
    """
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "cache-bed": "Cache a BED file from local file or BEDbase",
        "cache-tokens": "Cache tokens from local file or BEDbase",
        "cache-bedset": "Cache a BED set from local folder or BEDbase",
        "seek": "Seek the BED / BEDset path by giving identifier",
        "inspect-bedfiles": "Inspect the contents of bedfile cache folder",
        "inspect-bedsets": "Inspect the contents of bedset cache folder",
        "rm": "Remove the BED/BEDset from cache with given identifier",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["cache-bed"] = build_subparser_cache_bed(subparsers["cache-bed"])
    subparsers["cache-tokens"] = build_subparser_cache_tokens(subparsers["cache-tokens"])
    subparsers["cache-bedset"] = build_subparser_cache_bedset(subparsers["cache-bedset"])
    subparsers["seek"] = build_subparser_seek(subparsers["seek"])
    subparsers["inspect"] = build_subparser_inspect(subparsers["inspect-bedfiles"])
    subparsers["inspect"] = build_subparser_inspect(subparsers["inspect-bedsets"])
    subparsers["rm"] = build_subparser_remove(subparsers["rm"])
    return parser
