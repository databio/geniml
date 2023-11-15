# TODO: it won't work. BedFetch and BedDownloader doesn't exist
import os

from ..io import BedSet, RegionSet
from .bedfile_retrieval import BBClient


def download_bedset(args):
    """
    Download BED set from BEDbase
    """
    # init BBClient
    bbc = BBClient(args.cache_folder)
    bedset = bbc.load_bedset(args.bedset)
    print(f"BED set {args.bedset} with {len(bedset)} BED files has been cached")


def download_bedfile(args):
    """
    Download BED file from BED base
    """
    # init BBClient
    bbc = BBClient(args.cache_folder)
    bedfile = bbc.load_bed(args.input_identifier)
    print(f"BED file {args.input_identifier} has been cached")


def cache_local_bed_files(args):
    """
    Cache local BED files or BED sets (from local folders of BED files)
    """
    # init BBClient
    bbc = BBClient(args.cache_folder)
    # path from terminal input
    local_path = args.input_identifier
    # folder -> cached BedSet
    if os.path.isdir(local_path):
        bedset = BedSet(
            [os.path.join(local_path, file_name) for file_name in os.listdir(local_path)]
        )
        bbc.add_bedset_to_cache(bedset)
        print(f"BED set {bedset.compute_bedset_identifier()} has been cached")
    # file -> cached BED file
    else:
        bedfile = RegionSet(local_path)
        bbc.add_bed_to_cache(bedfile)
        print(f"BED file {bedfile.compute_bed_identifier()} has been cached")


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
    parser_bedset.set_defaults(func=download_bedset)

    # download BED files from BED base
    parser_ident = subparsers.add_parser("bedfile", help="Process identifiers")
    parser_ident.add_argument("--input-identifier", help="BED file identifier")
    parser_ident.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )
    parser_ident.set_defaults(func=download_bedfile)

    # cache local BED files / BED sets
    parser_local = subparsers.add_parser("local", help="Cache a local bed file")
    parser_local.add_argument("--input-identifier", help="Local BED file/folder path")
    parser_local.add_argument(
        "--cache-folder",
        default="bed_cache",
        help="Cache folder path (default: bed_cache)",
    )

    parser_local.set_defaults(func=cache_local_bed_files)

    return parser
