import argparse
from .bedfile_retrieval import BedProcessor, BedCacheManager, BedDownloader


def download_bedset(args):
    bed_downloader = BedDownloader()
    bed_downloader.download_bedset_data(args.bedset)


def download_and_process_bed_region(args):
    bed_processor = BedProcessor(args.cache_folder)
    bed_regions = bed_processor.download_and_process_bed_region_data(
        args.input_identifier, args.chr, args.start, args.end
    )
    print(bed_regions)


def process_local_bed_data(args):
    cache_manager = BedCacheManager(args.cache_folder)
    bed_local = cache_manager.process_local_bed_data(args.input_identifier)
    print(bed_local)


def process_identifiers(args):
    bed_processor = BedProcessor(args.cache_folder)
    bed_ident = bed_processor.process_identifiers(args.input_identifier)
    print(bed_ident)


def build_subparser(parser):
    """Build subparser for bbclient"""
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", description="Choose a subcommand"
    )
    subparsers.required = True

    parser_bedset = subparsers.add_parser("bedset", help="Download a bedset")
    parser_bedset.add_argument("--bedset", help="Bedset identifier")
    parser_bedset.set_defaults(func=download_bedset)

    parser_region = subparsers.add_parser("region", help="Download and process a bed region")
    parser_region.add_argument("--input-identifier", help="BED file identifier")
    parser_region.add_argument("--chr", help="Chromosome number")
    parser_region.add_argument("--start", type=int, help="Start position")
    parser_region.add_argument("--end", type=int, help="End position")
    parser_region.add_argument(
        "--cache-folder", default="bed_cache", help="Cache folder path (default: bed_cache)"
    )
    parser_region.set_defaults(func=download_and_process_bed_region)

    parser_local = subparsers.add_parser("local", help="Process a local bed file")
    parser_local.add_argument("--input-identifier", help="Local BED file path")
    parser_local.add_argument(
        "--cache-folder", default="bed_cache", help="Cache folder path (default: bed_cache)"
    )
    parser_local.set_defaults(func=process_local_bed_data)

    parser_ident = subparsers.add_parser("identifiers", help="Process identifiers")
    parser_ident.add_argument("--input-identifier", help="BED file identifier")
    parser_ident.add_argument(
        "--cache-folder", default="bed_cache", help="Cache folder path (default: bed_cache)"
    )
    parser_ident.set_defaults(func=process_identifiers)

    return parser
