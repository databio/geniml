import argparse
import logmuse
import os
import subprocess
import sys

from ubiquerg import VersionInHelpParser

from .hmm.cli import build_subparser as hmm_subparser
from .assess.cli import build_subparser as assess_subparser
from ._version import __version__

def build_argparser():
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    banner = "%(prog)s - Genomic Interval toolkit"
    additional_description = "\nhttps://gitk.databio.org"

    parser = VersionInHelpParser(
        prog="gitk",
        version=f"{__version__}",
        description=banner,
        epilog=additional_description,
    )

    # Individual subcommands
    msg_by_cmd = {
        "hmm": "Use an HMM to build a consensus peak set.",
        "lh": "Compute likelihood",
        "assess": "Assess a universe",
    }

    sp = parser.add_subparsers(dest="command")
    subparsers = {}
    for k,v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)

    subparsers["hmm"] = hmm_subparser(subparsers["hmm"])
    subparsers["assess"] = hmm_subparser(subparsers["assess"])

    return parser

def main(test_args=None):
    parser = logmuse.add_logging_options(build_argparser())
    args, _ = parser.parse_known_args()
    if test_args:
        args.__dict__.update(test_args)

    global _LOGGER
    _LOGGER = logmuse.logger_via_cli(args, make_root=True)

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    _LOGGER.info(f"Command: {args.command}")

    if args.command == "assess":
        pass

    if args.command == "lh":
        pass

    if args.command == "hmm":
        from .hmm.hmm import test_hmm
        test_hmm("testing")

        # run_hmm_save_bed(start=os.path.join(args.cov_folder, args.coverage_starts),
        #              end=os.path.join(args.cov_folder, args.coverage_ends),
        #              cove=os.path.join(args.cov_folder, args.coverage_body),
        #              out_file=args.out_file,
        #              normalize=args.normalize,
        #              save_max_cove=args.save_max_cove)

    return