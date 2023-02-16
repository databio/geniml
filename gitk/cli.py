import argparse
import logmuse
import os
import subprocess
import sys

from ubiquerg import VersionInHelpParser

from .hmm.cli import build_subparser as hmm_subparser
from .assess.cli import build_mode_parser as assess_subparser
from .likelihood.cli import build_subparser as likelihood_subparser
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
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)

    subparsers["hmm"] = hmm_subparser(subparsers["hmm"])
    subparsers["assess"] = assess_subparser(subparsers["assess"])
    subparsers["lh"] = likelihood_subparser(subparsers["lh"])

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
        _LOGGER.info(f"Subcommand: {args.subcommand}")
        if args.subcommand == "distance":
            from .assess.sweep_distances import run_distance
            run_distance(args.raw_data_folder, args.file_list,
                         args.universe, args.npool, args.flexible,
                         args.save_to_file, args.folder_out,
                         args.pref, args.save_each)

        if args.subcommand == "intersection":
            from .assess.sweep_intersect import run_intersection
            run_intersection(args.raw_data_folder, args.file_list,
                             args.universe, args.npool,
                             args.save_to_file, args.folder_out,
                             args.pref)

        if args.subcommand == "recovered":
            from .assess.sweep_recovered import run_recovered
            run_recovered(args.raw_data_folder, args.file_list,
                          args.universe, args.npool,
                          args.save_to_file, args.folder_out,
                          args.pref)

    if args.command == "lh":
        _LOGGER.info(f"Subcommand: {args.subcommand}")
        if args.subcommand == "model":
            from .likelihood.likelihood_model import main
            main(args.model_folder, args.coverage_folder, args.coverage_starts,
                 args.coverage_ends, args.coverage_body,
                 args.model_starts, args.model_ends, args.model_body,
                 args.file_list, args.file_no)

        if args.subcommand == "universe":
            from .likelihood.likelihood_universe import main
            main(args.coverage_file, args.merge,
                 args.filter_size, args.fout, args.cut_off)

    if args.command == "hmm":
        from .hmm.hmm import run_hmm_save_bed
        run_hmm_save_bed(start=os.path.join(args.cov_folder, args.coverage_starts),
                         end=os.path.join(args.cov_folder, args.coverage_ends),
                         cove=os.path.join(args.cov_folder, args.coverage_body),
                         out_file=args.out_file,
                         normalize=args.normalize,
                         save_max_cove=args.save_max_cove)

    return
