from typing import Dict
import logmuse
import sys
import os

from ubiquerg import VersionInHelpParser

from .assess.cli import build_mode_parser as assess_subparser
from .eval.cli import build_subparser as eval_subparser
from .hmm.cli import build_subparser as hmm_subparser
from .likelihood.cli import build_subparser as likelihood_subparser
from .scembed.argparser import build_argparser as scembed_subparser

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
        "scembed": "Embed single-cell data as region vectors",
        "eval": "Evaluate a set of region embeddings",
    }

    sp = parser.add_subparsers(dest="command")
    subparsers: Dict[str, VersionInHelpParser] = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)

    # build up subparsers for modules
    subparsers["hmm"] = hmm_subparser(subparsers["hmm"])
    subparsers["assess"] = assess_subparser(subparsers["assess"])
    subparsers["lh"] = likelihood_subparser(subparsers["lh"])
    subparsers["scembed"] = scembed_subparser(subparsers["scembed"])
    subparsers["eval"] = eval_subparser(subparsers["eval"])
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
            from .assess.distance import run_distance

            run_distance(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.npool,
                args.flexible,
                args.save_to_file,
                args.folder_out,
                args.pref,
                args.save_each,
            )

        if args.subcommand == "intersection":
            from .assess.intersection import run_intersection

            run_intersection(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.npool,
                args.save_to_file,
                args.folder_out,
                args.pref,
            )

        if args.subcommand == "recovered":
            from .assess.recovered import run_recovered

            run_recovered(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.npool,
                args.save_to_file,
                args.folder_out,
                args.pref,
            )

    if args.command == "lh":
        _LOGGER.info(f"Subcommand: {args.subcommand}")
        if args.subcommand == "build_model":
            from .likelihood.build_model import main

            main(
                args.model_folder,
                args.coverage_folder,
                args.coverage_prefix,
                args.file_no,
            )

        if args.subcommand == "universe_hard":
            from .likelihood.universe_hard import main

            main(
                args.coverage_file,
                args.fout,
                args.merge,
                args.filter_size,
                args.cut_off,
            )

        if args.subcommand == "universe_flexible":
            from .likelihood.universe_flexible import main

            main(args.model_folder, args.output_file)

    if args.command == "hmm":
        from .hmm.hmm import run_hmm_save_bed

        run_hmm_save_bed(
            coverage_folder=args.cov_folder,
            out_file=args.out_file,
            prefix=args.coverage_prefix,
            normalize=args.normalize,
            save_max_cove=args.save_max_cove,
        )

    if args.command == "scembed":
        from .scembed.scembed import main as scembed_main

        _LOGGER.info("Running scembed")
        pass
        # scembed_main(test_args)
    if args.command == "eval":
        if args.subcommand == "gdst":
            from gitk.eval.gdst import get_gds
            gds = get_gds(
                args.model_path, 
                args.embed_type, 
                args.num_samples)
            print(gds)
        if args.subcommand == "npt":
            from gitk.eval.npt import get_snpr
            npt = get_snpr(
                args.model_path, 
                args.embed_type, 
                args.K,
                args.num_samples,
                resolution=args.K
            )
            print(npt["SNPR"][0])
        if args.subcommand == "cct-tss":
            from gitk.eval.cct import get_scctss
            scctss = get_scctss(
                args.model_path, 
                args.embed_type, 
                args.save_folder, 
                args.Rscript_path, 
                args.assembly, 
                num_samples=args.num_samples, 
                threshold=args.threshold)
            print(scctss)
    return


if __name__ == "__main__":
    main()
