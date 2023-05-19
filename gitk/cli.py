from typing import Dict
import logmuse
import sys

from ubiquerg import VersionInHelpParser

from .assess.cli import build_mode_parser as assess_subparser
from .eval.cli import build_subparser as eval_subparser
from .hmm.cli import build_subparser as hmm_subparser
from .likelihood.cli import build_subparser as likelihood_subparser
from .scembed.argparser import build_argparser as scembed_subparser
from .bedspace.cli import build_argparser as bedspace_subparser

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
        "bedspace": "Coembed regionsets (bed files) and labels",
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
    subparsers["bedspace"] = bedspace_subparser(subparsers["bedspace"])

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

    _LOGGER.info(f"Command was: {args.command}")

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

        return

    if args.command == "bedspace":
        from .bedspace.const import PREPROCESS_CMD, TRAIN_CMD, DISTANCES_CMD, SEARCH_CMD

        _LOGGER.info(f"Subcommand: {args.subcommand}")

        if args.subcommand == PREPROCESS_CMD:
            from .bedspace.pipeline.preprocess import main as preprocess_main

            _LOGGER.info("Running bedspace preprocess")
            preprocess_main(
                args.input, args.metadata, args.universe, args.output, args.labels
            )

        elif args.subcommand == TRAIN_CMD:
            from .bedspace.pipeline.train import main as train_main

            _LOGGER.info("Running bedspace train")
            train_main(
                args.path_to_starspace,
                args.input,
                args.output,
                args.num_epochs,
                args.dim,
                args.learning_rate,
            )

        elif args.subcommand == DISTANCES_CMD:
            from .bedspace.pipeline.distances import main as distances_main

            _LOGGER.info("Running bedspace distances")
            distances_main(
                args.input,
                args.metadata,
                args.universe,
                args.output,
                args.labels,
                args.files,
                args.threshold,
            )

        elif args.subcommand == SEARCH_CMD:
            from .bedspace.const import SearchType
            from .bedspace.pipeline.search import run_scenario1 as scenario1
            from .bedspace.pipeline.search import run_scenario2 as scenario2
            from .bedspace.pipeline.search import run_scenario3 as scenario3

            if args.type == SearchType.l2r:
                _LOGGER.info("Running bedspace search (scenario 1)")
                scenario1(
                    args.query,
                    args.distances,
                    args.num_results,
                )
            elif args.type == SearchType.r2l:
                _LOGGER.info("Running bedspace search (scenario 2)")
                scenario2(
                    args.query,
                    args.distances,
                    args.num_results,
                )
            elif args.type == SearchType.l2l:
                _LOGGER.info("Running bedspace search (scenario 3)")
                scenario3(
                    args.query,
                    args.distances,
                    args.num_results,
                )

        else:
            # print help for this subcommand
            _LOGGER.info("Running bedspace help")


if __name__ == "__main__":
    main()
