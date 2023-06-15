from typing import Dict
import logmuse
import sys

from ubiquerg import VersionInHelpParser

from .assess.cli import build_subparser as assess_subparser
from .eval.cli import build_subparser as eval_subparser
from .universe.cli import build_mode_parser as universe_subparser
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
        "universe": "Use an HMM to build a consensus peak set.",
        "lh": "Compute likelihood",
        "assess": "Assess a universe",
        "scembed": "Embed single-cell data as region vectors",
        "eval": "Evaluate a set of region embeddings",
        "bedspace": "Coembed regionsets (bed files) and labels",
    }

    sp = parser.add_subparsers(dest="command")
    subparsers: Dict[str, VersionInHelpParser] = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)

    # build up subparsers for modules
    subparsers["universe"] = universe_subparser(subparsers["universe"])
    subparsers["assess"] = assess_subparser(subparsers["assess"])
    subparsers["lh"] = likelihood_subparser(subparsers["lh"])
    subparsers["scembed"] = scembed_subparser(subparsers["scembed"])
    subparsers["eval"] = eval_subparser(subparsers["eval"])
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

    _LOGGER.info(f"Command: {args.command}")

    if args.command == "assess":
        from .assess.assess import run_all_assessment_methods

        run_all_assessment_methods(
            args.raw_data_folder,
            args.file_list,
            args.universe,
            args.no_workers,
            args.folder_out,
            args.pref,
            args.save_each,
            args.overlap,
            args.distance,
            args.distance_flexible,
            args.distance_universe_to_file,
            args.distance_flexible_universe_to_file,
        )

    if args.command == "lh":
        # _LOGGER.info(f"Subcommand: {args.subcommand}")
        # if args.subcommand == "build_model":
        from .likelihood.build_model import main

        main(
            args.model_file,
            args.coverage_folder,
            args.coverage_prefix,
            args.file_no,
            args.force,
        )

    if args.command == "universe":
        _LOGGER.info(f"Subcommand: {args.subcommand}")
        if args.subcommand == "hmm":
            from .universe.hmm_universe import hmm_universe

            hmm_universe(
                coverage_folder=args.coverage_folder,
                out_file=args.output_file,
                prefix=args.coverage_prefix,
                normalize=args.not_normalize,
                save_max_cove=args.save_max_cove,
            )

        if args.subcommand == "ml":
            from .universe.ml_universe import ml_universe

            ml_universe(
                model_file=args.model_file,
                cove_folder=args.coverage_folder,
                cove_prefix=args.coverage_prefix,
                file_out=args.output_file,
            )

        if args.subcommand == "cc":
            from .universe.cc_universe import cc_universe

            cc_universe(
                cove=args.coverage_folder,
                cove_prefix=args.coverage_prefix,
                file_out=args.output_file,
                merge=args.merge,
                filter_size=args.filter_size,
                cutoff=args.cutoff,
            )
        if args.subcommand == "ccf":
            from .universe.ccf_universe import ccf_universe

            ccf_universe(
                cove=args.coverage_folder,
                cove_prefix=args.coverage_prefix,
                file_out=args.output_file,
            )

    if args.command == "scembed":
        _LOGGER.info("Running scembed")
        pass
        # scembed_main(test_args)
    if args.command == "eval":
        if args.subcommand == "gdst":
            from gitk.eval.gdst import get_gds

            gds = get_gds(args.model_path, args.embed_type, args.num_samples)
            print(gds)
        if args.subcommand == "npt":
            from gitk.eval.npt import get_snpr

            npt = get_snpr(
                args.model_path,
                args.embed_type,
                args.K,
                args.num_samples,
                resolution=args.K,
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
                threshold=args.threshold,
            )
            print(scctss)
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
