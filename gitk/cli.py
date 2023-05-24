import os.path
from typing import Dict
import logmuse
import sys
import pandas as pd

from ubiquerg import VersionInHelpParser

from .assess.cli import build_subparser as assess_subparser
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
        asses_results = []
        if args.distance:
            from .assess.distance import run_distance

            r_distance = run_distance(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.no_workers,
                False,
                args.folder_out,
                args.pref + "_dist_file_to_universe",
                args.save_each,
                False,
            )
            r_distance.columns = ["file", "median_dist_file_to_universe"]
            print(r_distance)
            asses_results.append(r_distance)

        if args.distance_flexible:
            from .assess.distance import run_distance

            r_distance_flex = run_distance(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.no_workers,
                True,
                args.folder_out,
                args.pref + "_dist_file_to_universe_flex",
                args.save_each,
                False,
            )
            r_distance_flex.columns = ["file", "median_dist_file_to_universe_flex"]
            print(r_distance_flex)
            asses_results.append(r_distance_flex)

        if args.distance_universe_to_file:
            from .assess.distance import run_distance

            r_distance_utf = run_distance(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.no_workers,
                False,
                args.folder_out,
                args.pref + "_dist_universe_to_file",
                args.save_each,
                True,
            )
            r_distance_utf.columns = ["file", "median_dist_universe_to_file"]
            print(r_distance_utf)
            asses_results.append(r_distance_utf)

        if args.distance_flexible_universe_to_file:
            from .assess.distance import run_distance

            r_distance_utf_flex = run_distance(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.no_workers,
                True,
                args.folder_out,
                args.pref + "median_dist_universe_to_file_flex",
                args.save_each,
                True,
            )
            r_distance_utf_flex.columns = ["file", "median_dist_universe_to_file_flex"]
            print(r_distance_utf_flex)
            asses_results.append(r_distance_utf_flex)

        if args.overlap:
            from .assess.intersection import run_intersection

            r_overlap = run_intersection(
                args.raw_data_folder,
                args.file_list,
                args.universe,
                args.no_workers,
            )
            r_overlap.columns = [
                "file",
                "univers/file",
                "file/universe",
                "universe&file",
            ]
            print(r_overlap)
            asses_results.append(r_overlap)
        if len(asses_results) == 0:
            raise Exception("No assessment method was provided")
        if args.save_to_file:
            df = asses_results[0]
            for i in asses_results[1:]:
                df = pd.merge(df, i, on="file")
            df.to_csv(
                os.path.join(args.folder_out, args.pref + "_data.csv"), index=False
            )

    if args.command == "lh":
        _LOGGER.info(f"Subcommand: {args.subcommand}")
        if args.subcommand == "build_model":
            from .likelihood.build_model import main

            main(
                args.model_file,
                args.coverage_folder,
                args.coverage_prefix,
                args.file_no,
                args.force,
            )

        if args.subcommand == "universe_hard":
            from .likelihood.universe_hard import main

            main(
                args.coverage_file, args.fout, args.merge, args.filter_size, args.cutoff
            )

        if args.subcommand == "universe_flexible":
            from .likelihood.universe_flexible import main

            main(
                args.model_file, args.cov_folder, args.coverage_prefix, args.output_file
            )

    if args.command == "hmm":
        from .hmm.hmm import run_hmm_save_bed

        run_hmm_save_bed(
            coverage_folder=args.cov_folder,
            out_file=args.out_file,
            prefix=args.coverage_prefix,
            normalize=args.not_normalize,
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


if __name__ == "__main__":
    main()
