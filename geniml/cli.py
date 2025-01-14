import logging
import os
import sys
from typing import Dict

import logmuse
from ubiquerg import VersionInHelpParser

from ._version import __version__
from .assess.cli import build_subparser as assess_subparser
from .bbclient.cli import build_subparser as bbclient_subparser
from .bedspace.cli import build_argparser as bedspace_subparser
from .eval.cli import build_subparser as eval_subparser
from .likelihood.cli import build_subparser as likelihood_subparser
from .region2vec.cli import build_subparser as region2vec_subparser
from .scembed.argparser import build_argparser as scembed_subparser
from .tokenization.cli import build_subparser as tokenization_subparser
from .universe.cli import build_mode_parser as universe_subparser


def print_inspect_beds(bb_cache_folder) -> None:
    """
    Print the bed files in the cache folder.

    :param bb_cache_folder: Cache folder path
    """
    from rich.console import Console
    from rich.table import Table

    from .bbclient import BBClient

    _LOGGER.info(f"Bedfiles directory:")
    bbc = BBClient(cache_folder=bb_cache_folder)
    result = bbc.list_beds()

    console = Console()

    # Create a Table
    table = Table(title="Cached Bedfiles")

    # Add columns
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Path", style="magenta")

    # Add rows from the dictionary
    for id, path in result.items():
        table.add_row(str(id), path)

    # Print the table
    console.print(table)

    console.print(f"Number of bed files: {len(result)}")


def print_inspect_bedsets(bb_cache_folder) -> None:
    """
    Print the bed sets in the cache folder.

    :param bb_cache_folder: Cache folder path
    """
    from rich.console import Console
    from rich.table import Table

    from .bbclient import BBClient

    _LOGGER.info(f"Bedsets directory:")
    bbc = BBClient(cache_folder=bb_cache_folder)
    result = bbc.list_bedsets()

    console = Console()

    # Create a Table
    table = Table(title="Cached Bedsets")

    # Add columns
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Path", style="magenta")

    # Add rows from the dictionary
    for id, path in result.items():
        table.add_row(str(id), path)

    # Print the table
    console.print(table)

    console.print(f"Number of bed sets: {len(result)}")


def build_argparser():
    """
    Builds argument parser.

    :return argparse.ArgumentParser: Argument parser
    """

    banner = "%(prog)s - Genomic Interval toolkit"
    additional_description = "\nhttps://geniml.databio.org"

    parser = VersionInHelpParser(
        prog="geniml",
        version=f"{__version__}",
        description=banner,
        epilog=additional_description,
    )

    # Individual subcommands
    msg_by_cmd = {
        "assess-universe": "Assess a universe",
        "bbclient": "Client for the BEDbase server",
        "bedspace": "Coembed regionsets (bed files) and labels",
        "build-universe": "Build a consensus peak set using one of provided model",
        "eval": "Evaluate a set of region embeddings",
        "lh": "Make likelihood model",
        "region2vec": "Train a region2vec model",
        "scembed": "Embed single-cell data as region vectors",
        "tokenize": "Tokenize BED files",
    }

    sp = parser.add_subparsers(dest="command")
    subparsers: Dict[str, VersionInHelpParser] = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)

    # build up subparsers for modules
    subparsers["assess-universe"] = assess_subparser(subparsers["assess-universe"])
    subparsers["bbclient"] = bbclient_subparser(subparsers["bbclient"])
    subparsers["bedspace"] = bedspace_subparser(subparsers["bedspace"])
    subparsers["build-universe"] = universe_subparser(subparsers["build-universe"])
    subparsers["eval"] = eval_subparser(subparsers["eval"])
    subparsers["lh"] = likelihood_subparser(subparsers["lh"])
    subparsers["region2vec"] = region2vec_subparser(subparsers["region2vec"])
    subparsers["scembed"] = scembed_subparser(subparsers["scembed"])
    subparsers["tokenize"] = tokenization_subparser(subparsers["tokenize"])

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

    if args.command == "assess-universe":
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

    if args.command == "bbclient":
        if args.subcommand in [
            "cache-bed",
            "cache-tokens",
            "cache-bedset",
            "seek",
            "inspect-bedfiles",
            "inspect-bedsets",
            "rm",
        ]:
            _LOGGER.info(f"Subcommand: {args.subcommand}")
            from .bbclient import BBClient

            bbc = BBClient(cache_folder=args.cache_folder)

        else:
            # if no subcommand, print help format of bbclient subparser
            # from https://stackoverflow.com/a/20096044/23054783
            import argparse

            subparsers_actions = [
                action
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ]
            # there will probably only be one subparser_action,
            # but better safe than sorry
            for subparsers_action in subparsers_actions:
                # get all subparsers and print help
                for choice, subparser in subparsers_action.choices.items():
                    if choice == "bbclient":
                        print(subparser.format_help())
                        sys.exit(1)
        if args.subcommand == "cache-bed":
            # if input is a BED file path
            if os.path.exists(args.identifier[0]):
                identifier = bbc.add_bed_to_cache(args.identifier[0])
                _LOGGER.info(f"BED file {identifier} has been cached")
            else:
                bbc.load_bed(args.identifier[0])

        if args.subcommand == "cache-tokens":
            bbc.add_bed_tokens_to_cache(args.bed_id[0], args.universe_id[0])

        if args.subcommand == "cache-bedset":
            if os.path.isdir(args.identifier[0]):
                from .io import BedSet

                bedset = BedSet(
                    [
                        os.path.join(args.identifier[0], file_name)
                        for file_name in os.listdir(args.identifier[0])
                    ]
                )
                bbc.add_bedset_to_cache(bedset)
                _LOGGER.info(f"BED set {bedset.compute_bedset_identifier()} has been cached")

            else:
                bbc.load_bedset(args.identifier[0])

        if args.subcommand == "seek":
            handler = logging.StreamHandler(sys.stdout)
            _LOGGER.addHandler(handler)
            _LOGGER.info(bbc.seek(args.identifier[0]))

        if args.subcommand == "inspect-bedfiles":
            print_inspect_beds(args.cache_folder)

        if args.subcommand == "inspect-bedsets":
            print_inspect_bedsets(args.cache_folder)

        if args.subcommand == "rm":
            file_path = bbc.seek(args.identifier[0])
            bbc._remove(file_path)
            _LOGGER.info(f"{file_path} is removed")

    if args.command == "build-universe":
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

    if args.command == "region2vec":
        from .region2vec import region2vec

        region2vec(
            token_folder=args.token_folder,
            save_dir=args.save_dir,
            num_shufflings=args.num_shuffle,
            num_processes=args.nworkers,
            embedding_dim=args.embed_dim,
            context_win_size=args.context_len,
            save_freq=args.save_freq,
            resume_path=args.resume,
            train_alg=args.train_alg,
            min_count=args.min_count,
            neg_samples=args.neg_samples,
            init_lr=args.init_lr,
            min_lr=args.min_lr,
            lr_scheduler=args.lr_mode,
            milestones=args.milestones,
            seed=args.seed,
        )
    if args.command == "tokenize":
        from .tokenization import hard_tokenization

        hard_tokenization(
            src_folder=args.data_folder,
            dst_folder=args.token_folder,
            universe_file=args.universe,
            fraction=args.fraction,
            num_workers=args.nworkers,
            bedtools_path=args.bedtools_path,
        )
    if args.command == "eval":
        if args.subcommand == "gdst":
            from geniml.eval.gdst import get_gdst_score

            gdst_score = get_gdst_score(
                args.model_path, args.embed_type, args.num_samples, args.seed
            )
            print(gdst_score)
        if args.subcommand == "npt":
            from geniml.eval.npt import get_npt_score

            npt_score = get_npt_score(
                args.model_path,
                args.embed_type,
                args.K,
                args.num_samples,
                args.seed,
                args.K,
                num_workers=args.num_workers,
            )
            print(npt_score["SNPR"][0])
        if args.subcommand == "ctt":
            from geniml.eval.ctt import get_ctt_score

            ctt_score = get_ctt_score(
                args.model_path,
                args.embed_type,
                args.seed,
                args.num_samples,
                args.num_workers,
            )

            print(ctt_score)
        if args.subcommand == "rct":
            from geniml.eval.rct import get_rct_score

            rct_score = get_rct_score(
                args.model_path,
                args.embed_type,
                args.bin_path,
                args.out_dim,
                args.cv_num,
                args.seed,
                args.num_workers,
            )
            print(rct_score)
        if args.subcommand == "bin-gen":
            import glob
            import pickle

            from geniml.eval.utils import get_bin_embeddings

            if os.path.exists(args.file_name):
                print(f"{args.file_name} exists!")
                return
            token_files = glob.glob(os.path.join(args.token_folder, "*"))
            bin_embed = get_bin_embeddings(args.universe, token_files)
            os.makedirs(os.path.dirname(args.file_name), exist_ok=True)
            with open(args.file_name, "wb") as f:
                pickle.dump(bin_embed, f)
            print(f"binary embeddings saved to {args.file_name}")

    if args.command == "bedspace":
        _LOGGER.info(f"Subcommand: {args.subcommand}")
        if args.subcommand == "preprocess":
            from .bedspace.preprocess import main as preprocess
            _LOGGER.info("Preprocessing data for training")
            preprocess(
                data_path=args.input,
                metadata=args.metadata,
                universe=args.universe,
                output=args.output,
                labels=args.labels
            )
        elif args.subcommand == "train":
            from .bedspace.train import main as train
            _LOGGER.info("Training the StarSpace model")
            train(
                path_to_starspace=args.path_to_starsapce,
                input=args.input,
                num_epochs=args.num_epochs,
                dim=args.dim,
                learning_rate=args.learning_rate,
                output=args.output
            )

        elif args.subcommand == "distances":
            from .bedspace.distances import main as distances
            _LOGGER.info("Computing distances between regionsets and labels")
            distances(
                input=args.input,
                path_to_starspace=args.path_to_starsapce,
                metadata_train=args.metadata_train,
                metadata_test=args.metadata_test,
                universe=args.universe,
                project_name=args.project_name,
                files=args.files,
                labels=args.labels,
                output=args.output,
                threshold=args.threshold
            )

        elif args.subcommand == "search":
            from .bedspace.search import main as search
            _LOGGER.info("Searching for regionsets similar to a query")
            search(
                query=args.query,
                search_type=args.type,
                distances=args.distances,
                num_results=args.num_results
            )

        else:
            _LOGGER.error(f"Unknown subcommand: {args.subcommand}")
            parser.print_help()
            sys.exit(1)
       
    return


if __name__ == "__main__":
    main()
