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
        if args.subcommand == "local":
            from .bbclient.cli import process_local_bed_data

            _LOGGER.info(f"Subcommand: {args.subcommand}")
            process_local_bed_data(args)

        if args.subcommand == "bedset":
            from .bbclient.cli import download_bedset

            _LOGGER.info(f"Subcommand: {args.subcommand}")
            download_bedset(args)

        if args.subcommand == "region":
            from .bbclient.cli import download_and_process_bed_region

            _LOGGER.info(f"Subcommand: {args.subcommand}")
            download_and_process_bed_region(args)

        if args.subcommand == "identifier":
            from .bbclient.cli import process_identifier

            _LOGGER.info(f"Subcommand: {args.subcommand}")
            process_identifier(args)

        if args.subcommand == "identifiers":
            from .bbclient.cli import process_identifiers

            _LOGGER.info(f"Subcommand: {args.subcommand}")
            process_identifiers(args)

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
            import os
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

    return


if __name__ == "__main__":
    main()
