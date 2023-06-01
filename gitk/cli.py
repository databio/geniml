from typing import Dict
import logmuse
import sys
import os

from ubiquerg import VersionInHelpParser

from .assess.cli import build_mode_parser as assess_subparser
from .eval.cli import build_subparser as eval_subparser
from .region2vec.cli import build_subparser as region2vec_subparser
from .tokenization.cli import build_subparser as tokenization_subparser
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
        "region2vec": "Train a region2vec model",
        "tokenize": "Tokenize BED files",
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
    subparsers["region2vec"] = region2vec_subparser(subparsers["region2vec"])
    subparsers["tokenize"] = tokenization_subparser(subparsers["tokenize"])
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
                args.no_workers,
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
                args.no_workers,
                args.save_to_file,
                args.folder_out,
                args.pref,
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
            from gitk.eval.gdst import get_gdst_score

            gdst_score = get_gdst_score(
                args.model_path, args.embed_type, args.num_samples, args.seed
            )
            print(gdst_score)
        if args.subcommand == "npt":
            from gitk.eval.npt import get_npt_score

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
            from gitk.eval.ctt import get_ctt_score

            ctt_score = get_ctt_score(
                args.model_path,
                args.embed_type,
                args.seed,
                args.num_samples,
                args.num_workers,
            )
            print(ctt_score)
        if args.subcommand == "rct":
            from gitk.eval.rct import get_rct_score

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
            from gitk.eval.utils import get_bin_embeddings
            import glob, pickle

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
