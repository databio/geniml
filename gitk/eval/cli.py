def build_subparser_gdst(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="path to a Region2Vec model or a Base model",
    )
    parser.add_argument("--embed-type", required=True, type=str, help="region2vec or base")
    parser.add_argument(
        "--num-samples",
        default=10000,
        type=int,
        help="number of samples used in calculation",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed",
    )

    return parser


def build_subparser_npt(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="path to a region2vec model or a Base model",
    )
    parser.add_argument("--embed-type", required=True, type=str, help="region2vec or base")
    parser.add_argument("--K", required=True, type=int, help="number of nearest regions")
    parser.add_argument(
        "--num-samples",
        default=1000,
        type=int,
        help="number of samples used in calculation",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--num-workers",
        default=10,
        type=int,
        help="number of parllel processes",
    )
    return parser


def build_subparser_ctt(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="path to a region2vec model or a Base model",
    )
    parser.add_argument("--embed-type", required=True, type=str, help="region2vec or base")

    parser.add_argument(
        "--num-samples",
        default=10000,
        type=int,
        help="number of samples used in calculation",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--num-workers",
        default=10,
        type=int,
        help="number of parllel processes",
    )

    return parser


def build_subparser_rct(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="path to a region2vec model or a Base model",
    )
    parser.add_argument(
        "--bin-path",
        required=True,
        type=str,
        help="path to a set of binary embedding",
    )
    parser.add_argument(
        "--embed-type",
        required=True,
        type=str,
        help="region2vec or base for model-path",
    )

    parser.add_argument(
        "--cv-num",
        default=5,
        type=int,
        help="number of folds in cross-validation",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--out-dim",
        default=-1,
        type=int,
        help="Used when the binary embeddings are very high-dimensional, i.e., there are many training files. Default -1 represents using all the dimensions",
    )
    parser.add_argument(
        "--num-workers",
        default=10,
        type=int,
        help="number of parllel processes",
    )

    return parser


def build_subparser_bingen(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """
    parser.add_argument(
        "--universe",
        required=True,
        type=str,
        help="path to a universe file",
    )
    parser.add_argument(
        "--token-folder",
        required=True,
        type=str,
        help="path to a folder storing tokenized files",
    )
    parser.add_argument(
        "--file-name",
        required=True,
        type=str,
        help="name of the generated binary embeddings",
    )

    return parser


def build_subparser(parser):
    sp = parser.add_subparsers(dest="subcommand")
    msg_by_cmd = {
        "gdst": "Genome distance scaling test",
        "npt": "Neighorhood preserving test",
        "ctt": "Cluster tendency test",
        "rct": "Reconstruction test",
        "bin-gen": "Generate binary embeddings",
    }
    subparsers = {}
    for k, v in msg_by_cmd.items():
        subparsers[k] = sp.add_parser(k, description=v, help=v)
    subparsers["gdst"] = build_subparser_gdst(subparsers["gdst"])
    subparsers["npt"] = build_subparser_npt(subparsers["npt"])
    subparsers["ctt"] = build_subparser_ctt(subparsers["ctt"])
    subparsers["rct"] = build_subparser_rct(subparsers["rct"])
    subparsers["bin-gen"] = build_subparser_bingen(subparsers["bin-gen"])
    return parser
