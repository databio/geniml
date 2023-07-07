def build_subparser(parser):
    """
    Builds argument parser.

    :return argparse.ArgumentParser: Argument parser
    """
    parser.add_argument("--token-folder", type=str, help="path to tokenized files")
    parser.add_argument("--num-shuffle", type=int, help="number of shufflings/training epochs")
    parser.add_argument("--embed-dim", type=int, help="embedding dimension")
    parser.add_argument("--context-len", type=int, help="Context window size (half)")
    parser.add_argument("--nworkers", type=int, default=10, help="number of workers")
    parser.add_argument(
        "--save-freq",
        type=int,
        default=-1,
        help="Save a model after the given number of training epochs. If -1, then only save the best and latest models",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="path to the folder that saves the training result",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to a trained model. If specified, the model will be used to initialize the region2vec embeddings",
    )
    parser.add_argument(
        "--train-alg",
        type=str,
        default="cbow",
        help="training algorithm, select from [cbow, skip-gram]",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="threshold for filtering out regions with low frequency in the internal vocabulary",
    )
    parser.add_argument(
        "--neg-samples",
        type=int,
        default=5,
        help="number of noise words in negative sampling, usually between 5-20",
    )
    parser.add_argument("--init-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 200])
    parser.add_argument(
        "--lr-mode",
        type=str,
        default="linear",
        choices=["milestone", "linear"],
        help="type of learning rate scheduler, milestone or linear",
    )
    parser.add_argument(
        "--update-vocab",
        type=str,
        default="once",
        help="[every] update at every epoch; [once] Update once since the vocabulary does not change",
    )
    parser.add_argument("--min-lr", type=float, default=1.0e-6, help="minimum learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser
