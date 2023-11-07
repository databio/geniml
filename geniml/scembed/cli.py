import os
import sys

import logmuse
from gensim.models import Word2Vec

from ._version import __version__
from .argparser import build_argparser
from .const import *
from .main import (
    convert_anndata_to_documents,
    load_scanpy_data,
    shuffle_documents,
    train,
)


def main():
    """MAIN"""
    parser = logmuse.add_logging_options(build_argparser())
    args, _ = parser.parse_known_args()

    if args.input is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    global _LOGGER
    _LOGGER = logmuse.logger_via_cli(args, make_root=True)
    _LOGGER.debug(f"versions: scembed {__version__}")
    _LOGGER.debug(f"Args: {args}")

    # TODO: store parameters in a file, not in the filenames?
    # TODO: load a config file for all the parameters instead of specifying

    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as error:
        _LOGGER.error(error)
        sys.exit(1)

    try:
        os.makedirs(os.path.join(args.output, "figs"), exist_ok=True)
    except OSError as error:
        _LOGGER.error(error)
        sys.exit(1)

    _LOGGER.info("Loading data")

    # load AnnData object
    sc_data = load_scanpy_data(args.input)

    # re-create documents dictionary from AnnData Object
    documents = convert_anndata_to_documents(sc_data)

    if args.model:
        model = Word2Vec.load(args.model)
        _LOGGER.info(f"Loaded {args.model}")
    else:
        model_name = (
            f"_nocells{args.nocells}_noreads{args.noreads}_"
            f"dim{args.dimension}_win{args.window_size}_"
            f"mincount{args.min_count}_"
            f"shuffle{args.shuffle_repeat}.model"
        )
        model_filename = os.path.join(args.output, args.title + model_name)
        _LOGGER.info(f"Shuffling documents")
        shuffeled_documents = shuffle_documents(documents, int(args.shuffle_repeat))
        _LOGGER.info(f"Constructing model")
        model = train(
            shuffeled_documents,
            window_size=int(args.window_size),
            dim=int(args.dimension),
            min_count=int(args.min_count),
            nothreads=int(args.nothreads),
        )
        model.save(model_filename)
        _LOGGER.info(f"Model saved as: {model_filename}")

    _LOGGER.info(f"Number of words in w2v model: {len(model.wv.vocab)}")
    _LOGGER.info(f"Pipeline Complete!")
