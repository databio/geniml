import logging
import sys
import os
import logmuse
from gensim.models import Word2Vec
import pandas as pd

from .argparser import build_argparser
from ._version import __version__
from .const import *
from .scembed import convert_anndata_to_documents, document_embedding_avg, load_scanpy_data, shuffling, train_Word2Vec, label_preprocessing, UMAP_plot, save_dict, load_dict

def main():
    """ MAIN """  
    parser = logmuse.add_logging_options(build_argparser())
    args, remaining_args = parser.parse_known_args()

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
        os.makedirs(args.output, exist_ok = True)
    except OSError as error: 
        _LOGGER.error(error)
        sys.exit(1)

    try:
        os.makedirs(os.path.join(args.output, "figs"), exist_ok = True)
    except OSError as error: 
        _LOGGER.error(error)
        sys.exit(1)

    _LOGGER.info("Loading data")
    
    if args.docs:
        documents = load_dict(args.docs)
        _LOGGER.info(f"Loaded {args.docs}")
    else:
        # load AnnData object
        sc_data = load_scanpy_data(args.input)

        # re-create documents dictionary
        documents = convert_anndata_to_documents(sc_data)

        docs_filename = os.path.join(args.output, args.title + "_documents.pkl")
        save_dict(documents, docs_filename)
        _LOGGER.info(f'Saved documents as {docs_filename}')

    if args.model:
        model = Word2Vec.load(args.model)
        _LOGGER.info(f"Loaded {args.model}")
    else:
        model_name = (f"_nocells{args.nocells}_noreads{args.noreads}_"
                      f"dim{args.dimension}_win{args.window_size}_"
                      f"mincount{args.min_count}_"
                      f"shuffle{args.shuffle_repeat}.model")
        model_filename = os.path.join(args.output, args.title + model_name)
        _LOGGER.info(f'Shuffling documents')
        shuffeled_documents = shuffling(documents, int(args.shuffle_repeat))
        _LOGGER.info(f'Constructing model')
        model = train_Word2Vec(shuffeled_documents,
                              window_size = int(args.window_size),
                              dim = int(args.dimension),
                              min_count = int(args.min_count),
                              nothreads = int(args.nothreads))
        model.save(model_filename)
        _LOGGER.info(f'Model saved as: {model_filename}')

    _LOGGER.info(f'Number of words in w2v model: {len(model.wv.vocab)}')

    _LOGGER.info(f'Calculate document embeddings.')
    if not args.embed_file:
        embeddings = document_embedding_avg(documents, model)
        embeddings_dictfile = os.path.join(
            args.output, args.title + "_embeddings.pkl")
        save_dict(embeddings, embeddings_dictfile)
        embeddings_csvfile = os.path.join(
            args.output, args.title + "_embeddings.csv")
        (pd.DataFrame.from_dict(data=embeddings, orient='index').
         to_csv(embeddings_csvfile, header=False))
        _LOGGER.info(f'Embeddings file saved as {embeddings_csvfile}')
    else:
        embeddings = load_dict(args.embed_file)
        _LOGGER.info(f"Loaded {args.embed_file}")

    x = pd.DataFrame(embeddings).values
    y = list(embeddings.keys())
    y = label_preprocessing(y, args.label_delimiter)

    _LOGGER.info(f'Generating plot.')
    coordinates_csvfile = os.path.join(
        args.output, args.title + "_xy_coords.csv")
    plot_name = (f"_nocells{args.nocells}_noreads{args.noreads}_"
                 f"dim{args.dimension}_win{args.window_size}_"
                 f"mincount{args.min_count}_"
                 f"shuffle{args.shuffle_repeat}_"
                 f"umap_nneighbours{args.umap_nneighbours}_"
                 f"umap-metric{args.umap_metric}.svg")
    plot_filename = os.path.join(args.output, "figs", plot_name)
    fig = UMAP_plot(x.T, y, args.title, int(args.umap_nneighbours),
                    coordinates_csvfile, args.umap_metric, args.rasterize)
    _LOGGER.info(f'Saving UMAP plot')
    fig.savefig(plot_filename, format = 'svg')
    _LOGGER.info(f'Pipeline Complete!')
