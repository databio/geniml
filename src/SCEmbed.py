#!/usr/bin/env python

__author__ = ["Jason Smith", "Erfaneh Gharavi"]
__version__ = "0.1.0"

import os
import argparse
from singlecellEmbedding import singlecellEmbedding

singlecellEmbeddingmodel = singlecellEmbedding()


def parse_arguments():
    """
    Parse command-line arguments passed to the pipeline.
    """
    # Argument Parsing
    ###########################################################################
    parser = argparse.ArgumentParser(
        description='SCEmbed version ' + __version__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Pipeline-specific arguments
    parser.add_argument("--intern", dest="intern",
                        default=False, action='store_true',
                        help="Use string interning to reduce memory use.")

    parser.add_argument("--vaex", dest="vaex",
                        default=False, action='store_true',
                        help="Use vaex method.")

    parser.add_argument("-i", "--input", default=None, type=str,
                        required=True,
                        help="Path to MarketMatrix format count matrix.")

    parser.add_argument("-n", "--names", default=None, type=str,
                        required=False,
                        help="Path to sample/barcodes names in a single "
                             "column tab-delimited format.")

    parser.add_argument("-o", "--output", default=None, type=str,
                        required=True,
                        help="Path to output directory to store results.")

    parser.add_argument("-t", "--title", default='scembed', type=str,
                        required=False,
                        help="Project/run title for naming output.")

    parser.add_argument("--docs", dest="docs", default=None,
                        help="Path to documents dictionary.")

    parser.add_argument("--model", dest="model", default=None,
                        help="Path to Word2Vec model.")

    parser.add_argument("--embed", dest="embed", default=None,
                        help="Path to document embeddings dictionary.")

    parser.add_argument("--nothreads", dest="nothreads", default=1,
                        help="Number of available processors for  "
                             "Word2Vec training.")

    parser.add_argument("--nocells", dest="nocells", default=5,
                        help="Minimum number of cells with a shared region "
                             "for that region to be included.")
                       
    parser.add_argument("--noreads", dest="noreads", default=2,
                        help="Minimum number of reads that overlap a region "
                             "for that region to be included.")

    parser.add_argument("--window-size", dest="window_size", default=100,
                        help="Word2Vec window size.")

    parser.add_argument("--dimension", dest="dimension", default=100,
                        help="Number of dimensions to train the word2vec "
                             "model.")

    parser.add_argument("--min-count", dest="min_count", default=10,
                        help="Minimum count for Word2Vec model.")

    parser.add_argument("--shuffle-repeat", dest="shuffle_repeat",
                        default=5,
                        help="Number of times to shuffle the document to "
                             "generate date for Word2Vec.")

    parser.add_argument("--umap-nneighbors", dest="umap_nneighbours",
                        default=100,
                        help="Number of neighbors for UMAP plot.")

    parser.add_argument("--umap-metric", dest="umap_metric",
                        default='euclidean', help="UMAP distance metric.")

    parser.add_argument("--rasterize", dest="rasterize",
                        default=False, action='store_true',
                        help="Rasterize the UMAP scatter plot to reduce "
                             "space and plot generation time.")

    parser.add_argument("--alt", dest="alt",
                        default=False, action='store_true',
                        help="Use the new chunking method.")

    parser.add_argument("--v2", dest="v2",
                        default=False, action='store_true',
                        help="Use the new chunking 2 doc method.")

    parser.add_argument("--log", dest="loglevel",
                        default='DEBUG', 
                        help="Choose the logging level: "
                             "DEBUG, INFO, WARNING, ERROR.")

    parser.add_argument("-V", "--version", action="version",
                        version="%(prog)s {v}".format(v=__version__))

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        raise SystemExit

    return args


# MAIN
args = parse_arguments()
# TODO: store parameters in a file, not in the filenames?

try:
    #print('Making output directory if necessary')  # DEBUG
    os.makedirs(args.output, exist_ok = True)
except OSError as error: 
    print(error) 

try:
    #print('Making figure output directory if necessary')  # DEBUG
    os.makedirs(os.path.join(args.output, "figs"), exist_ok = True)
except OSError as error: 
    print(error)

singlecellEmbeddingmodel.main(path_file=args.input,
                              names_file=args.names,
                              out_dir=args.output,
                              nocells=args.nocells,
                              noreads=args.noreads,
                              title=args.title,
                              docs_file=args.docs,
                              w2v_model=args.model,
                              embed_file=args.embed,
                              shuffle_repeat=args.shuffle_repeat,
                              window_size=args.window_size,
                              dimension=args.dimension,
                              min_count=args.min_count,
                              threads=args.nothreads,
                              umap_nneighbours=args.umap_nneighbours,
                              umap_metric=args.umap_metric,
                              rasterize=args.rasterize,
                              alt_method=args.alt,
                              v2_method=args.v2,
                              interned=args.intern,
                              use_vaex=args.vaex,
                              loglevel=args.loglevel)