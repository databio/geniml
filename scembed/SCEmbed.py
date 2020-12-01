#!/usr/bin/env python

__author__ = ["Erfaneh Gharavi"]
__version__ = "0.0.1"

import os
import argparse
from singlecellEmbedding import singlecellEmbedding

singlecellEmbeddingmodel = singlecellEmbedding()


def parse_arguments():
    """
    Parse command-line arguments passed to the pipeline.
    """
    # Argument Parsing from yaml file
    ###########################################################################
    parser = argparse.ArgumentParser(
        description='SCEmbed version ' + __version__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Pipeline-specific arguments
    parser.add_argument("-i", "--input", default=None, type=str,
                        required=True,
                        help="Path to region counts matrix file.")

    parser.add_argument("-o", "--output", default=None, type=str.lower,
                        required=True,
                        help="Path to output directory to store results.")

    parser.add_argument("--model", dest="model", default=None,
                        help="Path to previously built Word2Vec model.")

    parser.add_argument("--nothreads", dest="nothreads", default=1,
                        help="Number of available processors for  "
                             "Word2Vec training.")

    parser.add_argument("--nocells", dest="nocells", default=5,
                        help="Minimum number of cells with a shared region "
                             "for that region to be included.")
                       
    parser.add_argument("--noreads", dest="noreads", default=2,
                        help="Minimum number of reads that overlap a region "
                             "for that region to be included.")

    parser.add_argument("--dimension", dest="dimension", default=10,
                        help="Number of dimensions to train the word2vec "
                             "model.")

    parser.add_argument("--min-count", dest="min_count", default=100,
                        help="Minimum count for Word2Vec model.")

    parser.add_argument("--shuffle-repeat", dest="shuffle_repeat",
                        default=5,
                        help="Number of times to shuffle the document to "
                             "generate date for Word2Vec.")

    parser.add_argument("--umap-nneighbours", dest="umap_nneighbours",
                        default=100,
                        help="Number of neighbors for UMAP plot.")

    parser.add_argument("--window-size", dest="window_size", default=100,
                        help="Word2Vec window size.")

    parser.add_argument("-V", "--version", action="version",
                        version="%(prog)s {v}".format(v=__version__))

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        raise SystemExit

    return args


# MAIN
args = parse_arguments()

try:
    os.makedirs(args.output, exist_ok = True)
except OSError as error: 
    print(error) 

#output file names
model_filename = 'word2vec_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}.model'.format(args.nocells, args.noreads, args.dimension, args.window_size , args.min_count, args.shuffle_repeat, args.umap_nneighbours)
model_path = os.path.join(args.output, model_filename)

plot_filename = 'umapplot_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}.svg'.format(args.nocells, args.noreads, args.dimension, args.window_size, args.min_count, args.shuffle_repeat, args.umap_nneighbours)
fig_dir = os.path.join(args.output, "figs")
try:
    os.makedirs(fig_dir, exist_ok = True)
except OSError as error: 
    print(error) 
plot_path = os.path.join(fig_dir, plot_filename)

singlecellEmbeddingmodel.main(path_file=args.input,
                              nocells=args.nocells,
                              noreads=args.noreads,
                              w2v_model=args.model,
                              shuffle_repeat=args.shuffle_repeat,
                              window_size=args.window_size,
                              dimension=args.dimension,
                              min_count=args.min_count,
                              threads=args.nothreads,
                              umap_nneighbours=args.umap_nneighbours,
                              model_filename=model_path,
                              plot_filename=plot_path)