#!/usr/bin/env python

__author__ = ["Jason Smith"]
__version__ = "0.0.1"

import argparse
import scipy.io, scipy.sparse
import pandas as pd
import csv
import pathlib

def parse_arguments():
    """
    Parse command-line arguments passed to the pipeline.
    """
    # Argument Parsing
    ###########################################################################
    parser = argparse.ArgumentParser(
        description='Convert dense matrix to MatrixMarket format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Pipeline-specific arguments
    parser.add_argument("-i", "--input", default=None, type=str,
                        required=True,
                        help="Path to dense matrix.")

    parser.add_argument("-V", "--version", action="version",
                        version="%(prog)s {v}".format(v=__version__))

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        raise SystemExit

    return args


# MAIN
args = parse_arguments()

path_file = args.input
c_filename = pathlib.Path(path_file).stem + "_coords.tsv"
names_filename = pathlib.Path(path_file).stem + "_names.tsv"
mtx_filename = pathlib.Path(path_file).stem + ".mtx"

col_names = pd.read_csv(path_file, nrows=0, sep="\t").columns
types_dict = {'chr': str, 'start': int, 'end': int}
types_dict.update({col: 'int8' for col in col_names if col not in types_dict})

data = pd.read_csv(path_file, sep="\t", dtype=types_dict,
                   keep_default_na=False, error_bad_lines=False)

coords = data[['chr', 'start', 'end']]
# TODO: need to drop header
coords.to_csv(c_filename, sep='\t', index=False)
names = list(data.iloc[0:, 4:len(data.columns):1].columns)
with open(names_filename, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    for val in names:
        writer.writerow([val])

mat = data.drop(columns=['chr', 'start', 'end'])

scipy.io.mmwrite(mtx_filename, scipy.sparse.csr_matrix(mat))