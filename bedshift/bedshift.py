""" Computing configuration representation """

from __future__ import division, print_function
import argparse
import logmuse
import logging
import os
import pandas as pd
import numpy as np
import random
from . import __version__

_LOGGER = logging.getLogger(__name__)
chroms = ['chr'+str(num) for num in list(range(1, 23)) + ['X', 'Y']]
chrom_lens = [247249719, 242951149, 199501827, 191273063, 180857866, 170899992, 158821424, 146274826, 140273252, 135374737, 134452384, 132349534, 114142980, 106368585, 100338915, 88827254, 78774742, 76117153, 63811651, 62435964, 46944323, 49691432, 154913754, 57772954]


class _VersionInHelpParser(argparse.ArgumentParser):
    def format_help(self):
        """ Add version information to help text. """
        return "version: {}\n".format(__version__) + \
               super(_VersionInHelpParser, self).format_help()


def build_argparser():
    """
    Builds argument parser.

    :return argparse.ArgumentParser
    """

    banner = "%(prog)s - randomize BED files"
    additional_description = "\n..."

    parser = _VersionInHelpParser(
            description=banner,
            epilog=additional_description)

    parser.add_argument(
            "-V", "--version",
            action="version",
            version="%(prog)s {v}".format(v=__version__))

    parser.add_argument(
            "-b", "--bedfile", required=True,
            help="File path to bed file.")

    parser.add_argument(
            "-d", "--droprate", type=float, default=0.0,
            help="Droprate parameter")

    parser.add_argument(
            "-a", "--addrate", type=float, default=0.0,
            help="Addrate parameter")

    parser.add_argument(
            "--addmean", type=float, default=320.0,
            help="Mean add region length")

    parser.add_argument(
            "--addstdev", type=float, default=30.0,
            help="Stdev add length")

    parser.add_argument(
            "-p", "--shiftrate", type=float, default=0.0,
            help="Shift prob.")

    parser.add_argument(
            "-m", "--mean", type=float, default=0.0,
            help="Mean shift")

    parser.add_argument(
            "-s", "--stdev", type=float, default=150.0,
            help="Stdev shift")

    parser.add_argument(
            "-c", "--cutrate", type=float, default=0.0,
            help="Cut prob.")

    parser.add_argument(
            "--mergerate", type=float, default=0.0,
            help="Merge prob.")


    return parser




def shift(df, row, mean, stdev):
    # row = random.randint(0, df.shape[0] - 1)

    theshift = int(np.random.normal(mean, stdev))

    start = df.iloc[row][1]
    end = df.iloc[row][2]

    '''
    if random.random() < 0.5: # shift the start value
        df.at[row, 1] = start + theshift
    else: # shift the end value
        df.at[row, 2] = end + theshift

    if end < start or start > end:
        return drop(df, row)
    '''

    df.at[row, 1] = start + theshift
    df.at[row, 2] = end + theshift
    df.at[row, 3] = 1.0

    return df


def drop(df, row=None):
    if not row or row >= df.shape[0]:
        row = random.randint(0, df.shape[0] - 1)
    df = df.drop(row, axis=0)
    return df


def cut(df, row, mean, stdev):
    # row = random.randint(0, df.shape[0] - 1)

    chrom = df.iloc[row][0]
    start = df.iloc[row][1]
    end = df.iloc[row][2]

    if end-start < 0:
        print('ERROR: the end value of a region is less than the start value')
        exit(1)
    thecut = int(np.random.normal((start+end)/2, (end - start)/4))
    if thecut <= start:
        thecut = start + 1
    if thecut >= end:
        thecut = end - 1

    df = drop(df, row)
    new_segs = pd.DataFrame([[chrom, start, thecut, 2.0], [chrom, thecut, end, 2.0]])
    df = df.append(new_segs, ignore_index=True)

    # adjust the cut regions using the shift function
    df = shift(df, df.shape[0]-1, mean, stdev)
    return shift(df, df.shape[0]-2, mean, stdev)


def add(df, mean, stdev):
    index = random.randrange(len(chroms))
    chrom = chroms[index]
    start = random.randint(1, chrom_lens[index])
    end = min(start + max(int(np.random.normal(mean, stdev)), 20), chrom_lens[index])
    return df.append(pd.DataFrame([[chrom, start, end, 3.0]]), ignore_index=True)


def merge(df, row):
    tempdf = df.sort_values([0, 1]).reset_index(drop=True)

    if row >= tempdf.shape[0] - 1:
        print("df bounds exceeded")
        return df
    # ensure the selected two regions are the same chromosome
    # row = random.randint(0, df.shape[0] - 2)
    while tempdf.iloc[row][0] != tempdf.iloc[row+1][0]:
        row = random.randint(0, df.shape[0] - 2)

    return df.append([[tempdf.iloc[row][0], tempdf.iloc[row][1], tempdf.iloc[row+1][2], 4.0]], ignore_index=True)




def main():
    """ Primary workflow """

    parser = logmuse.add_logging_options(build_argparser())
    args, remaining_args = parser.parse_known_args()
    global _LOGGER
    _LOGGER = logmuse.logger_via_cli(args)

    _LOGGER.info("welcome to bedshift")
    _LOGGER.info("Shifting file: '{}'".format(args.bedfile))
    msg = """Params:
  droprate: {droprate}
  addrate: {addrate}
  addmean: {addmean}
  addstdev: {addstdev}
  shiftrate: {shiftrate}
  mean: {mean}
  stdev: {stdev}
  cutrate: {cutrate}
  mergerate: {mergerate}
  """

    _LOGGER.info(msg.format(
        droprate=args.droprate,
        addrate=args.addrate,
        addmean=args.addmean,
        addstdev=args.addstdev,
        shiftrate=args.shiftrate,
        mean=args.mean,
        stdev=args.stdev,
        cutrate=args.cutrate,
        mergerate=args.mergerate))

    if not args.bedfile:
        parser.print_help()
        _LOGGER.error("No bedfile given")
        sys.exit(1)

    df = pd.read_csv(args.bedfile, sep='\t', header=None, usecols=[0,1,2])
    df[3] = 0
    rows = df.shape[0]
    for i in range(rows):
        if random.random() < args.droprate:
            df = drop(df, i)
        if random.random() < args.addrate:
            df = add(df, args.addmean, args.addstdev)
        if random.random() < args.shiftrate:
            df = shift(df, i, args.mean, args.stdev)
        if random.random() < args.cutrate:
            df = cut(df, i, args.mean, args.stdev)
        if random.random() < args.mergerate:
            df = merge(df, i)

        '''
        if 0 <= x < 0.2:
            df = shift(df)
        elif 0.2 <= x < 0.4:
            df = delete(df)
        elif 0.4 <= x < 0.6:
            df = cut(df)
        elif 0.6 <= x < 0.8:
            df = create(df)
        else:
            df = merge(df)
        '''

    df.to_csv('changed_{}'.format(os.path.basename(args.bedfile)), sep='\t', header=False, index=False)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _LOGGER.error("Program canceled by user!")
        sys.exit(1)


