""" Perturb regions in bedfiles """

import argparse
import logging
import os
import sys
import random
import yaml

import logmuse
import pandas as pd
import numpy as np
import pyranges as pr

from bedshift._version import __version__

_LOGGER = logging.getLogger(__name__)

__all__ = ["Bedshift"]


class _VersionInHelpParser(argparse.ArgumentParser):
    def format_help(self):
        """ Add version information to help text. """
        return "version: {}\n".format(__version__) + \
               super(_VersionInHelpParser, self).format_help()


def build_argparser():
    """
    Builds argument parser.

    :return: argparse.ArgumentParser
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
        "-l", "--chrom-lengths", type=str, required=False,
        help="TSV text file with one row per chromosomes indicating chromosome sizes"
        )

    parser.add_argument(
        "-g", "--genome", type=str, required=False,
        help="Refgenie genome identifier (used for chrom sizes).")

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
        "--addfile", type=str, help="Add regions from a bedfile")

    parser.add_argument(
        "--valid_regions", type=str, help="valid regions in which regions can be randomly added")

    parser.add_argument(
        "-s", "--shiftrate", type=float, default=0.0,
        help="Shift probability")

    parser.add_argument(
        "--shiftmean", type=float, default=0.0,
        help="Mean shift")

    parser.add_argument(
        "--shiftstdev", type=float, default=150.0,
        help="Stdev shift")

    parser.add_argument(
        "-c", "--cutrate", type=float, default=0.0,
        help="Cut probability")

    parser.add_argument(
        "-m", "--mergerate", type=float, default=0.0,
        help="Merge probability. WARNING: will likely create regions that are thousands of base pairs long")

    parser.add_argument(
        "--dropfile", type=str, help="Drop regions from a bedfile")

    parser.add_argument(
        "-o", "--outputfile", type=str,
        help="output file name (including extension). if not specified, will default to bedshifted_{originalname}.bed")

    parser.add_argument(
        "-r", "--repeat", type=int, default=1,
        help="the number of times to repeat the operation")

    parser.add_argument(
        "-y", "--yaml_config", type=str,
        help="run yaml configuration file")

    return parser


class Bedshift(object):
    """
    The bedshift object with methods to perturb regions
    """

    def __init__(self, bedfile_path, chrom_sizes=None, delimiter='\t'):
        """
        Read in a .bed file to pandas DataFrame format

        :param str bedfile_path: the path to the BED file
        :param str chrom_sizes: the path to the chrom.sizes file
        :param str delimiter: the delimiter used in the BED file
        """
        self.bedfile_path = bedfile_path
        self.chrom_lens = {}
        if chrom_sizes:
            self._read_chromsizes(chrom_sizes)
        df = self.read_bed(bedfile_path, delimiter=delimiter)
        self.original_regions = df.shape[0]
        self.bed = df.astype({1: 'int64', 2: 'int64', 3: 'int64'}) \
                            .sort_values([0, 1, 2]).reset_index(drop=True)
        self.original_bed = self.bed.copy(deep=True)


    def _read_chromsizes(self, fp):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip().split('\t')
                    chrom = str(line[0])
                    size = int(line[1])
                    self.chrom_lens[chrom] = size
        except FileNotFoundError:
            _LOGGER.error("fasta file path {} invalid".format(fp))
            sys.exit(1)

        total_len = sum(self.chrom_lens.values())
        self.chrom_weights = [chrom_len / total_len for chrom_len in self.chrom_lens.values()]


    def reset_bed(self):
        """
        Reset the stored bedfile to the state before perturbations
        """

        self.bed = self.original_bed.copy(deep=True)


    def _check_rate(self, rate):
        if rate < 0 or rate > 1:
            _LOGGER.error("Rate must be between 0 and 1")
            sys.exit(1)


    def pick_random_chroms(self, n):
        """
        Utility function to pick a random chromosome

        :param str n: the number of random chromosomes to pick
        :return str, float chrom_str, chrom_len: chromosome number and length
        """
        chrom_strs = random.choices(list(self.chrom_lens.keys()), weights=self.chrom_weights, k=n)
        chrom_lens = [self.chrom_lens[chrom_str] for chrom_str in chrom_strs]
        return zip(chrom_strs, chrom_lens)


    def add(self, addrate, addmean, addstdev, valid_bed=None, delimiter='\t'):
        """
        Add regions

        :param float addrate: the rate to add regions
        :param float addmean: the mean length of added regions
        :param float addstdev: the standard deviation of the length of added regions
        :return int: the number of regions added
        """
        if addrate < 0:
            _LOGGER.error("Rate must be greater than or equal to 0")
            sys.exit(1)
        if addrate == 0:
            return 0
        if len(self.chrom_lens) == 0:
            _LOGGER.error("chrom.sizes file must be specified when adding regions")
            sys.exit(1)

        rows = self.bed.shape[0]
        num_add = int(rows * addrate)
        new_regions = {0: [], 1: [], 2: [], 3: []}
        if valid_bed:
            valid_regions = self.read_bed(valid_bed, delimiter)
            valid_regions[3] = valid_regions[2] - valid_regions[1]
            total_bp = valid_regions[3].sum()
            valid_regions[4] = valid_regions[3].apply(lambda x: x / total_bp)
            add_rows = random.choices(list(range(len(valid_regions))), weights=list(valid_regions[4]), k=num_add)
            for row in add_rows:
                data = valid_regions.loc[row]
                chrom = data[0]
                start = random.randint(data[1], data[2])
                end = start + int(np.random.normal(addmean, addstdev))
                new_regions[0].append(chrom)
                new_regions[1].append(start)
                new_regions[2].append(end)
                new_regions[3].append('A')
        else:
            random_chroms = self.pick_random_chroms(num_add)
            for chrom_str, chrom_len in random_chroms:
                start = random.randint(1, chrom_len)
                # ensure chromosome length is not exceeded
                end = min(start + int(np.random.normal(addmean, addstdev)), chrom_len)
                new_regions[0].append(chrom_str)
                new_regions[1].append(start)
                new_regions[2].append(end)
                new_regions[3].append(3)
        self.bed = self.bed.append(pd.DataFrame(new_regions), ignore_index=True)
        return num_add


    def add_from_file(self, fp, addrate, delimiter='\t'):
        """
        Add regions from another bedfile to this perturbed bedfile

        :param float addrate: the rate to add regions
        :param str fp: the filepath to the other bedfile
        :return int: the number of regions added
        """
        if addrate < 0:
            _LOGGER.error("Rate must be greater than or equal to 0")
            sys.exit(1)
        if addrate == 0:
            return 0
        if len(self.chrom_lens) == 0:
            _LOGGER.error("chrom.sizes file must be specified when adding regions")
            sys.exit(1)

        rows = self.bed.shape[0]
        num_add = int(rows * addrate)
        df = self.read_bed(fp, delimiter=delimiter)
        if num_add > df.shape[0]:
            num_add = df.shape[0]
        add_rows = random.sample(list(range(df.shape[0])), num_add)
        add_df = df.loc[add_rows].reset_index(drop=True)
        add_df[3] = pd.Series([3] * add_df.shape[0])
        self.bed = self.bed.append(add_df, ignore_index=True)
        return num_add


    def shift(self, shiftrate, shiftmean, shiftstdev):
        """
        Shift regions

        :param float shiftrate: the rate to shift regions (both the start and end are shifted by the same amount)
        :param float shiftmean: the mean shift distance
        :param float shiftstdev: the standard deviation of the shift distance
        :return int: the number of regions shifted
        """
        self._check_rate(shiftrate)
        if shiftrate == 0:
            return 0
        if len(self.chrom_lens) == 0:
            _LOGGER.error("chrom.sizes file must be specified when shifting regions")
            sys.exit(1)

        rows = self.bed.shape[0]
        shift_rows = random.sample(list(range(rows)), int(rows * shiftrate))
        new_row_list = []
        to_drop = []
        num_shifted = 0
        invalid_shifted = 0
        for row in shift_rows:
            drop_row, new_region = self._shift(row, shiftmean, shiftstdev) # shifted rows display a 1
            if drop_row is not None and new_region is not None:
                num_shifted += 1
                new_row_list.append(new_region)
                to_drop.append(drop_row)
            else:
                invalid_shifted += 1
        self.bed = self.bed.drop(to_drop)
        self.bed = self.bed.append(new_row_list, ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        if invalid_shifted > 0:
            _LOGGER.warning(f"{invalid_shifted} regions were prevented from being shifted outside of chromosome boundaries. Reported regions shifted will be less than expected.")
        return num_shifted


    def _shift(self, row, mean, stdev):
        theshift = int(np.random.normal(mean, stdev))

        chrom = str(self.bed.loc[row][0])
        start = self.bed.loc[row][1]
        end = self.bed.loc[row][2]

        if start + theshift < 0 or end + theshift > self.chrom_lens[chrom]:
            # check if the region is shifted out of chromosome length bounds
            return None, None

        return row, {0: chrom, 1: start + theshift, 2: end + theshift, 3: 1}


    def cut(self, cutrate):
        """
        Cut regions to create two new regions

        :param float cutrate: the rate to cut regions into two separate regions
        :return int: the number of regions cut
        """
        self._check_rate(cutrate)
        if cutrate == 0:
            return 0
        rows = self.bed.shape[0]
        cut_rows = random.sample(list(range(rows)), int(rows * cutrate))
        new_row_list = []
        to_drop = []
        for row in cut_rows:
            drop_row, new_regions = self._cut(row) # cut rows display a 2
            new_row_list.extend(new_regions)
            to_drop.append(drop_row)
        self.bed = self.bed.drop(to_drop)
        self.bed = self.bed.append(new_row_list, ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        return len(cut_rows)


    def _cut(self, row):
        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row][2]

        # choose where to cut the region
        thecut = (start + end) // 2 # int(np.random.normal((start+end)/2, (end - start)/6))
        if thecut <= start:
            thecut = start + 10
        if thecut >= end:
            thecut = end - 10

        ''' may add in later, this makes the api confusing!
        # adjust the cut regions using the shift function
        new_segs = self.__shift(new_segs, 0, meanshift, stdevshift)
        new_segs = self.__shift(new_segs, 1, meanshift, stdevshift)
        '''

        return row, [{0: chrom, 1: start, 2: thecut, 3: 2}, {0: chrom, 1: thecut, 2: end, 3: 2}]


    def merge(self, mergerate):
        """
        Merge two regions into one new region

        :param float mergerate: the rate to merge two regions into one
        :return int: number of regions merged
        """

        self._check_rate(mergerate)
        if mergerate == 0:
            return 0
        rows = self.bed.shape[0]
        merge_rows = random.sample(list(range(rows)), int(rows * mergerate))
        to_add = []
        to_drop = []
        for row in merge_rows:
            drop_row, add_row = self._merge(row)
            if add_row:
                to_add.append(add_row)
            to_drop.extend(drop_row)
        self.bed = self.bed.drop(to_drop)
        self.bed = self.bed.append(to_add, ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        return len(merge_rows)


    def _merge(self, row):
        # check if the regions being merged are on the same chromosome
        if row + 1 not in self.bed.index or self.bed.loc[row][0] != self.bed.loc[row+1][0]:
            return [], None

        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row+1][2]
        return [row, row+1], {0: chrom, 1: start, 2: end, 3: 4}


    def drop(self, droprate):
        """
        Drop regions

        :param float droprate: the rate to drop/remove regions
        :return int: the number of rows dropped
        """
        self._check_rate(droprate)
        if droprate == 0:
            return 0
        rows = self.bed.shape[0]
        drop_rows = random.sample(list(range(rows)), int(rows * droprate))
        self.bed = self.bed.drop(drop_rows)
        self.bed = self.bed.reset_index(drop=True)
        return len(drop_rows)

    def _find_overlap(self, fp):
        """
        find intersecting regions between the reference bedfile and the comparison file provided in the yaml config file

        :param str fp: the filepath to the other bedfile containing regions to be compared to the reference bedfile
        :return dataframe intersection: the dataframe consisting of matching regions
        """
        reference_pr = self.read_bed(self.bedfile_path)
        comparison_pr = self.read_bed(fp)
        reference_pr.columns = ['Chromosome', 'Start', 'End', 'modifications']
        comparison_pr.columns = ['Chromosome', 'Start', 'End', 'modifications']
        reference_pr = pr.PyRanges(reference_pr)
        comparison_pr = pr.PyRanges(comparison_pr)
        try:
            intersection = reference_pr.overlap(comparison_pr, how='first').as_df()
            intersection = intersection.drop(['modifications'], axis=1)
            intersection.columns = ['chrom', 'start', 'end']
        except ValueError:
            print("No interection found between two files.")
            sys.exit(1)
        return intersection


    def drop_from_file(self, fp, droprate, delimiter='\t'):
        """
        drop regions from another bedfile to this perturbed bedfile

        :param float droprate: the rate to drop regions
        :param str fp: the filepath to the other bedfile containing regions to be dropped
        :return int: the number of regions dropped
        """
        if droprate < 0:
            _LOGGER.error("Rate must be greater than or equal to 0")
            sys.exit(1)
        if droprate == 0:
            return 0

        rows = self.bed.shape[0]
        num_drop = int(rows * droprate)
        drop_bed = self.read_bed(fp, delimiter=delimiter)
        drop_rows = drop_bed.shape[0]

        if num_drop >= drop_rows:
            print("Number of regions to be dropped ({}) is larger than the provided bedfile size ({}). Dropping {} regions.".format(num_drop, drop_rows, drop_rows))
            num_drop = drop_rows
        intersect_regions = self._find_overlap(fp)
        rows2drop = random.sample(list(range(len(intersect_regions))), num_drop)

        self.bed = self.bed.drop(intersect_regions.index[rows2drop]).reset_index(drop=True)
        return num_drop


    def all_perturbations(self,
                          addrate=0.0, addmean=320.0, addstdev=30.0,
                          addfile=None, valid_regions=None,
                          shiftrate=0.0, shiftmean=0.0, shiftstdev=150.0,
                          cutrate=0.0,
                          mergerate=0.0,
                          droprate=0.0,
                          dropfile=None,
                          yaml=None,
                          bedshifter=None):
        '''
        Perform all five perturbations in the order of shift, add, cut, merge, drop.

        :param float addrate: the rate (as a proportion of the total number of regions) to add regions
        :param float addmean: the mean length of added regions
        :param float addstdev: the standard deviation of the length of added regions
        :param string addfile: the file containing regions to be added
        :param string valid_regions: the file containing regions where new regions can be added
        :param float shiftrate: the rate to shift regions (both the start and end are shifted by the same amount)
        :param float shiftmean: the mean shift distance
        :param float shiftstdev: the standard deviation of the shift distance
        :param float cutrate: the rate to cut regions into two separate regions
        :param float mergerate: the rate to merge two regions into one
        :param float droprate: the rate to drop/remove regions
        :param string dropfile: the file containing regions to be dropped
        :param string yaml: the yaml_config filepath
        :param string bedshifter: Bedshift instance
        :return int: the number of total regions perturbed
        '''

        n = 0
        n += self.shift(shiftrate, shiftmean, shiftstdev)
        if addfile:
            n += self.add_from_file(addfile, addrate)
        else:
            n += self.add(addrate, addmean, addstdev, valid_regions)
        n += self.cut(cutrate)
        n += self.merge(mergerate)
        if dropfile:
            n += self.drop_from_file(dropfile, droprate)
        else:
            n +=self.drop(droprate)
        if yaml:
            n += self.handle_yaml(bedshifter, yaml)
        return n


    def to_bed(self, outfile_name):
        """
        Write a pandas dataframe back into BED file format

        :param str outfile_name: The name of the output BED file
        """
        self.bed.sort_values([0,1,2], inplace=True)
        self.bed.to_csv(outfile_name, sep='\t', header=False, index=False, float_format='%.0f')
        print('The output bedfile located in {} has {} regions. The original bedfile had {} regions.' \
              .format(outfile_name, self.bed.shape[0], self.original_regions))


    def read_bed(self, bedfile_path, delimiter='\t'):
        """
        Read a BED file into pandas dataframe

        :param str bedfile_path: The path to the BED file
        """
        try:
            df = pd.read_csv(bedfile_path, sep=delimiter, header=None, usecols=[0,1,2], engine='python')
        except FileNotFoundError:
            _LOGGER.error("BED file path {} invalid".format(bedfile_path))
            sys.exit(1)
        except:
            _LOGGER.error("file {} could not be read".format(bedfile_path))
            sys.exit(1)

        # if there is 'chrom', 'start', 'stop' in the table, move them to header
        if not str(df.iloc[0, 1]).isdigit():
            df.columns = df.iloc[0]
            df = df[1:]

        df[3] = 0 # column indicating which modifications were made
        return df


    def _print_sample_config(self):
        """
        bedshift_operations:
          - add:
            rate: 0.1
            mean: 100
            stdev: 20
          - add_from_file:
            file: tests/test.bed
            rate: 0.1
            delimiter: \t
          - drop_from_file:
            file: tests/test.bed
            rate: 0.1
          - add_from_file:
            file: tests/test.bed
            rate: 0.2
          - cut:
            rate: 0.2
          - shift:
            rate: 0.3
            mean: 100
            stdev: 200
          - merge:
            rate: 0.15
          - drop:
            rate: 0.30
        """
        print(self._print_sample_config.__doc__)
        print("No changes made.")


    def _read_from_yaml(self, fp):
        """
        Loads yaml config data

        :param float fp: the path to the configuration file
        :return int: loaded yaml data
        """
        with open(fp, "r") as yaml_file:
            config_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        print("Loaded configuration settings from {}".format(fp))
        return config_data


    def handle_yaml(self, bedshifter, yaml_fp):
        """
        Performs operations provided in the yaml config file in the order they were provided.

        :param str bedshifter: the current instance of Bedshift
        :param float yaml_fp: the path to the configuration file
        :return int: the number of total regions perturbed
        """
        data = self._read_from_yaml(yaml_fp)
        operations = [operation for operation in data["bedshift_operations"]]
        num_changed = 0

        for operation in operations:
            ##### add #####
            if set(['add', 'rate', 'mean', 'stdev']) == set(list(operation.keys())):
                rate = operation['rate']
                mean = operation['mean']
                std = operation['stdev']
                num_added = bedshifter.add(rate, mean, std)
                num_changed += num_added
                print("\t{} regions added.".format(num_added))

            ##### add_from_file with no delimiter provided #####
            elif set(['add_from_file', 'file', 'rate']) == set(list(operation.keys())):
                fp = operation['file']
                if os.path.isfile(fp):
                    add_rate = operation['rate']
                    num_added = bedshifter.add_from_file(fp, add_rate)
                    num_changed += num_added
                    print("\t{} regions added from {}.".format(num_added, fp))
                else:
                    print ("File \'{}\' does not exist.".format(fp))
                    sys.exit(1)

            ##### add_from_file with delimiter provided #####
            elif set(['add_from_file', 'file', 'rate', 'delimiter']) == set(list(operation.keys())):
                fp = operation['file']
                if os.path.isfile(fp):
                    add_rate = operation['rate']
                    delimiter = operation['delimiter']
                    num_added = bedshifter.add_from_file(fp, add_rate, delimiter)
                    num_changed += num_added
                    print("\t{} regions added from {}.".format(num_added, fp))
                else:
                    print ("File \'{}\' does not exist.".format(fp))
                    sys.exit(1)

            ##### drop #####
            elif set(['drop', 'rate']) == set(list(operation.keys())):
                rate = operation['rate']
                num_dropped = bedshifter.drop(rate)
                num_changed += num_dropped
                print("\t{} regions dropped.".format(num_dropped))

            ##### drop_from_file with no delimiter provided #####
            elif set(['drop_from_file', 'file', 'rate']) == set(list(operation.keys())):
                fp = operation['file']
                if os.path.isfile(fp):
                    drop_rate = operation['rate']
                    num_dropped = bedshifter.drop_from_file(fp, drop_rate)
                    num_changed += num_dropped
                    print("\t{} regions dropped from {}.".format(num_dropped, fp))
                else:
                    print ("File \'{}\' does not exist.".format(fp))
                    sys.exit(1)

            ##### drop_from_file with delimiter provided #####
            elif set(['drop_from_file', 'file', 'rate', 'delimiter']) == set(list(operation.keys())):
                fp = operation['file']
                if os.path.isfile(fp):
                    drop_rate = operation['rate']
                    delimiter = operation['delimiter']
                    num_dropped = bedshifter.drop_from_file(fp, drop_rate, delimiter)
                    num_changed += num_dropped
                    print("\t{} regions dropped from {}.".format(num_dropped, fp))
                else:
                    print ("File \'{}\' does not exist.".format(fp))
                    sys.exit(1)

            ##### shift #####
            elif set(['shift', 'rate', 'mean', 'stdev']) == set(list(operation.keys())):
                rate = operation['rate']
                mean = operation['mean']
                std = operation['stdev']
                num_shifted = bedshifter.shift(rate, mean, std)
                num_changed += num_shifted
                print("\t{} regions shifted.".format(num_shifted))

            ##### cut #####
            elif set(['cut', 'rate']) == set(list(operation.keys())):
                rate = operation['rate']
                num_cut = bedshifter.cut(rate)
                num_changed += num_cut
                print("\t{} regions cut.".format(num_cut))

            ##### merge #####
            elif set(['merge', 'rate']) == set(list(operation.keys())):
                rate = operation['rate']
                num_merged = bedshifter.merge(rate)
                num_changed += num_merged
                print("\t{} regions merged.".format(num_merged))

            else:
                print("Invalid settings entered in the config file. Please refer to the example below.")
                self._print_sample_config()
                sys.exit(1)

        return num_changed


def main():
    """ Primary workflow """

    parser = logmuse.add_logging_options(build_argparser())
    args, remaining_args = parser.parse_known_args()
    global _LOGGER
    _LOGGER = logmuse.logger_via_cli(args)

    _LOGGER.info("Welcome to bedshift version {}".format(__version__))
    _LOGGER.info("Shifting file: '{}'".format(args.bedfile))

    if not args.bedfile:
        parser.print_help()
        _LOGGER.error("No BED file given")
        sys.exit(1)

    if args.chrom_lengths:
        pass
    elif args.genome:
        try:
            import refgenconf
            rgc = refgenconf.RefGenConf(refgenconf.select_genome_config())
            args.chrom_lengths = rgc.seek(args.genome, "fasta", None, "chrom_sizes")
        except ModuleNotFoundError:
            _LOGGER.error("You must have package refgenconf installed to use a refgenie genome")
            sys.exit(1)
    else:
        if args.addrate > 0 or args.shiftrate > 0:
            _LOGGER.error("You must provide either chrom sizes or a refgenie genome.")
            sys.exit(1)

    msg = """Params:
  chrom.sizes file: {chromsizes}
  shift:
    shift rate: {shiftrate}
    shift mean distance: {shiftmean}
    shift stdev: {shiftstdev}
  add:
    rate: {addrate}
    add mean length: {addmean}
    add stdev: {addstdev}
    add file: {addfile}
    valid regions: {valid_regions}
  cut rate: {cutrate}
  drop rate: {droprate}
  drop regions from file: {dropfile}
  merge rate: {mergerate}
  outputfile: {outputfile}
  repeat: {repeat}
  yaml_config: {yaml_config}
"""

    if args.repeat < 1:
        _LOGGER.error("repeats specified is less than 1")
        sys.exit(1)

    if args.outputfile:
        outfile = args.outputfile
    else:
        outfile = 'bedshifted_{}'.format(os.path.basename(args.bedfile))

    _LOGGER.info(msg.format(
        bedfile=args.bedfile,
        chromsizes=args.chrom_lengths,
        droprate=args.droprate,
        dropfile=args.dropfile,
        addrate=args.addrate,
        addmean=args.addmean,
        addstdev=args.addstdev,
        addfile=args.addfile,
        valid_regions=args.valid_regions,
        shiftrate=args.shiftrate,
        shiftmean=args.shiftmean,
        shiftstdev=args.shiftstdev,
        cutrate=args.cutrate,
        mergerate=args.mergerate,
        outputfile=outfile,
        repeat=args.repeat,
        yaml_config=args.yaml_config))


    bedshifter = Bedshift(args.bedfile, args.chrom_lengths)
    for i in range(args.repeat):
        n = bedshifter.all_perturbations(args.addrate, args.addmean, args.addstdev,
                                            args.addfile, args.valid_regions,
                                            args.shiftrate, args.shiftmean, args.shiftstdev,
                                            args.cutrate,
                                            args.mergerate,
                                            args.droprate,
                                            args.dropfile,
                                            args.yaml_config,
                                            bedshifter)
        print("\t" + str(n) + " regions changed in total.\n")
        if args.repeat == 1:
            bedshifter.to_bed(outfile)
        else:
            modified_outfile = outfile.rsplit("/")
            modified_outfile[-1] = "rep" + str(i+1) + "_" + modified_outfile[-1]
            modified_outfile = "/".join(modified_outfile)
            bedshifter.to_bed(modified_outfile)
        bedshifter.reset_bed()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _LOGGER.error("Program canceled by user!")
        sys.exit(1)
