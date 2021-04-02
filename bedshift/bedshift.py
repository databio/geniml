""" Perturb regions in bedfiles """

import logging
import logmuse
import math
import numpy as np
import os
import pandas as pd
import pyranges as pr
import random
import sys

from bedshift._version import __version__
from bedshift import arguments
from bedshift import BedshiftYAMLHandler

_LOGGER = logging.getLogger(__name__)

__all__ = ["Bedshift"]


class Bedshift(object):
    """
    The bedshift object with methods to perturb regions
    """

    def __init__(self, bedfile_path, chrom_sizes=None, delimiter="\t"):
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
        self.original_num_regions = df.shape[0]
        self.bed = (
            df.astype({0: "object", 1: "int64", 2: "int64", 3: "object"})
            .sort_values([0, 1, 2])
            .reset_index(drop=True)
        )
        self.original_bed = self.bed.copy()

    def _read_chromsizes(self, fp):
        """
        Read chromosome sizes file

        :param str fp: path to the chrom sizes file
        """
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip().split("\t")
                    chrom = str(line[0])
                    size = int(line[1])
                    self.chrom_lens[chrom] = size
        except FileNotFoundError:
            _LOGGER.error("fasta file path {} invalid".format(fp))
            sys.exit(1)

        total_len = sum(self.chrom_lens.values())
        self.chrom_weights = [
            chrom_len / total_len for chrom_len in self.chrom_lens.values()
        ]

    def reset_bed(self):
        """
        Reset the stored bedfile to the state before perturbations
        """
        self.bed = self.original_bed.copy()

    def _precheck(self, rate, requiresChromLens=False, isAdd=False):
        """
        Check if the rate of perturbation is too high or low

        :param float rate: the rate of perturbation
        :param bool requiresChromLens: check if the perturbation requires a chromosome lengths file
        :param bool isAdd: if True, do a special check for the add rate
        """
        if isAdd:
            if rate < 0:
                _LOGGER.error("Rate must be greater than 0")
                sys.exit(1)
        else:
            if rate < 0 or rate > 1:
                _LOGGER.error("Rate must be between 0 and 1")
                sys.exit(1)
        if requiresChromLens:
            if len(self.chrom_lens) == 0:
                _LOGGER.error("chrom.sizes file must be specified")
                sys.exit(1)

    def pick_random_chroms(self, n):
        """
        Utility function to pick a random chromosome

        :param str n: the number of random chromosomes to pick
        :return str, float chrom_str, chrom_len: chromosome number and length
        """
        chrom_strs = random.choices(
            list(self.chrom_lens.keys()), weights=self.chrom_weights, k=n
        )
        chrom_lens = [self.chrom_lens[chrom_str] for chrom_str in chrom_strs]
        return zip(chrom_strs, chrom_lens)

    def add(self, addrate, addmean, addstdev, valid_bed=None, delimiter="\t"):
        """
        Add regions

        :param float addrate: the rate to add regions
        :param float addmean: the mean length of added regions
        :param float addstdev: the standard deviation of the length of added regions
        :param str valid_bed: the file with valid regions where new regions can be added
        :param str delimiter: the delimiter used in valid_bed
        :return int: the number of regions added
        """
        if valid_bed:
            self._precheck(addrate, requiresChromLens=False, isAdd=True)
        else:
            self._precheck(addrate, requiresChromLens=True, isAdd=True)

        rows = self.bed.shape[0]
        num_add = int(rows * addrate)
        new_regions = {0: [], 1: [], 2: [], 3: []}
        if valid_bed:
            valid_regions = self.read_bed(valid_bed, delimiter)
            valid_regions[3] = valid_regions[2] - valid_regions[1]
            total_bp = valid_regions[3].sum()
            valid_regions[4] = valid_regions[3].apply(lambda x: x / total_bp)
            add_rows = random.choices(
                list(range(len(valid_regions))),
                weights=list(valid_regions[4]),
                k=num_add,
            )
            for row in add_rows:
                data = valid_regions.loc[row]
                chrom = data[0]
                start = random.randint(data[1], data[2])
                end = start + int(np.random.normal(addmean, addstdev))
                new_regions[0].append(chrom)
                new_regions[1].append(start)
                new_regions[2].append(end)
                new_regions[3].append("A")
        else:
            random_chroms = self.pick_random_chroms(num_add)
            for chrom_str, chrom_len in random_chroms:
                start = random.randint(1, chrom_len)
                # ensure chromosome length is not exceeded
                end = min(start + int(np.random.normal(addmean, addstdev)), chrom_len)
                new_regions[0].append(chrom_str)
                new_regions[1].append(start)
                new_regions[2].append(end)
                new_regions[3].append("A")
        self.bed = self.bed.append(pd.DataFrame(new_regions), ignore_index=True)
        return num_add

    def add_from_file(self, fp, addrate, delimiter="\t"):
        """
        Add regions from another bedfile to this perturbed bedfile

        :param float addrate: the rate to add regions
        :param str fp: the filepath to the other bedfile
        :return int: the number of regions added
        """
        self._precheck(addrate, requiresChromLens=False, isAdd=True)

        rows = self.bed.shape[0]
        num_add = int(rows * addrate)
        df = self.read_bed(fp, delimiter=delimiter)
        dflen = len(df)
        if num_add > dflen:
            _LOGGER.warning(
                "Number of regions to be added ({}) is larger than the provided bedfile size ({}). Adding {} regions.".format(
                    num_add, dflen, dflen
                )
            )
            num_add = dflen
        add_rows = random.sample(list(range(dflen)), num_add)
        add_df = df.loc[add_rows].reset_index(drop=True)
        add_df[3] = pd.Series(["A"] * add_df.shape[0])
        self.bed = self.bed.append(add_df, ignore_index=True)
        return num_add

    def shift(self, shiftrate, shiftmean, shiftstdev, shift_rows=[]):
        """
        Shift regions

        :param float shiftrate: the rate to shift regions (both the start and end are shifted by the same amount)
        :param float shiftmean: the mean shift distance
        :param float shiftstdev: the standard deviation of the shift distance
        :return int: the number of regions shifted
        """
        self._precheck(shiftrate, requiresChromLens=True)

        rows = self.bed.shape[0]
        if len(shift_rows) == 0:
            shift_rows = random.sample(list(range(rows)), int(rows * shiftrate))
        new_row_list = []
        to_drop = []
        num_shifted = 0
        invalid_shifted = 0
        for row in shift_rows:
            drop_row, new_region = self._shift(
                row, shiftmean, shiftstdev
            )  # shifted rows display a 1
            if drop_row is not None and new_region:
                num_shifted += 1
                new_row_list.append(new_region)
                to_drop.append(drop_row)
            else:
                invalid_shifted += 1
        self.bed = self.bed.drop(to_drop)
        self.bed = self.bed.append(new_row_list, ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        if invalid_shifted > 0:
            _LOGGER.warning(
                f"{invalid_shifted} regions were prevented from being shifted outside of chromosome boundaries. Reported regions shifted will be less than expected."
            )
        return num_shifted

    def _shift(self, row, mean, stdev):
        theshift = int(np.random.normal(mean, stdev))

        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row][2]
        if start + theshift < 0 or end + theshift > self.chrom_lens[str(chrom)]:
            # check if the region is shifted out of chromosome length bounds
            return None, None

        return row, {0: chrom, 1: start + theshift, 2: end + theshift, 3: "S"}

    def shift_from_file(self, fp, shiftrate, shiftmean, shiftstdev, delimiter="\t"):
        """
        Shift regions that overlap the specified file's regions

        :param str fp: the file on which to find overlaps
        :param float shiftrate: the rate to shift regions (both the start and end are shifted by the same amount)
        :param float shiftmean: the mean shift distance
        :param float shiftstdev: the standard deviation of the shift distance
        :param str delimiter: the delimiter used in fp
        :return int: the number of regions shifted
        """
        self._precheck(shiftrate, requiresChromLens=True)

        rows = self.bed.shape[0]
        num_shift = int(rows * shiftrate)

        intersect_regions = self._find_overlap(fp)
        original_colnames = self.bed.columns
        intersect_regions.columns = [str(col) for col in intersect_regions.columns]
        self.bed.columns = [str(col) for col in self.bed.columns]
        indices_of_overlap_regions = self.bed.reset_index().merge(intersect_regions)[
            "index"
        ]
        self.bed.columns = [int(col) for col in self.bed.columns]

        interlen = len(indices_of_overlap_regions)
        if num_shift > interlen:
            _LOGGER.warning(
                "Desired regions shifted ({}) is greater than the number of overlaps found ({}). Shifting {} regions.".format(
                    num_shift, interlen, interlen
                )
            )
            num_shift = len(indices_of_overlap_regions)

        elif interlen > num_shift:
            indices_of_overlap_regions = indices_of_overlap_regions.sample(n=num_shift)

        indices_of_overlap_regions = indices_of_overlap_regions.to_list()

        return self.shift(shiftrate, shiftmean, shiftstdev, indices_of_overlap_regions)

    def cut(self, cutrate):
        """
        Cut regions to create two new regions

        :param float cutrate: the rate to cut regions into two separate regions
        :return int: the number of regions cut
        """
        self._precheck(cutrate)

        rows = self.bed.shape[0]
        cut_rows = random.sample(list(range(rows)), int(rows * cutrate))
        new_row_list = []
        to_drop = []
        for row in cut_rows:
            drop_row, new_regions = self._cut(row)  # cut rows display a 2
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
        thecut = (
            start + end
        ) // 2  # int(np.random.normal((start+end)/2, (end - start)/6))
        if thecut <= start:
            thecut = start + 10
        if thecut >= end:
            thecut = end - 10

        """ may add in later, this makes the api confusing!
        # adjust the cut regions using the shift function
        new_segs = self.__shift(new_segs, 0, meanshift, stdevshift)
        new_segs = self.__shift(new_segs, 1, meanshift, stdevshift)
        """

        return (
            row,
            [
                {0: chrom, 1: start, 2: thecut, 3: "C"},
                {0: chrom, 1: thecut, 2: end, 3: "C"},
            ],
        )

    def merge(self, mergerate):
        """
        Merge two regions into one new region

        :param float mergerate: the rate to merge two regions into one
        :return int: number of regions merged
        """
        self._precheck(mergerate)

        rows = self.bed.shape[0]
        merge_rows = random.sample(list(range(rows)), int(rows * mergerate))
        to_add = []
        to_drop = []
        for row in merge_rows:
            drop_rows, add_row = self._merge(row)
            if drop_rows and add_row:
                to_add.append(add_row)
                to_drop.extend(drop_rows)
        self.bed = self.bed.drop(to_drop)
        self.bed = self.bed.append(to_add, ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        return len(to_drop)

    def _merge(self, row):
        # check if the regions being merged are on the same chromosome
        if (
            row + 1 not in self.bed.index
            or self.bed.loc[row][0] != self.bed.loc[row + 1][0]
        ):
            return None, None

        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row + 1][2]
        return [row, row + 1], {0: chrom, 1: start, 2: end, 3: "M"}

    def drop(self, droprate):
        """
        Drop regions

        :param float droprate: the rate to drop/remove regions
        :return int: the number of rows dropped
        """
        self._precheck(droprate)

        rows = self.bed.shape[0]
        drop_rows = random.sample(list(range(rows)), int(rows * droprate))
        self.bed = self.bed.drop(drop_rows)
        self.bed = self.bed.reset_index(drop=True)
        return len(drop_rows)

    def drop_from_file(self, fp, droprate, delimiter="\t"):
        """
        drop regions that overlap between the reference bedfile and the provided bedfile.

        :param float droprate: the rate to drop regions
        :param str fp: the filepath to the other bedfile containing regions to be dropped
        :return int: the number of regions dropped
        """
        self._precheck(droprate)

        rows = self.bed.shape[0]
        num_drop = int(rows * droprate)
        drop_bed = self.read_bed(fp, delimiter=delimiter)

        intersect_regions = self._find_overlap(drop_bed)
        original_colnames = self.bed.columns
        intersect_regions.columns = [str(col) for col in intersect_regions.columns]
        self.bed.columns = [str(col) for col in self.bed.columns]
        indices_of_overlap_regions = self.bed.reset_index().merge(intersect_regions)[
            "index"
        ]
        self.bed.columns = [int(col) for col in self.bed.columns]

        interlen = len(indices_of_overlap_regions)
        if num_drop > interlen:
            _LOGGER.warning(
                "Desired regions dropped ({}) is greater than the number of overlaps found ({}). Dropping {} regions.".format(
                    num_drop, interlen, interlen
                )
            )
            num_drop = len(indices_of_overlap_regions)
        elif interlen > num_drop:
            indices_of_overlap_regions = indices_of_overlap_regions.sample(n=num_drop)
        indices_of_overlap_regions = indices_of_overlap_regions.to_list()

        self.bed = self.bed.drop(indices_of_overlap_regions)
        return num_drop

    def _find_overlap(self, fp, reference=None):
        """
        Find intersecting regions between the reference bedfile and the comparison file provided in the yaml config file.

        :param str fp: path to file, or pandas DataFrame, for comparison
        :param str reference: path to file, or pandas DataFrame, for reference. If None, then defaults to the original BED file provided to the Bedshift constructor
        :return pd.DataFrame: a DataFrame of overlapping regions
        """
        if reference is None:
            reference_bed = self.original_bed.copy()
        else:
            if isinstance(reference, pd.DataFrame):
                reference_bed = reference.copy()
            elif isinstance(reference, str):
                reference_bed = self.read_bed(reference)
            else:
                raise Exception("unsupported input type: {}".format(type(reference)))
        if isinstance(fp, pd.DataFrame):
            comparison_bed = fp.copy()
        elif isinstance(fp, str):
            comparison_bed = self.read_bed(fp)
        else:
            raise Exception("unsupported input type: {}".format(type(reference)))
        reference_bed.columns = ["Chromosome", "Start", "End", "modifications"]
        comparison_bed.columns = ["Chromosome", "Start", "End", "modifications"]
        reference_pr = pr.PyRanges(reference_bed)
        comparison_pr = pr.PyRanges(comparison_bed)
        intersection = reference_pr.overlap(comparison_pr, how="first").as_df()
        if len(intersection) == 0:
            raise Exception(
                "no intersection found between {} and {}".format(
                    reference_bed, comparison_bed
                )
            )
        intersection = intersection.drop(["modifications"], axis=1)
        intersection.columns = [0, 1, 2]
        return intersection

    def all_perturbations(
        self,
        addrate=0.0,
        addmean=320.0,
        addstdev=30.0,
        addfile=None,
        valid_regions=None,
        shiftrate=0.0,
        shiftmean=0.0,
        shiftstdev=150.0,
        shiftfile=None,
        cutrate=0.0,
        mergerate=0.0,
        droprate=0.0,
        dropfile=None,
        yaml=None,
    ):
        """
        Perform all five perturbations in the order of shift, add, cut, merge, drop.

        :param float addrate: the rate (as a proportion of the total number of regions) to add regions
        :param float addmean: the mean length of added regions
        :param float addstdev: the standard deviation of the length of added regions
        :param str addfile: the file containing regions to be added
        :param str valid_regions: the file containing regions where new regions can be added
        :param float shiftrate: the rate to shift regions (both the start and end are shifted by the same amount)
        :param float shiftmean: the mean shift distance
        :param float shiftstdev: the standard deviation of the shift distance
        :param str shiftfile: the file containing regions to be shifted
        :param float cutrate: the rate to cut regions into two separate regions
        :param float mergerate: the rate to merge two regions into one
        :param float droprate: the rate to drop/remove regions
        :param str dropfile: the file containing regions to be dropped
        :param str yaml: the yaml_config filepath
        :param bedshift.Bedshift bedshifter: Bedshift instance
        :return int: the number of total regions perturbed
        """
        if yaml:
            return BedshiftYAMLHandler.BedshiftYAMLHandler(self, yaml).handle_yaml()
        n = 0
        if shiftrate > 0:
            if shiftfile:
                n += self.shift_from_file(shiftfile, shiftrate, shiftmean, shiftstdev)
            else:
                n += self.shift(shiftrate, shiftmean, shiftstdev)
        if addrate > 0:
            if addfile:
                n += self.add_from_file(addfile, addrate)
            else:
                n += self.add(addrate, addmean, addstdev, valid_regions)
        if cutrate > 0:
            n += self.cut(cutrate)
        if mergerate > 0:
            n += self.merge(mergerate)
        if droprate > 0:
            if dropfile:
                n += self.drop_from_file(dropfile, droprate)
            else:
                n += self.drop(droprate)
        return n

    def to_bed(self, outfile_name):
        """
        Write a pandas dataframe back into BED file format

        :param str outfile_name: The name of the output BED file
        """
        self.bed.sort_values([0, 1, 2], inplace=True)
        self.bed.to_csv(
            outfile_name, sep="\t", header=False, index=False, float_format="%.0f"
        )

    def read_bed(self, bedfile_path, delimiter="\t"):
        """
        Read a BED file into pandas dataframe

        :param str bedfile_path: The path to the BED file
        """
        try:
            df = pd.read_csv(
                bedfile_path,
                sep=delimiter,
                header=None,
                usecols=[0, 1, 2],
                engine="python",
            )
        except FileNotFoundError:
            _LOGGER.error("BED file path {} invalid".format(bedfile_path))
            sys.exit(1)
        except:
            _LOGGER.error("file {} could not be read".format(bedfile_path))
            sys.exit(1)

        # if there is a header line in the table, remove it
        if not str(df.iloc[0, 1]).isdigit():
            df = df[1:].reset_index(drop=True)

        df[3] = "-"  # column indicating which modifications were made
        return df


def main():
    """ Primary workflow """

    parser = logmuse.add_logging_options(arguments.build_argparser())
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
            _LOGGER.error(
                "You must have package refgenconf installed to use a refgenie genome"
            )
            sys.exit(1)

    msg = arguments.param_msg

    if args.repeat < 1:
        _LOGGER.error("repeats specified is less than 1")
        sys.exit(1)

    if args.outputfile:
        outfile_base = args.outputfile
    else:
        outfile_base = "bedshifted_{}".format(os.path.basename(args.bedfile))

    _LOGGER.info(
        msg.format(
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
            shiftfile=args.shiftfile,
            cutrate=args.cutrate,
            mergerate=args.mergerate,
            outputfile=outfile_base,
            repeat=args.repeat,
            yaml_config=args.yaml_config,
        )
    )

    bedshifter = Bedshift(args.bedfile, args.chrom_lengths)
    _LOGGER.info(f"Generating {args.repeat} repetitions...")

    pct_reports = [int(x * args.repeat / 100) for x in [5, 25, 50, 75, 100]]

    for i in range(args.repeat):
        n = bedshifter.all_perturbations(
            args.addrate,
            args.addmean,
            args.addstdev,
            args.addfile,
            args.valid_regions,
            args.shiftrate,
            args.shiftmean,
            args.shiftstdev,
            args.shiftfile,
            args.cutrate,
            args.mergerate,
            args.droprate,
            args.dropfile,
            args.yaml_config,
        )
        if args.repeat == 1:
            bedshifter.to_bed(outfile_base)
            _LOGGER.info(
                "REGION COUNT | original: {}\tnew: {}\tchanged: {}\t\noutput file: {}".format(
                    bedshifter.original_num_regions,
                    bedshifter.bed.shape[0],
                    str(n),
                    outfile_base,
                )
            )
        else:
            basename, ext = os.path.splitext(os.path.basename(outfile_base))
            dirname = os.path.dirname(outfile_base)
            digits = int(math.log10(args.repeat)) + 1

            rep = str(i + 1).zfill(digits)
            modified_outfile_path = os.path.join(dirname, f"{basename}_rep{rep}{ext}")
            bedshifter.to_bed(modified_outfile_path)

            pct_finished = int((100 * (i + 1)) / args.repeat)
            if i + 1 in pct_reports:
                _LOGGER.info(
                    f"Rep {i+1}. Finished: {pct_finished}%. Output file: {modified_outfile_path}"
                )

        bedshifter.reset_bed()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _LOGGER.error("Program canceled by user!")
        sys.exit(1)
