"""Perturb regions in bedfiles"""

import logging
import random

import genomicranges as gr
import numpy as np
import pandas as pd

from .yaml_handler import BedshiftYAMLHandler

_LOGGER = logging.getLogger(__name__)

__all__ = ["Bedshift"]


class Bedshift(object):
    """The bedshift object with methods to perturb regions."""

    def __init__(self, bedfile_path, chrom_sizes=None, delimiter="\t"):
        """Read in a .bed file to pandas DataFrame format.

        Args:
            bedfile_path (str): The path to the BED file.
            chrom_sizes (str): The path to the chrom.sizes file.
            delimiter (str): The delimiter used in the BED file.
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
        """Read chromosome sizes file.

        Args:
            fp (str): Path to the chrom sizes file.
        """
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip().split("\t")
                    chrom = str(line[0])
                    size = int(line[1])
                    self.chrom_lens[chrom] = size
        except FileNotFoundError:
            msg = "Fasta file path {} invalid".format(fp)
            _LOGGER.error(msg)
            raise FileNotFoundError(msg)

        total_len = sum(self.chrom_lens.values())
        self.chrom_weights = [chrom_len / total_len for chrom_len in self.chrom_lens.values()]

    def reset_bed(self):
        """Reset the stored bedfile to the state before perturbations."""
        self.bed = self.original_bed.copy()

    def _precheck(self, rate, requiresChromLens=False, isAdd=False):
        """Check if the rate of perturbation is too high or low.

        Args:
            rate (float): The rate of perturbation.
            requiresChromLens (bool): Check if the perturbation requires a chromosome lengths file.
            isAdd (bool): If True, do a special check for the add rate.
        """
        if isAdd:
            if rate < 0:
                msg = "Rate must be greater than 0"
                _LOGGER.error(msg)
                raise ValueError(msg)
        else:
            if rate < 0 or rate > 1:
                msg = "Rate must be between 0 and 1"
                _LOGGER.error(msg)
                raise ValueError(msg)
        if requiresChromLens:
            if len(self.chrom_lens) == 0:
                msg = "chrom.sizes file must be specified"
                _LOGGER.error(msg)
                raise FileNotFoundError(msg)

    def pick_random_chroms(self, n):
        """Utility function to pick a random chromosome.

        Args:
            n (str): The number of random chromosomes to pick.

        Returns:
            zip: Tuples of (chrom_str, chrom_len) containing chromosome number and length.
        """
        chrom_strs = random.choices(list(self.chrom_lens.keys()), weights=self.chrom_weights, k=n)
        chrom_lens = [self.chrom_lens[chrom_str] for chrom_str in chrom_strs]
        return zip(chrom_strs, chrom_lens)

    def add(self, addrate, addmean, addstdev, valid_bed=None, delimiter="\t"):
        """Add regions.

        Args:
            addrate (float): The rate to add regions.
            addmean (float): The mean length of added regions.
            addstdev (float): The standard deviation of the length of added regions.
            valid_bed (str): The file with valid regions where new regions can be added.
            delimiter (str): The delimiter used in valid_bed.

        Returns:
            int: The number of regions added.
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
        self.bed = pd.concat([self.bed, pd.DataFrame(new_regions)], ignore_index=True)
        return num_add

    def add_from_file(self, fp, addrate, delimiter="\t"):
        """Add regions from another bedfile to this perturbed bedfile.

        Args:
            fp (str): The filepath to the other bedfile.
            addrate (float): The rate to add regions.
            delimiter (str): The delimiter used in the bedfile.

        Returns:
            int: The number of regions added.
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
        self.bed = pd.concat([self.bed, add_df], ignore_index=True)
        return num_add

    def shift(self, shiftrate, shiftmean, shiftstdev, shift_rows=[]):
        """Shift regions.

        Args:
            shiftrate (float): The rate to shift regions (both the start and end are shifted by the same amount).
            shiftmean (float): The mean shift distance.
            shiftstdev (float): The standard deviation of the shift distance.
            shift_rows (list): Specific rows to shift.

        Returns:
            int: The number of regions shifted.
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
        self.bed = pd.concat([self.bed, pd.DataFrame(new_row_list)], ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        if invalid_shifted > 0:
            _LOGGER.warning(
                f"{invalid_shifted} regions were prevented from being shifted outside of chromosome boundaries. Reported regions shifted will be less than expected."
            )
        return num_shifted

    def _shift(self, row, mean, stdev):
        """Shift a single region.

        Args:
            row (int): The index of the row to shift.
            mean (float): The mean shift distance.
            stdev (float): The standard deviation of the shift distance.

        Returns:
            tuple: A tuple of (row_index, shifted_region_dict) or (None, None) if shift is invalid.
        """
        theshift = int(np.random.normal(mean, stdev))

        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row][2]
        if start + theshift < 0 or end + theshift > self.chrom_lens[str(chrom)]:
            # check if the region is shifted out of chromosome length bounds
            return None, None

        return row, {0: chrom, 1: start + theshift, 2: end + theshift, 3: "S"}

    def shift_from_file(self, fp, shiftrate, shiftmean, shiftstdev, delimiter="\t"):
        """Shift regions that overlap the specified file's regions.

        Args:
            fp (str): The file on which to find overlaps.
            shiftrate (float): The rate to shift regions (both the start and end are shifted by the same amount).
            shiftmean (float): The mean shift distance.
            shiftstdev (float): The standard deviation of the shift distance.
            delimiter (str): The delimiter used in fp.

        Returns:
            int: The number of regions shifted.
        """
        self._precheck(shiftrate, requiresChromLens=True)

        rows = self.bed.shape[0]
        num_shift = int(rows * shiftrate)

        intersect_regions = self._find_overlap(fp)
        original_colnames = self.bed.columns
        intersect_regions.columns = [str(col) for col in intersect_regions.columns]
        self.bed.columns = [str(col) for col in self.bed.columns]
        indices_of_overlap_regions = self.bed.reset_index().merge(intersect_regions)["index"]
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
        """Cut regions to create two new regions.

        Args:
            cutrate (float): The rate to cut regions into two separate regions.

        Returns:
            int: The number of regions cut.
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
        self.bed = pd.concat([self.bed, pd.DataFrame(new_row_list)], ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        return len(cut_rows)

    def _cut(self, row):
        """Cut a single region into two regions.

        Args:
            row (int): The index of the row to cut.

        Returns:
            tuple: A tuple of (row_index, list_of_two_new_regions).
        """
        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row][2]

        # choose where to cut the region
        thecut = (start + end) // 2  # int(np.random.normal((start+end)/2, (end - start)/6))
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
        """Merge two regions into one new region.

        Args:
            mergerate (float): The rate to merge two regions into one.

        Returns:
            int: Number of regions merged.
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
        self.bed = pd.concat([self.bed, pd.DataFrame(to_add)], ignore_index=True)
        self.bed = self.bed.reset_index(drop=True)
        return len(to_drop)

    def _merge(self, row):
        """Merge a region with the next region.

        Args:
            row (int): The index of the row to merge.

        Returns:
            tuple: A tuple of (list_of_rows_to_drop, merged_region_dict) or (None, None) if merge is invalid.
        """
        # check if the regions being merged are on the same chromosome
        if row + 1 not in self.bed.index or self.bed.loc[row][0] != self.bed.loc[row + 1][0]:
            return None, None

        chrom = self.bed.loc[row][0]
        start = self.bed.loc[row][1]
        end = self.bed.loc[row + 1][2]
        return [row, row + 1], {0: chrom, 1: start, 2: end, 3: "M"}

    def drop(self, droprate):
        """Drop regions.

        Args:
            droprate (float): The rate to drop/remove regions.

        Returns:
            int: The number of rows dropped.
        """
        self._precheck(droprate)

        rows = self.bed.shape[0]
        drop_rows = random.sample(list(range(rows)), int(rows * droprate))
        self.bed = self.bed.drop(drop_rows)
        self.bed = self.bed.reset_index(drop=True)
        return len(drop_rows)

    def drop_from_file(self, fp, droprate, delimiter="\t"):
        """Drop regions that overlap between the reference bedfile and the provided bedfile.

        Args:
            fp (str): The filepath to the other bedfile containing regions to be dropped.
            droprate (float): The rate to drop regions.
            delimiter (str): The delimiter used in the bedfile.

        Returns:
            int: The number of regions dropped.
        """
        self._precheck(droprate)

        rows = self.bed.shape[0]
        num_drop = int(rows * droprate)
        drop_bed = self.read_bed(fp, delimiter=delimiter)

        intersect_regions = self._find_overlap(drop_bed)
        # original_colnames = self.bed.columns
        intersect_regions.columns = [str(col) for col in intersect_regions.columns]
        self.bed.columns = [str(col) for col in self.bed.columns]
        indices_of_overlap_regions = self.bed.reset_index().merge(intersect_regions)["index"]
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

    def set_seed(self, seednum):
        """Set the random seed for reproducible perturbations.

        Args:
            seednum (int): The seed value.

        Raises:
            ValueError: If seednum cannot be converted to an integer.
        """
        try:
            seednum = int(seednum)
            random.seed(seednum)
            np.random.seed(seednum)
        except ValueError:
            msg = "Seed should be an integer, not {}.".format(type(seednum))
            _LOGGER.error(msg)
            raise ValueError(msg)

    def _find_overlap(self, fp, reference=None):
        """Find intersecting regions between the reference bedfile and the comparison file.

        Args:
            fp (str or pd.DataFrame): Path to file, or pandas DataFrame, for comparison.
            reference (str or pd.DataFrame): Path to file, or pandas DataFrame, for reference. If None, then defaults to the original BED file provided to the Bedshift constructor.

        Returns:
            pd.DataFrame: A DataFrame of overlapping regions.
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
        reference_bed.columns = ["seqnames", "starts", "ends", "modifications"]
        comparison_bed.columns = ["seqnames", "starts", "ends", "modifications"]

        reference_gr = gr.GenomicRanges.from_pandas(reference_bed)
        comparison_gr = gr.GenomicRanges.from_pandas(comparison_bed)
        intersection_gr = reference_gr.subset_by_overlaps(comparison_gr)
        intersection = intersection_gr.to_pandas()

        if len(intersection) == 0:
            raise Exception(
                "no intersection found between {} and {}".format(reference_bed, comparison_bed)
            )

        intersection = intersection[["seqnames", "starts", "ends"]]
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
        seed=None,
    ):
        """Perform all five perturbations in the order of shift, add, cut, merge, drop.

        Args:
            addrate (float): The rate (as a proportion of the total number of regions) to add regions.
            addmean (float): The mean length of added regions.
            addstdev (float): The standard deviation of the length of added regions.
            addfile (str): The file containing regions to be added.
            valid_regions (str): The file containing regions where new regions can be added.
            shiftrate (float): The rate to shift regions (both the start and end are shifted by the same amount).
            shiftmean (float): The mean shift distance.
            shiftstdev (float): The standard deviation of the shift distance.
            shiftfile (str): The file containing regions to be shifted.
            cutrate (float): The rate to cut regions into two separate regions.
            mergerate (float): The rate to merge two regions into one.
            droprate (float): The rate to drop/remove regions.
            dropfile (str): The file containing regions to be dropped.
            yaml (str): The yaml_config filepath.
            seed (int): A seed for allowing reproducible perturbations.

        Returns:
            int: The number of total regions perturbed.
        """
        if seed:
            self.set_seed(seed)
        if yaml:
            return BedshiftYAMLHandler(self, yaml).handle_yaml()
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
        """Write a pandas dataframe back into BED file format.

        Args:
            outfile_name (str): The name of the output BED file.
        """
        self.bed.sort_values([0, 1, 2], inplace=True)
        self.bed.to_csv(outfile_name, sep="\t", header=False, index=False, float_format="%.0f")

    def read_bed(self, bedfile_path, delimiter="\t"):
        """Read a BED file into pandas dataframe.

        Args:
            bedfile_path (str): The path to the BED file.
            delimiter (str): The delimiter used in the BED file.

        Returns:
            pd.DataFrame: The BED file as a pandas DataFrame.
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
            msg = "BED file path {} invalid".format(bedfile_path)
            _LOGGER.error(msg)
            raise FileNotFoundError(msg)
        except:
            msg = "File {} could not be read".format(bedfile_path)
            _LOGGER.error(msg)
            raise Exception(msg)

        # if there is a header line in the table, remove it
        if not str(df.iloc[0, 1]).isdigit():
            df = df[1:].reset_index(drop=True)

        df[3] = "-"  # column indicating which modifications were made
        return df
