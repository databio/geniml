"""Perturb regions in bedfiles"""

import logging
import os
import random
import tempfile

import numpy as np
from gtars.models import RegionSet

from .yaml_handler import BedshiftYAMLHandler

_LOGGER = logging.getLogger(__name__)

__all__ = ["Bedshift"]


def _list_to_regionset(regions):
    """Convert a list of lists to a RegionSet via a temporary BED file.

    Args:
        regions (list): A list of lists, each containing [chrom, start, end, ...].

    Returns:
        RegionSet: A RegionSet constructed from the regions.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        for r in regions:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
        tmp_path = f.name
    rs = RegionSet(tmp_path)
    os.unlink(tmp_path)
    return rs


class Bedshift(object):
    """The bedshift object with methods to perturb regions."""

    def __init__(self, bedfile_path, chrom_sizes=None):
        """Read in a .bed file to a list of lists.

        Args:
            bedfile_path (str): The path to the BED file.
            chrom_sizes (str): The path to the chrom.sizes file.
        """
        self.bedfile_path = bedfile_path
        self.chrom_lens = {}
        if chrom_sizes:
            self._read_chromsizes(chrom_sizes)
        self.bed = self.read_bed(bedfile_path)
        self.original_num_regions = len(self.bed)
        self.bed.sort(key=lambda r: (r[0], r[1], r[2]))
        self.original_bed = [row[:] for row in self.bed]  # deep copy

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
        self.bed = [row[:] for row in self.original_bed]

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

    def _validate_region(self, start, end):
        """Return True if the region is valid (start < end and start >= 0)."""
        return start >= 0 and start < end

    def _remove_invalid_regions(self):
        """Remove any regions where start >= end."""
        before = len(self.bed)
        self.bed = [r for r in self.bed if r[1] < r[2]]
        removed = before - len(self.bed)
        if removed > 0:
            _LOGGER.warning(f"Removed {removed} invalid regions (start >= end)")
        return removed

    def _sort_bed(self):
        """Sort bed by chromosome, start, end."""
        self.bed.sort(key=lambda r: (r[0], r[1], r[2]))

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

    def add(self, addrate, addmean, addstdev, valid_bed=None):
        """Add regions.

        Args:
            addrate (float): The rate to add regions.
            addmean (float): The mean length of added regions.
            addstdev (float): The standard deviation of the length of added regions.
            valid_bed (str): The file with valid regions where new regions can be added.

        Returns:
            int: The number of regions added.
        """
        if valid_bed:
            self._precheck(addrate, requiresChromLens=False, isAdd=True)
        else:
            self._precheck(addrate, requiresChromLens=True, isAdd=True)

        rows = len(self.bed)
        num_add = int(rows * addrate)
        new_rows = []

        if valid_bed:
            valid_regions = self.read_bed(valid_bed)
            total_bp = sum(r[2] - r[1] for r in valid_regions)
            weights = [(r[2] - r[1]) / total_bp for r in valid_regions]
            add_rows = random.choices(
                list(range(len(valid_regions))),
                weights=weights,
                k=num_add,
            )
            for row in add_rows:
                data = valid_regions[row]
                chrom = data[0]
                start = random.randint(data[1], data[2])
                length = max(1, abs(int(np.random.normal(addmean, addstdev))))
                end = min(start + length, data[2])
                if end <= start:
                    end = start + 1
                new_rows.append([chrom, start, end, "A"])
        else:
            random_chroms = self.pick_random_chroms(num_add)
            for chrom_str, chrom_len in random_chroms:
                start = random.randint(1, chrom_len)
                length = max(1, abs(int(np.random.normal(addmean, addstdev))))
                end = min(start + length, chrom_len)
                if end <= start:
                    end = start + 1
                new_rows.append([chrom_str, start, end, "A"])

        self.bed.extend(new_rows)
        self._sort_bed()
        return num_add

    def add_from_file(self, fp, addrate):
        """Add regions from another bedfile to this perturbed bedfile.

        Args:
            fp (str): The filepath to the other bedfile.
            addrate (float): The rate to add regions.

        Returns:
            int: The number of regions added.
        """
        self._precheck(addrate, requiresChromLens=False, isAdd=True)

        rows = len(self.bed)
        num_add = int(rows * addrate)
        regions = self.read_bed(fp)
        reglen = len(regions)
        if num_add > reglen:
            _LOGGER.warning(
                "Number of regions to be added ({}) is larger than the provided bedfile size ({}). Adding {} regions.".format(
                    num_add, reglen, reglen
                )
            )
            num_add = reglen
        add_indices = random.sample(list(range(reglen)), num_add)
        for i in add_indices:
            row = regions[i][:]
            row[3] = "A"
            self.bed.append(row)
        self._sort_bed()
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

        rows = len(self.bed)
        if len(shift_rows) == 0:
            shift_rows = random.sample(list(range(rows)), int(rows * shiftrate))
        new_row_list = []
        to_drop = []
        num_shifted = 0
        invalid_shifted = 0
        for row in shift_rows:
            drop_row, new_region = self._shift(row, shiftmean, shiftstdev)
            if drop_row is not None and new_region:
                num_shifted += 1
                new_row_list.append(new_region)
                to_drop.append(drop_row)
            else:
                invalid_shifted += 1
        for idx in sorted(to_drop, reverse=True):
            del self.bed[idx]
        self.bed.extend(new_row_list)
        self._sort_bed()
        if invalid_shifted > 0:
            _LOGGER.warning(
                f"{invalid_shifted} regions were prevented from being shifted outside of chromosome boundaries."
            )
        return num_shifted

    def _shift(self, row, mean, stdev):
        """Shift a single region.

        Args:
            row (int): The index of the row to shift.
            mean (float): The mean shift distance.
            stdev (float): The standard deviation of the shift distance.

        Returns:
            tuple: A tuple of (row_index, shifted_region_list) or (None, None) if shift is invalid.
        """
        theshift = int(np.random.normal(mean, stdev))

        chrom = self.bed[row][0]
        start = self.bed[row][1]
        end = self.bed[row][2]
        new_start = start + theshift
        new_end = end + theshift
        if new_start < 0 or new_end > self.chrom_lens[str(chrom)]:
            return None, None
        if new_start >= new_end:
            return None, None

        return row, [chrom, new_start, new_end, "S"]

    def shift_from_file(self, fp, shiftrate, shiftmean, shiftstdev):
        """Shift regions that overlap the specified file's regions.

        Args:
            fp (str): The file on which to find overlaps.
            shiftrate (float): The rate to shift regions (both the start and end are shifted by the same amount).
            shiftmean (float): The mean shift distance.
            shiftstdev (float): The standard deviation of the shift distance.

        Returns:
            int: The number of regions shifted.
        """
        self._precheck(shiftrate, requiresChromLens=True)

        rows = len(self.bed)
        num_shift = int(rows * shiftrate)

        intersect_regions = self._find_overlap(fp)
        intersect_set = {(r[0], r[1], r[2]) for r in intersect_regions}
        indices_of_overlap = [
            i for i, r in enumerate(self.bed) if (r[0], r[1], r[2]) in intersect_set
        ]

        interlen = len(indices_of_overlap)
        if num_shift > interlen:
            _LOGGER.warning(
                "Desired regions shifted ({}) is greater than the number of overlaps found ({}). Shifting {} regions.".format(
                    num_shift, interlen, interlen
                )
            )
            num_shift = interlen
        elif interlen > num_shift:
            indices_of_overlap = random.sample(indices_of_overlap, num_shift)

        return self.shift(shiftrate, shiftmean, shiftstdev, indices_of_overlap)

    def cut(self, cutrate):
        """Cut regions to create two new regions.

        Args:
            cutrate (float): The rate to cut regions into two separate regions.

        Returns:
            int: The number of regions cut.
        """
        self._precheck(cutrate)

        rows = len(self.bed)
        cut_rows = random.sample(list(range(rows)), int(rows * cutrate))
        new_row_list = []
        to_drop = []
        num_cut = 0
        for row in cut_rows:
            drop_row, new_regions = self._cut(row)
            if drop_row is not None and new_regions:
                new_row_list.extend(new_regions)
                to_drop.append(drop_row)
                num_cut += 1
        for idx in sorted(to_drop, reverse=True):
            del self.bed[idx]
        self.bed.extend(new_row_list)
        self._sort_bed()
        return num_cut

    def _cut(self, row):
        """Cut a single region into two regions.

        Args:
            row (int): The index of the row to cut.

        Returns:
            tuple: A tuple of (row_index, list_of_two_new_regions) or (None, None) if region is too small.
        """
        chrom = self.bed[row][0]
        start = self.bed[row][1]
        end = self.bed[row][2]

        # Region must be at least 2bp to cut into two valid regions
        if end - start < 2:
            return None, None

        thecut = random.randint(start + 1, end - 1)

        return (
            row,
            [
                [chrom, start, thecut, "C"],
                [chrom, thecut, end, "C"],
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

        self._sort_bed()
        rows = len(self.bed)
        merge_rows = random.sample(list(range(rows)), int(rows * mergerate))
        to_add = []
        to_drop = []
        for row in merge_rows:
            drop_rows, add_row = self._merge(row)
            if drop_rows and add_row:
                to_add.append(add_row)
                to_drop.extend(drop_rows)
        for idx in sorted(set(to_drop), reverse=True):
            del self.bed[idx]
        self.bed.extend(to_add)
        self._sort_bed()
        return len(to_drop)

    def _merge(self, row):
        """Merge a region with the next region.

        Args:
            row (int): The index of the row to merge.

        Returns:
            tuple: A tuple of (list_of_rows_to_drop, merged_region_list) or (None, None) if merge is invalid.
        """
        if row + 1 >= len(self.bed) or self.bed[row][0] != self.bed[row + 1][0]:
            return None, None

        chrom = self.bed[row][0]
        start = min(self.bed[row][1], self.bed[row + 1][1])
        end = max(self.bed[row][2], self.bed[row + 1][2])
        return [row, row + 1], [chrom, start, end, "M"]

    def drop(self, droprate):
        """Drop regions.

        Args:
            droprate (float): The rate to drop/remove regions.

        Returns:
            int: The number of rows dropped.
        """
        self._precheck(droprate)

        rows = len(self.bed)
        drop_rows = random.sample(list(range(rows)), int(rows * droprate))
        for idx in sorted(drop_rows, reverse=True):
            del self.bed[idx]
        self._sort_bed()
        return len(drop_rows)

    def drop_from_file(self, fp, droprate):
        """Drop regions that overlap between the reference bedfile and the provided bedfile.

        Args:
            fp (str): The filepath to the other bedfile containing regions to be dropped.
            droprate (float): The rate to drop regions.

        Returns:
            int: The number of regions dropped.
        """
        self._precheck(droprate)

        rows = len(self.bed)
        num_drop = int(rows * droprate)
        drop_bed = self.read_bed(fp)

        intersect_regions = self._find_overlap(drop_bed)
        intersect_set = {(r[0], r[1], r[2]) for r in intersect_regions}
        indices_of_overlap = [
            i for i, r in enumerate(self.bed) if (r[0], r[1], r[2]) in intersect_set
        ]

        interlen = len(indices_of_overlap)
        if num_drop > interlen:
            _LOGGER.warning(
                "Desired regions dropped ({}) is greater than the number of overlaps found ({}). Dropping {} regions.".format(
                    num_drop, interlen, interlen
                )
            )
            num_drop = interlen
        elif interlen > num_drop:
            indices_of_overlap = random.sample(indices_of_overlap, num_drop)

        for idx in sorted(indices_of_overlap, reverse=True):
            del self.bed[idx]
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
            fp (str or list): Path to file, or list of lists, for comparison.
            reference (str or list): Path to file, or list of lists, for reference.
                If None, then defaults to the original BED file provided to the Bedshift constructor.

        Returns:
            list: A list of [chrom, start, end] lists representing overlapping regions.
        """
        # Build reference region data
        if reference is None:
            ref_data = self.original_bed
        elif isinstance(reference, list):
            ref_data = reference
        elif isinstance(reference, str):
            ref_data = self.read_bed(reference)
        else:
            raise Exception("unsupported input type: {}".format(type(reference)))

        # Build comparison region data
        if isinstance(fp, list):
            comp_data = fp
        elif isinstance(fp, str):
            comp_data = self.read_bed(fp)
        else:
            raise Exception("unsupported input type: {}".format(type(fp)))

        # Convert list-of-lists to RegionSet via tempfile
        ref_rs = _list_to_regionset(ref_data)
        comp_rs = _list_to_regionset(comp_data)

        # Use RegionSet overlap detection
        overlap_rs = ref_rs.subset_by_overlaps(comp_rs)

        if len(overlap_rs) == 0:
            raise Exception("no intersection found")

        # Convert back to list of lists
        result = []
        for i in range(len(overlap_rs)):
            region = overlap_rs[i]
            result.append([region.chr, region.start, region.end])
        return result

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

        self._remove_invalid_regions()
        return n

    def to_bed(self, outfile_name):
        """Write regions to a BED file.

        Args:
            outfile_name (str): The name of the output BED file.
        """
        self._remove_invalid_regions()
        self._sort_bed()
        with open(outfile_name, "w") as f:
            for row in self.bed:
                f.write(f"{row[0]}\t{int(row[1])}\t{int(row[2])}\n")

    def read_bed(self, bedfile_path):
        """Read a BED file into a list of lists.

        Args:
            bedfile_path (str): The path to the BED file.

        Returns:
            list: A list of lists, each containing [chrom, start, end, mod_flag].
        """
        try:
            rs = RegionSet(bedfile_path)
        except Exception:
            msg = "File {} could not be read".format(bedfile_path)
            _LOGGER.error(msg)
            raise Exception(msg)

        if len(rs) == 0:
            raise Exception(f"File {bedfile_path} is empty")

        regions = []
        for i in range(len(rs)):
            region = rs[i]
            regions.append([region.chr, region.start, region.end, "-"])
        return regions
