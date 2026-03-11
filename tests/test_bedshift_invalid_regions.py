"""Tests for bedshift invalid region bug (GitHub issue #49).

Validates that bedshift never produces regions where start >= end,
even after many rounds of perturbation.
"""

import os
import random
import tempfile

import numpy as np
import pytest

from geniml.bedshift import bedshift

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bedshift")


def _count_invalid_regions(bed):
    """Count regions where start >= end."""
    return sum(1 for row in bed if row[1] >= row[2])


def _make_bedfile(regions):
    """Write regions to a temp BED file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False)
    for chrom, start, end in regions:
        f.write(f"{chrom}\t{start}\t{end}\n")
    f.close()
    return f.name


@pytest.fixture
def bs():
    return bedshift.Bedshift(
        os.path.join(DATA_FOLDER_PATH, "test.bed"),
        chrom_sizes=os.path.join(DATA_FOLDER_PATH, "hg38.chrom.sizes"),
    )


class TestCutTinyRegions:
    """Test that _cut handles regions too small to cut."""

    def _make_bs_with_regions(self, regions):
        """Create a Bedshift object with specific regions."""
        path = _make_bedfile(regions)
        bs = bedshift.Bedshift(
            path,
            chrom_sizes=os.path.join(DATA_FOLDER_PATH, "hg38.chrom.sizes"),
        )
        os.unlink(path)
        return bs

    def test_cut_1bp_region(self):
        """A 1bp region (end - start = 1) cannot be cut; should be skipped."""
        bs = self._make_bs_with_regions([("chr1", 100, 101)])
        # _cut should return (None, None) for uncuttable region
        result = bs._cut(0)
        drop_row, new_regions = result
        # Either skipped (None, None) or valid regions
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_2bp_region(self):
        """A 2bp region can be cut into two 1bp regions."""
        bs = self._make_bs_with_regions([("chr1", 100, 102)])
        drop_row, new_regions = bs._cut(0)
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_5bp_region(self):
        bs = self._make_bs_with_regions([("chr1", 100, 105)])
        drop_row, new_regions = bs._cut(0)
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_10bp_region(self):
        bs = self._make_bs_with_regions([("chr1", 100, 110)])
        drop_row, new_regions = bs._cut(0)
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_15bp_region(self):
        bs = self._make_bs_with_regions([("chr1", 100, 115)])
        drop_row, new_regions = bs._cut(0)
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_19bp_region(self):
        """19bp region triggers the old fallback bug."""
        bs = self._make_bs_with_regions([("chr1", 100, 119)])
        drop_row, new_regions = bs._cut(0)
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_20bp_region(self):
        bs = self._make_bs_with_regions([("chr1", 100, 120)])
        drop_row, new_regions = bs._cut(0)
        if drop_row is not None:
            for region in new_regions:
                assert region[1] < region[2], f"Invalid cut result: {region}"

    def test_cut_many_tiny_regions(self):
        """Cut a batch of tiny regions; all results must be valid."""
        regions = [
            ("chr1", 1000 + i * 100, 1000 + i * 100 + size)
            for i, size in enumerate([1, 2, 3, 5, 8, 10, 15, 19, 20])
        ]
        bs = self._make_bs_with_regions(regions)
        bs.cut(1.0)  # cut all
        assert _count_invalid_regions(bs.bed) == 0, (
            f"Found {_count_invalid_regions(bs.bed)} invalid regions after cutting tiny regions"
        )


class TestMergeUnsorted:
    """Test that merge works correctly even when data is not positionally sorted."""

    def _make_bs_with_regions(self, regions):
        path = _make_bedfile(regions)
        bs = bedshift.Bedshift(
            path,
            chrom_sizes=os.path.join(DATA_FOLDER_PATH, "hg38.chrom.sizes"),
        )
        os.unlink(path)
        return bs

    def test_merge_reversed_order(self):
        """Two same-chrom regions where the second has smaller coordinates."""
        # After sorting by the constructor, these will be ordered.
        # But we can unsort them manually to simulate post-perturbation state.
        regions = [("chr1", 1000, 2000), ("chr1", 500, 900)]
        bs = self._make_bs_with_regions(regions)
        # Manually unsort to simulate post-perturbation state
        bs.bed = [
            ["chr1", 5000, 6000, "-"],
            ["chr1", 500, 900, "-"],
        ]
        drop_rows, merged = bs._merge(0)
        if drop_rows is not None:
            assert merged[1] < merged[2], f"Invalid merged region: {merged}"

    def test_merge_via_public_method(self):
        """The public merge method should produce only valid regions."""
        regions = [
            ("chr1", 1000, 2000),
            ("chr1", 5000, 6000),
            ("chr1", 500, 900),
            ("chr1", 3000, 4000),
        ]
        bs = self._make_bs_with_regions(regions)
        # Scramble internal order
        random.shuffle(bs.bed)
        bs.merge(0.5)
        assert _count_invalid_regions(bs.bed) == 0


class TestAddNegativeLength:
    """Test that add never produces regions with end <= start."""

    def test_add_with_high_stdev(self, bs):
        """High stdev relative to mean can produce negative lengths."""
        random.seed(42)
        np.random.seed(42)
        bs.add(0.5, addmean=10, addstdev=100)
        invalid = _count_invalid_regions(bs.bed)
        assert invalid == 0, f"Found {invalid} invalid regions after add with high stdev"

    def test_add_with_zero_mean(self, bs):
        """Zero mean with any stdev will frequently produce negative lengths."""
        random.seed(123)
        np.random.seed(123)
        bs.add(0.5, addmean=0, addstdev=50)
        invalid = _count_invalid_regions(bs.bed)
        assert invalid == 0, f"Found {invalid} invalid regions after add with zero mean"


class TestStressMultiRound:
    """Stress test: many rounds of all_perturbations should never produce invalid regions."""

    def test_100_rounds(self, bs):
        """Run 100 rounds of perturbations on 1000 regions.

        This is the primary reproducer for GitHub issue #49.
        """
        random.seed(42)
        np.random.seed(42)
        for round_num in range(100):
            bs.all_perturbations(
                addrate=0.1,
                addmean=320.0,
                addstdev=30.0,
                shiftrate=0.1,
                shiftmean=0.0,
                shiftstdev=150.0,
                cutrate=0.1,
                mergerate=0.05,
                droprate=0.1,
            )
            invalid = _count_invalid_regions(bs.bed)
            assert invalid == 0, (
                f"Round {round_num}: found {invalid} invalid regions out of {len(bs.bed)} total"
            )


class TestValidation:
    """Test the _validate_region utility and to_bed safety net."""

    def test_validate_region_valid(self, bs):
        assert bs._validate_region(0, 100) is True
        assert bs._validate_region(50, 51) is True

    def test_validate_region_invalid(self, bs):
        assert bs._validate_region(100, 50) is False  # start > end
        assert bs._validate_region(100, 100) is False  # start == end
        assert bs._validate_region(-1, 100) is False  # negative start

    def test_to_bed_filters_invalid(self, bs, tmp_path):
        """to_bed should not write invalid regions even if they exist internally."""
        # Inject an invalid region
        bs.bed.append(["chr1", 5000, 3000, "X"])
        outfile = os.path.join(tmp_path, "out.bed")
        bs.to_bed(outfile)
        # Read back and verify no invalid regions
        with open(outfile) as f:
            for line in f:
                parts = line.strip().split("\t")
                start, end = int(parts[1]), int(parts[2])
                assert start < end, f"Invalid region in output: {line.strip()}"
