import unittest
import os

from bedshift import bedshift
from bedshift import BedshiftYAMLHandler


class TestBedshift(unittest.TestCase):
    def setUp(self):
        self.bs = bedshift.Bedshift('tests/test.bed', chrom_sizes="tests/hg38.chrom.sizes")

    def test_read_bed(self):
        reader = bedshift.Bedshift('tests/header_test.bed')
        self.assertTrue(list(reader.bed.columns), [0,1,2,3])
        self.assertTrue(list(reader.bed.index), [0,1,2])

    def test_read_chrom_sizes(self):
        self.bs._read_chromsizes("tests/hg19.chrom.sizes")
        self.assertEqual(len(self.bs.chrom_lens), 93)

    def test_add(self):
        added = self.bs.add(0.1, 100, 20)
        self.assertEqual(added, 1000)
        self.bs.reset_bed()

    def test_check_rate(self):
        self.assertRaises(SystemExit, self.bs.shift, -0.1, 250, 250)
        self.assertRaises(SystemExit, self.bs.cut, 1.5)

    def test_add_high_rate(self):
        added = self.bs.add(1.23, 500, 123)
        self.assertEqual(added, 12300)
        self.bs.reset_bed()

    def test_add_valid_regions(self):
        added = self.bs.add(0.5, 2000, 1000, valid_bed='tests/small_test.bed', delimiter='\t')
        self.assertEqual(added, 5000)
        self.bs.to_bed('tests/add_valid_test.bed')
        self.bs.reset_bed()

    def test_add_from_file(self):
        added = self.bs.add_from_file("tests/test.bed", 0.25)
        self.assertEqual(added, 2500)
        self.bs.reset_bed()

    def test_drop(self):
        dropped = self.bs.drop(0.315)
        self.assertEqual(dropped, 3150)
        self.bs.reset_bed()

    def test_shift(self):
        shifted = self.bs.shift(0.129, 200, 30)
        self.assertAlmostEqual(shifted, 1290, places=-2)
        self.bs.reset_bed()

    def test_cut(self):
        cut = self.bs.cut(0.909)
        self.assertEqual(cut, 9090)
        self.bs.reset_bed()

    def test_merge(self):
        merged = self.bs.merge(0.2)
        self.assertAlmostEqual(merged, 4000, places=-3)
        self.bs.reset_bed()

    def test_combo(self):
        _ = self.bs.drop(0.4)
        _ = self.bs.add(0.2, 200, 10)
        self.assertEqual(len(self.bs.bed), 7200)
        self.bs.reset_bed()

    def test_drop_from_file(self):
        dropped = self.bs.drop_from_file("tests/test.bed", 0.25)
        self.assertEqual(dropped, 2500)
        self.bs.reset_bed()

    def test_drop_from_file_high_rate(self):
        dropped = self.bs.drop_from_file("tests/test.bed", 1)
        self.assertEqual(dropped, 10000)
        self.bs.reset_bed()

    def test_drop_from_file_zero_rate(self):
        dropped = self.bs.drop_from_file("tests/test.bed", 0)
        self.assertEqual(dropped, 0)
        self.bs.reset_bed()

    def test_all_perturbations1(self):
        perturbed = self.bs.all_perturbations(
                            addrate=0.5, addmean=320.0, addstdev=20.0,
                            shiftrate=0.23, shiftmean=-10.0, shiftstdev=120.0,
                            cutrate=0.12,
                            droprate=0.42)
        self.assertAlmostEqual(perturbed, 16156, places=-2)
        self.assertAlmostEqual(len(self.bs.bed), 9744, places=-2)
        self.bs.reset_bed()

    def test_all_perturbations2(self):
        perturbed = self.bs.all_perturbations(
                            addrate=0.3, addmean=320.0, addstdev=20.0,
                            shiftrate=0.3, shiftmean=-10.0, shiftstdev=120.0,
                            cutrate=0.1,
                            mergerate=0.11,
                            droprate=0.03)
        # merge sometimes merges more or less than expected because it depends
        # if the randomly chosen regions are adjacent
        self.assertAlmostEqual(perturbed, 9400, places=-3)

    def test_to_bed(self):
        self.bs.to_bed('tests/py_output.bed')
        self.assertTrue(os.path.exists('tests/py_output.bed'))

    def test_small_file(self):
        bs_small = bedshift.Bedshift('tests/small_test.bed', chrom_sizes="tests/hg38.chrom.sizes")
        shifted = bs_small.shift(0.3, 50, 50)
        self.assertEqual(shifted, 0)
        shifted = bs_small.shift(1.0, 50, 50)
        self.assertEqual(shifted, 1)
        added = bs_small.add(0.2, 100, 50)
        self.assertEqual(added, 0)
        added = bs_small.add(1.0, 100, 50)
        self.assertEqual(added, 1)
        added = bs_small.add(2.0, 100, 50)
        self.assertEqual(added, 4)


class TestBedshiftYAMLHandler(unittest.TestCase):
    def test_handle_yaml(self):
        bedshifter = bedshift.Bedshift('tests/test.bed', chrom_sizes="tests/hg38.chrom.sizes")
        yamled = BedshiftYAMLHandler.BedshiftYAMLHandler(bedshifter=bedshifter, yaml_fp="tests/bedshift_analysis.yaml").handle_yaml()
        bedshifter.reset_bed()

        added = bedshifter.add(addrate=0.1, addmean=100, addstdev=20)
        f_drop_10 = bedshifter.drop_from_file(fp="tests/test.bed", droprate=0.1)
        f_shift_30 = bedshifter.shift_from_file(fp="tests/test2.bed",
                                            shiftrate=0.50,
                                            shiftmean=100,
                                            shiftstdev=200)
        f_added_20 = bedshifter.add_from_file(fp="tests/small_test.bed", addrate=0.2)
        cut = bedshifter.cut(cutrate=0.2)
        shifted = bedshifter.shift(shiftrate=0.3, shiftmean=100, shiftstdev=200)
        dropped = bedshifter.drop(droprate=0.3)
        merged = bedshifter.merge(mergerate=0.15)

        total = added+f_drop_10+f_shift_30+f_added_20+cut+dropped+shifted+merged

        # yamled and total both should be around 16750, but can vary by over 100
        self.assertAlmostEqual(yamled, total, places=-3)
        bedshifter.reset_bed()
