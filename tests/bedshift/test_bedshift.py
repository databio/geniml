import os
import pytest

from geniml.bedshift import bedshift
from geniml.bedshift import BedshiftYAMLHandler

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

class TestBedshift():
    def test_read_bed(self):
        reader = bedshift.Bedshift(os.path.join(SCRIPT_PATH, "header_test.bed"))
        assert list(reader.bed.columns) == [0, 1, 2, 3]
        assert list(reader.bed.index) == [0, 1, 2]

    def test_read_chrom_sizes(self, bs):
        bs._read_chromsizes(os.path.join(SCRIPT_PATH, "hg19.chrom.sizes"))
        assert len(bs.chrom_lens) == 93

    def test_add(self, bs):
        added = bs.add(0.1, 100, 20)
        assert added == 100
        bs.reset_bed()

    def test_check_rate(self, bs):
        with pytest.raises(ValueError):
            bs.shift(-0.1, 250, 250)
        with pytest.raises(ValueError):
            bs.cut(1.5)

    def test_add_high_rate(self, bs):
        added = bs.add(1.23, 500, 123)
        assert added == 1230
        bs.reset_bed()

    def test_add_valid_regions(self, bs):
        added = bs.add(
            0.5, 2000, 1000, valid_bed=os.path.join(SCRIPT_PATH, "small_test.bed"), delimiter="\t"
        )
        assert added == 500
        # bs.to_bed(os.path.join(SCRIPT_PATH, "add_valid_test.bed"))
        bs.reset_bed()

    def test_add_from_file(self, bs):
        added = bs.add_from_file(os.path.join(SCRIPT_PATH, "test.bed"), 0.25)
        assert added == 250
        bs.reset_bed()

    def test_drop(self, bs):
        dropped = bs.drop(0.315)
        assert dropped == 315
        bs.reset_bed()

    def test_shift(self, bs):
        shifted = bs.shift(0.129, 200, 30)
        assert shifted == pytest.approx(129, 2)
        bs.reset_bed()

    def test_cut(self, bs):
        cut = bs.cut(0.909)
        assert cut == 909
        bs.reset_bed()

    def test_merge(self, bs):
        merged = bs.merge(0.2)
        assert merged == pytest.approx(400, 3)
        bs.reset_bed()

    def test_combo(self, bs):
        _ = bs.drop(0.4)
        _ = bs.add(0.2, 200, 10)
        assert len(bs.bed) == 720
        bs.reset_bed()

    @pytest.mark.skip("Not implemented yet")
    def test_drop_from_file(self, bs):
        dropped = bs.drop_from_file(os.path.join(SCRIPT_PATH, "test.bed"), 0.25)
        self.assertEqual(dropped, 250)
        bs.reset_bed()

    @pytest.mark.skip("Not implemented yet")
    def test_drop_from_file_high_rate(self, bs):
        dropped = bs.drop_from_file(os.path.join(SCRIPT_PATH, "test.bed"), 1)
        assert dropped == 100
        bs.reset_bed()

    @pytest.mark.skip("Not implemented yet")
    def test_drop_from_file_zero_rate(self, bs):
        dropped = bs.drop_from_file(os.path.join(SCRIPT_PATH, "test.bed"), 0)
        assert dropped ==  0
        bs.reset_bed()

    @pytest.mark.skip("Not implemented yet")
    def test_all_perturbations1(self, bs):
        perturbed = bs.all_perturbations(
            addrate=0.5,
            addmean=320.0,
            addstdev=20.0,
            shiftrate=0.23,
            shiftmean=-10.0,
            shiftstdev=120.0,
            cutrate=0.12,
            droprate=0.42,
        )
        assert perturbed == pytest.approx(16156, 2)
        assert len(bs.bed) == pytest.approx(9744, 2)
        bs.reset_bed()

    @pytest.mark.skip("Not implemented yet")
    def test_all_perturbations2(self, bs):
        perturbed = bs.all_perturbations(
            addrate=0.3,
            addmean=320.0,
            addstdev=20.0,
            shiftrate=0.3,
            shiftmean=-10.0,
            shiftstdev=120.0,
            cutrate=0.1,
            mergerate=0.11,
            droprate=0.03,
        )
        # merge sometimes merges more or less than expected because it depends
        # if the randomly chosen regions are adjacent
        assert perturbed == pytest.approx(9400, 3)

    @pytest.mark.skip("Not implemented yet")
    def test_to_bed(self, tmp_path, bs):
        bs.to_bed(os.path.join(tmp_path, "py_output.bed"))
        assert os.path.exists(os.path.join(tmp_path, "py_output.bed"))

    def test_small_file(self):
        bs_small = bedshift.Bedshift(
            os.path.join(SCRIPT_PATH, "small_test.bed"),
            chrom_sizes=os.path.join(SCRIPT_PATH, "hg38.chrom.sizes"),
        )
        shifted = bs_small.shift(0.3, 50, 50)
        assert shifted == 0
        shifted = bs_small.shift(1.0, 50, 50)
        assert shifted == 1
        added = bs_small.add(0.2, 100, 50)
        assert added == 0
        added = bs_small.add(1.0, 100, 50)
        assert added == 1
        added = bs_small.add(2.0, 100, 50)
        assert added == 4


class TestBedshiftYAMLHandler():
    @pytest.mark.skip("Not implemented yet")
    def test_handle_yaml(self):
        bedshifter = bedshift.Bedshift(
            os.path.join(SCRIPT_PATH, "test.bed"),
            chrom_sizes=os.path.join(SCRIPT_PATH, "hg38.chrom.sizes"),
        )
        yamled = BedshiftYAMLHandler.BedshiftYAMLHandler(
            bedshifter=bedshifter, yaml_fp=os.path.join(SCRIPT_PATH, "bedshift_analysis.yaml")
        ).handle_yaml()
        bedshifter.reset_bed()

        added = bedshifter.add(addrate=0.1, addmean=100, addstdev=20)
        f_drop_10 = bedshifter.drop_from_file(
            fp=os.path.join(SCRIPT_PATH, "test.bed"), droprate=0.1
        )
        f_shift_30 = bedshifter.shift_from_file(
            fp=os.path.join(SCRIPT_PATH, "test2.bed"),
            shiftrate=0.50,
            shiftmean=100,
            shiftstdev=200,
        )
        f_added_20 = bedshifter.add_from_file(
            fp=os.path.join(SCRIPT_PATH, "small_test.bed"), addrate=0.2
        )
        cut = bedshifter.cut(cutrate=0.2)
        shifted = bedshifter.shift(shiftrate=0.3, shiftmean=100, shiftstdev=200)
        dropped = bedshifter.drop(droprate=0.3)
        merged = bedshifter.merge(mergerate=0.15)

        total = added + f_drop_10 + f_shift_30 + f_added_20 + cut + dropped + shifted + merged

        # yamled and total both should be around 16750, but can vary by over 100
        assert yamled == pytest.approx(total, 3)
        bedshifter.reset_bed()
