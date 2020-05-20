# Bedshift

Bedshift is a tool for randomly perturbing bedfiles. Different kinds of perturbations supported are region shifts, drops, adds, cuts, and merges. This tool is particularly useful for creating test datasets for various tasks, since there often is no ground truth dataset to compare to. By perturbing a file, analysis can be done on both the perturbed file and the original file and be compared.

## Installing

This package is not available on pypi yet, but is available for local install by cloning the repository.

```
git clone https://github.com/databio/bedshift.git
cd bedshift
pip install .
```

## Quickstart

The package is available for use both as a command line interface and a python package.

The following examples will shift 10% of the regions and add 10% new regions in `examples/test.bed`. The output is located at `bedshifted_test.bed`.

CLI:

```
bedshift -h
bedshift -b tests/test.bed -p 0.1 -a 0.1
```

Python:

```py
import bedshift

bedshifter = bedshift.Bedshift('tests/test.bed')
bedshifter.shift(shiftrate=0.1, shiftmean=0.0, shiftstdev=120.0)
bedshifter.add(addrate=0.1, addmean=320.0, addstdev=20.0)
bedshifter.to_bed('tests/test_output.bed')
```
