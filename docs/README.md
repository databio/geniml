# Bedshift

Bedshift is a tool for randomly perturbing BED file regions. The perturbations supported on regions are shift, drop, add, cut, and merge. This tool is particularly useful for creating test datasets for various tasks, since there often is no ground truth dataset to compare to. By perturbing a file, a pipeline or analysis can be run on both the perturbed file and the original file, then be compared.

## Installing

The package is available for public download on PyPi.

```
pip install bedshift
```

## Quickstart

The package is available for use both as a command line interface and a python package. To get started, type on the command line:

```
bedshift -h
```

The following examples will shift 10% of the regions and add 10% new regions in `examples/test.bed`. The output is located at `bedshifted_test.bed`.

CLI:

```
bedshift -b tests/test.bed -s 0.1 -a 0.1
```

Python:

```py
import bedshift

bedshifter = bedshift.Bedshift('tests/test.bed')
bedshifter.shift(shiftrate=0.1, shiftmean=0.0, shiftstdev=120.0)
bedshifter.add(addrate=0.1, addmean=320.0, addstdev=20.0)
bedshifter.to_bed('tests/test_output.bed')
```
