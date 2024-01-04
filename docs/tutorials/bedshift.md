# Randomizing BED files with bedshift


Bedshift is a tool for randomly perturbing BED file regions. The perturbations supported on regions are shift, drop, add, cut, and merge. This tool is particularly useful for creating test datasets for various tasks, since there often is no ground truth dataset to compare to. By perturbing a file, a pipeline or analysis can be run on both the perturbed file and the original file, then be compared.

## Installing

Bedshift is part of the `geniml` package distributed on PyPI.

```
pip install geniml
```


## Quickstart

The package is available for use both as a command line interface and a python package. To get started, type on the command line:

```
bedshift -h
```

The following examples will shift 10% of the regions and add 10% new regions in `examples/test.bed`. The -l argument is the file in which chromosome sizes are located, and is only required for adding and/or shifting regions. The output is located at `bedshifted_test.bed`.

CLI:

```
bedshift -l hg38.chrom.sizes -b tests/test.bed -s 0.1 -a 0.1
```

Python:

```py
import bedshift

bedshifter = bedshift.Bedshift('tests/test.bed', 'hg38.chrom.sizes')
bedshifter.shift(shiftrate=0.1, shiftmean=0.0, shiftstdev=120.0)
bedshifter.add(addrate=0.1, addmean=320.0, addstdev=20.0)
bedshifter.to_bed('tests/test_output.bed')
```


## Example Repository

If you're looking to use Bedshift in your own experiment, we created an [example repository](https://github.com/databio/bedshift_analysis) containing working code to:

1. Produce a large dataset of Bedshift files
2. Run a pipeline on the dataset and obtain results
3. Aggregate and visualize the results

It integrates the [PEP](http://pep.databio.org/en/latest/) and [looper](http://looper.databio.org/en/latest/) workflow allowing you to easily
run the project out of the box.

## Generate a random BED file

To generate a random BED file, you need to start with is a BED file with one region, which you will delete later. (It will appear at the top of the BED files, so it will be easier to delete later.) On the command line:

```
echo "chr1\t1\t1000" > random.bed
```

The next step is to construct a bedshift command to add regions to this file. We will need to specify the rate of add, which in this case is going to be the number of the new regions we want. Let's try generating 1,000 new regions. Don't forget to specify a chromosome sizes file, which is required for adding regions.

```
bedshift -b random.bed -l hg38.chrom.sizes -a 1000
```

The printed output should say that 1000 regions changed. Finally, go into the output at bedshifted_random.bed and delete the original region. If you specified repeats and have many output files that need to have the original region deleted, here is a handy command to delete the first line of every BED file. (Warning: make sure there are no other BED files in the folder before using this command.)

```
find . -name "*.bed" -exec sed -i '.bak' '1d' {} \;
```

This `find` command will find all BED files and execute a `sed` command to remove the first line. The `sed` command will operate in place and create `.bak` backup files, which can be removed later.


# Shift, Add, and Drop From File

"From file" means that the regions selected to shift, add, or drop are specified from a provided file. These features provide the ability to finely control what regions are perturbed. For example, if you have a BED file specifying exon regions and you want to add only exons, you can use `--addfile`.

## Add from file example

```
bedshift -b mydata.bed -a 0.07 --addfile exons.bed
```

Specifying `--addfile` with `-a` add rate will increase the size of `mydata.bed` by 7% with new regions selected from `exons.bed`.

## Shift from file example

Shift from file first calculates which regions overlap between the specified `--shiftfile` and `--bedfile`, then selects which regions to shift among those overlaps.

```
bedshift -b mydata.bed -s 0.42 --shiftmean 5 --shiftstdev 5 --shiftfile snp.bed
```

In this example, we only want to shift regions that are SNPs. The number of shifted regions is 42% of the total regions in `mydata.bed`. Notice here that unlike `--addfile`, we still have to specify the shift mean and standard deviation. This is because `--shiftfile` tells which regions to shift, but not by how much.

## Drop from file example

Drop from file, like shift from file, calculates overlaps between the specified `--dropfile` and `--bedfile`, then selects regions from those overlaps to drop.

```
bedshift -b mydata.bed -d 0.4 -dropfile snp.bed
```

This command will drop regions that overlap with SNPs. The number of dropped regions is 40% of the total regions in `mydata.bed`.



## Use a YAML File to Specify Perturbations

Sometimes the default settings of bedshift does not allow enough control over perturbations. For example, the order of perturbations is fixed as shift, add, cut, merge, drop, so if you wanted to change the order you would have to specify multiple commands.
The same problem arises when you want to run multiple "add from file" commands - there is just no way to do it using a single command.

This is why we created the YAML config file perturbation option. In the YAML file, users can specify as many perturbations as they want, along with the parameters specific to each perturbation. An example of a YAML config file follows:

```
bedshift_operations:
  - add_from_file:
    file: exons.bed
    rate: 0.2
  - add_from_file:
    file: snp.bed
    rate: 0.05
  - shift_from_file:
    file: exons.bed
    rate: 0.4
    mean: 100
    stdev: 85
  - shift_from_file:
    file: snp.bed
    rate: 0.4
    mean: 2
    stdev: 1
  - merge:
    rate: 0.15
```

The order of perturbations is run in the same order they are specified. So in this example, we add from two different files, then also shift those regions that were just added. Finally we perform a merge at 15% rate.


# How to bedshift all files in a directory

## Using shell

Assuming you are bedshifting all files in the current working directory using the same parameter, use the following shell script (changing parameters as needed), which iterates over files in the directory and applies bedshift:

```
#!/bin/bash
for filename in *.bed; do
	CHROM_LENGTHS=hg38.chrom.sizes
	BEDFILE=$filename
	DROP_RATE=0.3

	ADD_RATE=0.2
	ADD_MEAN=320.0
	ADD_STDEV=30.0

	SHIFT_RATE=0.2
	SHIFT_MEAN=0.0
	SHIFT_STDEV=150.0

	CUT_RATE=0.0
	MERGE_RATE=0.0

	bedshift --bedfile $BEDFILE --chrom-lengths $CHROM_LENGTHS --droprate $DROP_RATE --addrate $ADD_RATE --addmean $ADD_MEAN --addstdev $ADD_STDEV --shiftrate $SHIFT_RATE --shiftmean $SHIFT_MEAN --shiftstdev $SHIFT_STDEV --cutrate $CUT_RATE --mergerate $MERGE_RATE
done
```

## Using Python

In Python, you need the `os` library to get the filenames in a directory. Then you loop through the filenames and apply bedshift.

```py
import bedshift
import os

files = os.listdir('/path/to/data/')
for file in files:
	if file.endswith('.bed'):
		# you may also pass in a chrom.sizes file as the 
		# second argument if you are adding or shifting regions
		b = bedshift.Bedshift(file)
		b.all_perturbations(cutrate=0.3, droprate=0.2)
		b.to_bed('bedshifted_' + file)
```


# Add Random Regions Only in Valid Regions

Using the basic `--add` option, regions are added randomly onto any chromosome at any location, without any regard for non-coding regions. For use cases of Bedshift more rooted in biology, this effect is not desirable. The `--add-valid` option gives the user the ability to specify a BED file indicating areas where it is valid to add regions. Thus, if an `--add-valid` file has only coding regions, then regions will be randomly added only in those areas. Here is an example:

```
bedshift -b mydata.bed -a 0.5 --add-valid coding.bed --addmean 500 --addstdev 200
```

`coding.bed` contains large regions of the genome which are coding. Added regions can be anywhere inside of those regions. In addition, the method considers the size of the valid regions in deciding where the new regions will be added, so the smaller valid regions will contain proportionally less new regions than the larger valid regions.