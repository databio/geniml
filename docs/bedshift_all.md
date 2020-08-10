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

In Python, you need the `os` library to get the filenames in a directory. After that, running bedshift is as easy as passing parameters.

```py
import bedshift
import os


datafolder = '.'

files = os.listdir()
for file in files:
	if file.endswith('.bed'):
		# you may also pass in a chrom.sizes file as the 
		# second argument if you are adding or shifting regions
		b = bedshift.Bedshift(file)
		b.all_perturbations(cutrate=0.3, droprate=0.2)
		b.to_bed('bedshifted_' + file)
```
