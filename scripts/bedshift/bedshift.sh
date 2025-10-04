#!/bin/bash

# Change NUM_BEDFILES, BEDFILE, and OUTPUT_FILE.
# Other parameters are set to the default values. Edit as needed.
# To run, execute ./bedscript.sh in Terminal.

NUM_BEDFILES=NUMBER
for (( c=1; c<=$NUM_BEDFILES; c++ ))
do
	BEDFILE=PATH/TO/ORIGINAL/FILE$c.bed
	DROP_RATE=0.0

	ADD_RATE=0.0
	ADD_MEAN=320.0
	ADD_STDEV=30.0

	SHIFT_RATE=0.0
	SHIFT_MEAN=0.0
	SHIFT_STDEV=150.0

	CUT_RATE=0.0
	MERGE_RATE=0.0
	OUTPUT_FILE=PATH/TO/PERTURBED/FILE$c.bed

	geniml bedshift --bedfile $BEDFILE --droprate $DROP_RATE --addrate $ADD_RATE --addmean $ADD_MEAN --addstdev $ADD_STDEV --shiftrate $SHIFT_RATE --shiftmean $SHIFT_MEAN --shiftstdev $SHIFT_STDEV --cutrate $CUT_RATE --mergerate $MERGE_RATE --outputfile $OUTPUT_FILE
done
