#!/usr/bin/python

import os, sys
from bedshift import bedshift

os.chdir(sys.path[0])

bedshifter = bedshift.Bedshift()

df = bedshifter.read_bed('test.bed')
original_rows = df.shape[0]
df = bedshifter.all_perturbations(df, 0.3, 320.0, 20.0, 0.3, -10.0, 120.0, 0.1, 0.11, 0.03)

bedshifter.write_bed(df, 'py_output.bed')
