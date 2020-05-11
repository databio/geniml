import os, sys
import bedshift

os.chdir(sys.path[0])

bedshifter = bedshift.Bedshift()

df = bedshifter.read_bed('test.bed')
original_rows = df.shape[0]

df = bedshifter.all_perturbations(df, addrate=0.3, addmean=320.0, addstdev=20.0, addfile=None, shiftrate=0.3, shiftmean=-10.0, shiftstdev=120.0, cutrate=0.1, mergerate=0.11, droprate=0.03)
bedshifter.write_bed(df, 'py_output.bed')

df = bedshifter.all_perturbations(df, addrate=0.3, addmean=320.0, addstdev=20.0, addfile='py_output.bed', shiftrate=0.3, shiftmean=100.0, shiftstdev=30.0)
bedshifter.write_bed(df, 'py_output2.bed')
