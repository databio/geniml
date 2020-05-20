import os, sys
import bedshift

os.chdir(sys.path[0])

bedshifter = bedshift.Bedshift('test.bed')

bedshifter.all_perturbations(addrate=0.3, addmean=320.0, addstdev=20.0,
							 shiftrate=0.3, shiftmean=-10.0, shiftstdev=120.0,
							 cutrate=0.1, 
							 mergerate=0.11, 
							 droprate=0.03)
bedshifter.to_bed('py_output.bed')
bedshifter.reset_bed()

bedshifter.all_perturbations(addrate=0.3, addmean=320.0, addstdev=20.0, 
							 addfile='py_output.bed', 
							 shiftrate=0.3, shiftmean=100.0, shiftstdev=30.0)
bedshifter.to_bed('py_output2.bed')
