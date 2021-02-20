# Bedshift

Docs: bedshift.databio.org

Install from PyPi: `pip install bedshift`

Install from local repository: `pip install .`

## Command line

Run with: `bedshift -l hg38.chrom.sizes -b BEDFILE`

See `bedshift -h` for parameters.

## Python

```py
import bedshift

bedshifter = bedshift.Bedshift('tests/test.bed', 'hg38.chrom.sizes')
bedshifter.all_perturbations(addrate=0.3, addmean=320.0, addstdev=20.0, 
							 shiftrate=0.3, shiftmean=-10.0, shiftstdev=120.0, 
							 cutrate=0.1, 
							 mergerate=0.11, 
							 droprate=0.03)
# can also run single operations: shift, add, cut, merge, drop

bedshifter.to_bed('test_output.bed')
```



## Development

test changes:

```
./run-tests.sh
```

Double check the output files to see if the regions make sense.

build docs:



