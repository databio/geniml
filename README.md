# Bedshift

Install with: `pip install --user .`

## Command line

Run with: `bedshift -b BEDFILE` or `./bedshift.sh` if running bedshift on multiple bedfiles with a set of parameters, which are editable in bedshift.sh.

See: `bedshift --help` for parameters.

## Python

```py
import bedshift

bedshifter = bedshift.Bedshift()

df = bedshifter.read_bed('test.bed')
df = bedshifter.all_perturbations(df, addrate=0.3, addmean=320.0, addstdev=20.0, shiftrate=0.3, shiftmean=-10.0, shiftstdev=120.0, cutrate=0.1, mergerate=0.11, droprate=0.03)
# can also run single operations: shift, add, cut, merge, drop

bedshifter.write_bed(df, 'output_file.bed')
```




## Development

test changes:

```
pip install --user .

./run-tests.sh
```

if the tests complete, then bedshift is working properly
