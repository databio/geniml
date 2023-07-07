# How to assess universe fit to collection of BED files

Given a collection of genomic interval sets, and a proposed universe, we would like to assess how well the fits the genomic interval sets.
We will use universes produced [earlier](consensus-peaks.md). This module provides several complementary methods to assess fit.


We can assess fit either from CLI, or from within Python. Both overlap and distance based assessments can be run using: `gitk assess ...` with appropriate flags.

```
 gitk assess --assessment-method1 \
             --assessment-method2 \
             --...
             --raw-data-folder tests/consesnus/raw/\
             --file-list tests/consesnus/file_list.txt \
             --universe tests/consenus/universe/universe.bed \
             --save-to-file \
             --folder-out tests/consesnus/results/intersection/ \
             --pref test \
             --no-workers 1
```
Where:

- ``--raw-data-folder``, takes the path to folder with files from the collection
- ``--file-list``, takes the path to file with list of files
- ``--universe``, takes the path to file with the assessed universe
- ``--save-to-file``,  is a flag that specifies whether to out put table with each row 
containing file name and results of chosen metrics
- ``--folder-out``, takes the path to folder in which put the output file
- ``--pref``, takes a prefix of output file name
- ``--no-workers``, takes the number of workers that should be used
- ``--save-each``, is a flag that specifies whether to save between the closest peaks to file

## Base-level overlap measure

First test checks how much of our file is present in the universe and how much additional information is present in the universe. We can check that by adding ```--overlap``` to ```gitk assess ...```. In the result files it will output columns with: number of bp in universe but not in file, number of bp in file but not the universe, and number of bp both in universe and file.

We can also use it directly from Python like this:

```
from gitk.assess.intersection import run_intersection

run_intersection("test/consesnus/raw/",
                        "tests/consesnus/file_list.txt",
                        "tests/consenus/universe/universe.bed",
                        no_workers=1)
```

Or, we can calculate F10 score of the universe using:

```
from gitk.assess.intersection import get_f_10_score

get_f_10_score("test/consesnus/raw/",
               "tests/consesnus/file_list.txt",
               "tests/consenus/universe/universe.bed",
               no_workers=1)
```

## Region boundary distance measure

Next, we can calculate the distance between query and universe. To do that we can choose from :
 - ```distance``` - calculates distance from region in query to the nearest region in the universe
 - ```distance-universe-to-file```- calculates distance from region in query to the nearest region in the universe accounting for universe flexibility
 - ```distance-flexible``` - calculates distance from region in universe to the nearest region in the query
 - ```distance-flexible-universe-to-file``` - calculates distance from region in universe to the nearest region in the query accounting for universe flexibility


All presented distance measures can be done using python, which will result in matrix where first column is file names and the second one is median of distances. 

```
from gitk.assess.distance import run_distance

d_median = run_distance("tests/consesnus/raw",
                  "tests/consesnus/file_list.txt",
                  "tests/consesnus/universe/universe.bed",
                  npool=2)
```
Additionally, we can directly calculate the closeness score using:

```
from gitk.assess.distance import get_closeness_score

closeness_score = get_closeness_score("tests/consesnus/raw",
                                      "tests/consesnus/file_list.txt",
                                      "tests/consesnus/universe/universe.bed",
                                      no_workers=2)
```


## Universe likelihood

We can also calculate the likelihood of universe given collection of file. For that we
will need [likelihood model](consensus-peaks.md#making-likelihood-model-). We can do it
either for hard universe:

```
from gitk.assess.likelihood import hard_universe_likelihood

lh_hard = hard_universe_likelihood("tests/consesnus/lh_model.tar",
                         "tests/consenus/universe/universe.bed",
                         "tests/consesnus/coverage", "all")
```

or with taking into account universe flexibility:

```
from gitk.assess.likelihood import likelihood_flexible_universe

lh_flexible = likelihood_flexible_universe("tests/consesnus/lh_model.tar",
                         "tests/consenus/universe/universe.bed",
                         "tests/consesnus/coverage", "all")
```