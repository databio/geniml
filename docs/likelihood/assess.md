# How to assess universe fit to collection of files?
Given a universe it is immportant to assess it fit to collection of data. 
We will use here universes produced [eriler](consensus-peaks.md).


## Intersection assessment
First test checks how much of our file is present in the universe and how much 
additional information is present in the univers. We can check that with:

```
gitk assess intersection --raw_data_folder tests/consesnus/raw/\
                         --file_list tests/consesnus/file_list.txt \
                         --universe tests/consenus/universe/universe.bed \
                         --save_to_file \
                         --folder_out tests/consesnus/results/intersection/ \
                         --pref test \
                         --no_workers 1
```
Where:

- ``--raw_data_folder``, takes the path to folder with files from the collection
- ``--file_list``, takes the path to file with list of files
- ``--universe``, takes the path to file with the assessed universe
- ``--save_to_file``,  is a flag that specifies whether to out put table with each row 
containing file name, number of bp in univers but not in file, number of bp in file 
but not the univers, and number of bp both in univers and file
- ``--folder_out``, takes the path to folder in which put the output file
- ``--pref``, takes a prefix of output file name
- ``--no_workers``, takes the number of workers that should be used

Similarly, we can do it in python:

```
from gitk.assess.intersection import run_intersection

F_10 = run_intersection("test/data/raw",
                        "tests/consesnus/file_list.txt",
                        "tests/consenus/universe/universe.bed",
                        no_workers=1)
```
where we can directly return F10 score of the universe. 

## Distance distribution assessment
```
 gitk assess distance --raw_data_folder tests/consesnus/raw/\
                      --file_list tests/consesnus/file_list.txt \
                      --universe tests/consenus/universe/universe.bed \
                      --save_to_file \
                      --folder_out tests/consesnus/results/distance/ \
                      --pref test_flex \
                      --npool 1 \
                      --save_each \
                      --flexible
```

Where:

- ``--raw_data_folder``, takes the path to folder with files from the collection
- ``--file_list``, takes the path to file with list of files
- ``--universe``, takes the path to file with the assessed universe
- ``--save_to_file``,  is a flag that specifies whether to out put table with each row 
containing file name, median distance to start, median distance to end
- ``--folder_out``, takes the path to folder in which put the output file
- ``--pref``, takes a prefix of output file name
- ``--no_workers``, takes the number of workers that should be used
- ``--save_each``, is a flag that specifies whether to save for each peak in the file
distance to the closest peak in the universe
- ``--flexible``, is a flag that specifies whether we should treat the univers as 
a flexible one

Similarly, we can do it in python:

```
from gitk.assess.distance import run_distance

d_median = run_distance("test/data/raw",
                  "test/data/file_list.txt",
                  "test/results/universe/ML_hard.bed",
                  npool=2)
```
where we can directly return average of median of distances to start and ends. 

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