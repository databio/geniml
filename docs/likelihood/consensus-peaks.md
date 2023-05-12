# How to create consensus peak set from a collection of BED files?

We will start with simple example of how to creat a consensus peak set from 
collection of files. Example data that will be used can be found in 
```tests/consenus/raw``` . 

## Data preprocessing
First step of analysis is creating three tracks with genome coverage by peaks,
their starts and ends. To do that we have to:
1. install [uniwig](https://github.com/databio/uniwig/tree/smoothing), make sure to use branch dev
2. use [create_unsorted.sh](https://github.com/databio/uniwig/blob/smoothing/create_unsorted.sh) to make three bigWig:
    - {prefix}_start.bw - with smoothed coverage of genome by starts
    - {prefix}_core.bw - with coverage of genome by peaks
    - {prefix}_start.bw - with smoothed coverage of genome by ends

In this tutorial we will use prefix "all" as it is a default prefix in
```gitk``` module
## Coverage universe
We will start by making a coverage universe with cutoff that results in maximum 
likelihood universe. We can do it through CLI:

```
 gitk lh universe_hard --coverage_file tests/consenus/coverage/all_core.bw  \
                       --fout tests/consenus/universe/universe.bed

```  

Where:

- ```--coverage_file```, takes the path to bigWig file with genome coverage by collection 
- ```--fout```, takes the path to output file 

Or we can import it directly into python:
```
from gitk.likelihood.universe_hard import main as cutoff

cutoff("tests/consenus/coverage/all_core.bw",
        fout="tests/consenus/universe/universe.bed")
```

Depending on the task we can also smooth the output universe by setting ``--merge`` 
flag with the distance beloved witch peaks should be merged together and 
``--filter_size`` with minimum size of peak that should be part of the universe. We can also not use the maximum likelihood cut-off and instead of it use user defined cutoff. For that we have to set ``--cutoff`` . If we set it to 1 we get union universe, and when to number of files we will get intersection universe.

## Maximum likelihood universe
Another type of universe that we can make is maximum likelihood flexible universe. To make it first we have to have a likelihood model of genome coverage by collection of files.

#### Making likelihood model:
To make a likelihood model we can use this CLI:

```
gitk lh build_model --model_folder tests/consenus/model.tar \
                    --coverage_folder tests/consenus/coverage/ \
                    --file_no x 
```

Where:

- ```--model_folder```, takes the name of tar archive that will contain the likelihood model
- ```--file_no```, number of files used in analysis
- ```--coverage_folder``` path to folder with coverage tracks

Or, we can do it directly in python:

```
from gitk.likelihood.build_model import main

main("tests/consenus/model.tar", "tests/consesnus/coverage",
     "all",
     file_no=4)
```

#### Making universe:
Now that we have the model we make the universe:

```
gitk lh universe_flexible --model_folder tests/consenus/model.tar \
                          --output_file tests/consenus/universe/universe.bed \
                          --cov_folder tests/consesnus/coverage
```

Where:

- ```--model_folder```, takes the name of tar archive that contains the likelihood model
- ```--output_file```, takes the path to output file 
- ```--cov_folder``` path to folder with coverage tracks

Similarly, we can do it in python:

```
from gitk.likelihood.universe_flexible import main

main("tests/consesnus/model.tar",
     "/home/hee6jn/Documents/gitk/tests/consesnus/coverage",
     "all",
     "tests/consenus/universe/universe.bed")
```

## HMM 
Another approach to making flexible universes is using Hidden Markov Models.
We can do it for example with:

```
gitk hmm --out_file tests/consenus/universe/universe.bed \
         --cov_folder tests/consenus/coverage/ \
         --normlaize \
         --save_max_cove
```

Where:

- ```--out_file```, takes the path to output file 
- ```--cov_folder```, path to folder with coverage tracks
- ```--coverage_prefix``` prefix used in uniwig for making files, default is "all"
- ```--not_normlaize```, is a flag that specifies whether not to normalize tracks before running HMM
- ```--save_max_cove```,  is a flag that specifies whether to save maximum coverage of each output peak

Similarly, we can do it in python:

```
from gitk.hmm.hmm import run_hmm_save_bed

run_hmm_save_bed("tests/consenus/coverage/",
                 "tests/consenus/universe/universe.bed",
                 save_max_cove=True)
```