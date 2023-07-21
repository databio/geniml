# How to build a new universe

We will start with simple example of how to create a consensus peak set from a
collection of files. Example data can be found in `tests/consenus/raw`. 

## Data preprocessing

First step of analysis is creating three tracks with genome coverage by peaks,
their starts and ends. To do that we have to:

1. install [uniwig](https://github.com/databio/uniwig/tree/smoothing), make sure to use branch dev
2. use [create_unsorted.sh](https://github.com/databio/uniwig/blob/smoothing/create_unsorted.sh) to make three bigWig:
    - {prefix}_start.bw - with smoothed coverage of genome by starts
    - {prefix}_core.bw - with coverage of genome by peaks
    - {prefix}_start.bw - with smoothed coverage of genome by ends

In this tutorial we will use prefix "all" as it is a default prefix in
`gitk` module

## Coverage cutoff universe

We will start by making a coverage universe with cutoff that results in maximum 
likelihood universe. We can build it through CLI:

```console
 gitk build-universe cc --coverage-folder tests/consenus/coverage/ \
                        --output-file tests/consenus/universe/universe.bed

```  

Where:

- `--coverage-folder`, takes the path to bigWig folder with genome coverage by collection 
- `--output-file`, takes the path to output file 

Or we can import it directly into Python:

```
from gitk.universe.cc_universe import cc_universe

cc_universe("tests/consenus/coverage/all_core.bw",
        file_out="tests/consenus/universe/universe.bed")
```

Depending on the task we can also smooth the output universe by setting `--merge` 
flag with the distance beloved witch peaks should be merged together and 
`--filter-size` with minimum size of peak that should be part of the universe. We can also not use the maximum likelihood cut-off and instead of it use user defined cutoff. For that we have to set `--cutoff` . If we set it to 1 we get union universe, and when to number of files we will get intersection universe.

## Coverage cutoff flexible universe
Next presented universe is coverage cutoff flexible universe. We can do it through CLI:

```
 gitk build-universe ccf --coverage-folder tests/consenus/coverage/ \
                       --output-file tests/consenus/universe/universe.bed

```  

Where:

- `--coverage-folder`, takes the path to bigWig file with genome coverage by collection 
- `--output-file`, takes the path to output file 

Or we can import it directly into python:
```
from gitk.universe.ccf_universe import ccf_universe

ccf_universe("tests/consenus/coverage/all_core.bw",
        file_out="tests/consenus/universe/universe.bed")
```

## Maximum likelihood universe
Another type of universe that we can make is maximum likelihood flexible universe. To make it first we have to have a likelihood model of genome coverage by collection of files.

#### Making likelihood model:
To make a likelihood model we can use this CLI:

```
gitk lh build_model --model-file tests/consenus/model.tar \
                    --coverage-folder tests/consenus/coverage/ \
                    --file-no 4 
```

Where:

- `--model-file`, takes the name of tar archive that will contain the likelihood model
- `--file-no`, number of files used in analysis
- `--coverage-folder` path to folder with coverage tracks

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
gitk build-universe ml --model-file tests/consenus/model.tar \
                          --output-file tests/consenus/universe/universe.bed \
                          --coverage-folder tests/consesnus/coverage
```

Where:

- `--model-file`, takes the name of tar archive that contains the likelihood model
- `--output-file`, takes the path to output file 
- `--coverage-folder` path to folder with coverage tracks

Similarly, we can do it in python:

```
from gitk.universe.ml_universe import ml_universe

ml_universe("tests/consesnus/model.tar",
     "/home/hee6jn/Documents/gitk/tests/consesnus/coverage",
     "all",
     "tests/consenus/universe/universe.bed")
```

## HMM 
Another approach to making flexible universes is using Hidden Markov Models.
We can do it for example with:

```
gitk build-universe hmm --out-file tests/consenus/universe/universe.bed \
         --coverage-folder tests/consenus/coverage/ \
         --save-max-cove
```

Where:

- `--output-file`, takes the path to output file 
- `--coverage-folder`, path to folder with coverage tracks
- `--coverage-prefix` prefix used in uniwig for making files, default is "all"
- `--not-normlaize`, is a flag that specifies whether not to normalize tracks before running HMM
- `--save-max-cove`,  is a flag that specifies whether to save maximum coverage of each output peak

Similarly, we can do it in python:

```
from gitk.universe.hmm_universe import hmm_universe

hmm_universe("tests/consenus/coverage/",
                 "tests/consenus/universe/universe.bed",
                 save_max_cove=True)
```