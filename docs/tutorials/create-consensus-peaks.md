# How to build a new universe?

## Data preprocessing
In this tutorial, you will use CLI of geniml package to build different types of universes from example files, which can be downloaded from XXX. In there you will find a compressed folder:

```
consensus:
    - raw
        test_1.bed
        test_2.bed
        test_3.bed
        test_4.bed
    file_list.txt
    chrom.sizes
```

In the raw folder there are example BED files used in this tutorial and in file_list.txt are names of files you will analyze. Additionally there is a file with chromosome sizes, which you will use to preprocess the data. 

To build any kind of a universe you need bigWig files with genome coverage by the analyzed collection, which can be made it using [uniwig](https://github.com/databio/uniwig/). First we have to combine all the analyzed files into one BED file:

```
cat raw/* > raw/combined_files.bed
```

This combined file can next be used to prepare the genome coverage tracks, with window size for smoothing of breakpoints set to 25:

```
$UNIWIG_PATH/bin/uniwig -m 25 raw/combined_files.bed chrom.sizes coverage/all
```

This will create three files: `coverage/all_start.bw`, `coverage/all_core.bw`, `coverage/all_end.bw`, with coverage of the genome by regions' starts, regions and regions' ends respectively. Those files can be loaded into Genomic Viewer for visualization.  

## Coverage cutoff universe

First, you will create a coverage cutoff universe (CC). This is the simplest type of a universe that only includes genomic positions with coverage greater or equal to cutoff *x*. This cutoff by default is calculated using simple likelihood model that calculates the probability of appearing in a collection. The universe can be build just based on genome coverage:

```
geniml build-universe cc --coverage-folder coverage/ \
                          --output-file universe_cc.bed

```

Where:

- `--coverage-folder`, takes the path to bigWig folder with genome coverage by collection
- `--output-file`, takes the path to output file

Or we can import it directly into Python:

```
from geniml.universe.cc_universe import cc_universe

cc_universe("tests/consenus/coverage/",
        file_out="tests/consenus/universe/universe.bed")
```

Depending on the task we can also smooth the output universe by setting `--merge`
flag with the distance beyond which peaks should be merged together and
`--filter-size` with minimum size of peak that should be part of the universe. We can also not use the maximum likelihood cut-off and instead of it use user defined cutoff. For that we have to set `--cutoff`. If we set it to 1 we get union universe, and when to number of files we will get intersection universe.

## Coverage cutoff flexible universe
A more complex version of coverage cutoff universe is coverage cutoff flexible universe (CCF). In contrast to its' fixed version it produces flexible universe. It builds confidence interval around the maximum likelihood cutoff. This results in two values one for the cutoff for boundaries, and the other one for the region core. Despite the fact that the CFF universe is more complex it is build using the same input as the CC universe: 

```
geniml build-universe ccf --coverage-folder coverage/ \
                           --output-file universe_ccf.bed

```

Where:

- `--coverage-folder`, takes the path to bigWig folder with genome coverage by collection
- `--output-file`, takes the path to output file

Or we can import it directly into Python:
```
from geniml.universe.ccf_universe import ccf_universe

ccf_universe("tests/consenus/coverage/",
        file_out="tests/consenus/universe/universe.bed")
```


## Maximum likelihood universe
In the previous examples both CC anf CCF universes used simple likelihood model to calculate the cutoff. However, we also developed more complex likelihood model that takes into account the positions of starts and ends of the regions in the collection. This LH model can build based on coverage files and number of analyzed files:

```
geniml lh build_model --model-file model.tar \
                      --coverage-folder coverage/ \
                      --file-no `wc -l file_list.txt`
```

The resulting tar archiver contains LH model. This model can be used as a scoring function that assigns to each position probability of it being a start, core or end of a region. It can be both used for universe assessment and universe building. Combination of LH model and optimization algorithm for building flexible universes results in maximum likelihood universe (ML):

```
geniml build-universe ml --model-file model.tar \
                         --coverage-folder coverage/ \
                         --output-file universe_ml.bed 
```

## HMM 
The forth presented method of creating universes utilizes Hidden Markov Models (HMM). In this approach the parts of flexible regions are hidden states of the model, while genome coverage by the collections are emissions. The resulting universe is called Hidden Markov Model universe. It can be build only based on the genome coverage by the collection:

```
geniml build-universe hmm --coverage-folder coverage/ \
                          --output-file universe_hmm.bed

```

# How to assess new universe?

So far you used many different methods for creating new universes. But choosing, which universe represents data the best can be challenging. To help with this we created a tutorial that can be found [here](../notebooks/assess-universe.ipynb), which presents different  methods that assess universe fit to the collection of files.