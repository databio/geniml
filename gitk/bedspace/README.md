# BEDSpace
## Overview
`bedspace` uses the StarSpace method (Wu et al., 2018) to jointly embed genomic interval regions sets with associated metadata into a shared latent embedding space. This facilitates fast search and retrieval of similar region sets and their associated metadata.

## Installation
`bedspace` is a part of the `gitk` package, which can be installed from PyPI using `pip install gitk`. To ensure that everything is working correctly, run the following command:

```
python -c "from gitk import bedspace"
```

## Usage
There are four main commands in `bedspace`:
1. `bedspace preprocess`: preprocesses a set of genomic interval regions and their associated metadata into a format that can be used by `bedspace train`.
2. `bedspace train`: trains a StarSpace model on the preprocessed data.
3. `bedspace distances`: computes the distances between all of the trained model's region sets and meta data labels.
4. `bedspace search` searches for the most similar region sets and metadata labels to a given query. There are three scenarios in this command, which will be described in turn.

### `bedspace preprocess`
The `preprocess` command will prepare a set of region sets and metadata labels for training. This includes things like adding the `__label__` prefix to metadata labels, and converting the region sets into a format that can be used by StarSpace. The command takes in a set of region sets and metadata labels, and outputs a set of preprocessed region sets and metadata labels. The command can be run as follows:

```
gitk bedspace preprocess \
    --input <path to input region sets> \
    --metadata <path to input metadata labels> \
    --universe <path to universe file> \
    --labels <path to the labels file> \
    --output <path to output preprocessed region sets> 
```
### `bedspace train`
The `train` command will train a StarSpace model on the preprocessed region sets and metadata labels. It requires that you have ran the `preprocess` command first. The `train` command takes in a set of preprocessed region sets and metadata labels, and outputs a trained StarSpace model. The command can be run as follows:

```
gitk bedspace train \
    --path-to-starspace <path to StarSpace executable> \
    --input <path to preprocessed region sets> \
    --output <path to output trained model> \
    --dim <dimension of embedding space> \
    --epochs <number of epochs to train for> \
    --lr <learning rate>
```

### `bedspace distances`
The `distances` command will compute the distances between all of the region sets and metadata labels in the trained model. It requires that you have ran the `train` command first. The `distances` command takes in a trained StarSpace model, and outputs a set of distances between all of the region sets and metadata labels in the model. The command can be run as follows:

```
gitk bedspace distances \
    --input <path to trained model> \
    --metadata <path to input metadata labels> \
    --universe <path to universe file> \
    --labels <path to labels file> \
    --files <path to region sets> \
    --output <path to output distances>
```

### `bedspace search`
There are three scenarios when using the `search` command. 1) You have a query region set and want to find the most similar metadata labels, 2) You have a query metadata label and want to find the most similar region sets, and 3) You have a query region set and want to find the most similar region sets. These are labeled `r2l`, ``l2r``, and `r2r` respectively. The `search` command requires that you have ran the `distances` command first. The `search` command requires you to specify the search type so it knows which scenario you are using. It also requires a query. Example usages for each type are given below:

#### `r2l`
```
gitk bedspace search \
    -t lr2
    -d <path to distances> \
    -n <number of results to return> \
    path/to/regions.bed
```

#### `l2r`
```
gitk bedspace search \
    -t rl2
    -d <path to distances> \
    -n <number of results to return> \
    K562
```

#### `r2r`
```
gitk bedspace search \
    -t rr2
    -d <path to distances> \
    -n <number of results to return> \
    path/to/regions.bed
```