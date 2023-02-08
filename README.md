# Genomic interval toolkit

## Introduction

This repository is a suite of tools related to applying machine learning approaches in understanding genomic interval data.

## Repository organization

This repo is divided into modules:

- [gitk/hmm](gitk/hmm) - Building HMMs
- [gitk/assess](gitk/assess) - Assess universe fit
- [gitk/likelihood](gitk/likelihood) - Calculate likelihood of universe

Each module corresponds to a CLI command (*e.g* `gitk hmm` or `gitk likelihood`), with the corresponding code contained within a sub-folder named after the model. Inside each sub-folder is also:

- a `cli.py` file that defines the command-line interface and provides an argparser.
- a `README.md` file that describes how to use the code


## Install

```
pip install --user --upgrade .
```

## Run

You can run the modules then with

```
gitk <module> ...
```

