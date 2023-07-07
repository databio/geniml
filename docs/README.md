# Genomic interval toolkit

## Introduction

`gitk` is a suite of tools for applying machine learning approaches to genomic interval data. It is organized as a set of modules that provide related functions, such as building HMMs, assessing genomic interval universes, calculating likelihoods of consensus genomic interval sets, and computing single-cell clusters.

## Install

```
pip install --user --upgrade .
```

## gitk modules

- [gitk/hmm](gitk/hmm) - Building HMMs
- [gitk/assess](gitk/assess) - Assess universe fit
- [gitk/likelihood](gitk/likelihood) - Calculate likelihood of universe
- [gitk/scembed](gitk/scembed) - Compute single-cell clusters from a cell-feature matrix using Word2Vec

## Using modules from Python

This repo is divided into modules. Each module should be written in a way that it provides utility as a Python library. For example, you can call functions in the `hmm` module like this:

```
import gitk

gitk.hmm.function()
```

## Command-line interfaces

In addition to being importable from Python, *some* modules also provide a CLI. For these, developers provide a subcommand for CLI use. The root `gitk` package provides a generalized command-line interface with the command `gitk`. The modules that provide CLIs then correspond to CLI commands, *e.g* `gitk hmm` or `gitk likelihood`, with the corresponding code contained within a sub-folder named after the model:

```
gitk <module> ...
```

This is implemented within each module folder with:

- `gitk/<module>/cli.py` - defines the command-line interface and provides a subparser for this module's CLI command.

