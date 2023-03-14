# Genomic interval toolkit

## Introduction

This repository is a suite of tools related to applying machine learning approaches in understanding genomic interval data.

## Install

```
pip install --user --upgrade .
```

## Repository organization

This repo is divided into modules. Each module is in a subfolder:

- [gitk/hmm](gitk/hmm) - Building HMMs
- [gitk/assess](gitk/assess) - Assess universe fit
- [gitk/likelihood](gitk/likelihood) - Calculate likelihood of universe
- [gitk/scembed](gitk/scembed) - Compute single-cell clusters from a cell-feature matrix using Word2Vec

### Using modules from Python

Each module should be written in a way that it provides utility as a Python library. It should contain at least these files:

- `README.md` - describes how to use the code
- `<module>.py`, and other `.py` files - functions that provide utility for this module.

*All* the functions should be written to be useful via import, calling with `gitk.<module>.<function>`. For example:

```
import gitk

gitk.hmm.function()
```

### Command-line interfaces

In addition to being importable from Python, *some* modules also provide a CLI. For these, developers provide a subcommand for CLI use. The root `gitk` package provides a generalized command-line interface with the command `gitk`. The modules that provide CLIs then correspond to CLI commands, *e.g* `gitk hmm` or `gitk likelihood`, with the corresponding code contained within a sub-folder named after the model:

```
gitk <module> ...
```

This is implemented within each module folder with:

- `gitk/<module>/cli.py` - defines the command-line interface and provides a subparser for this module's CLI command.


## Adding a new module

1. Put your module in a subfolder
2. Make sure to include a `__init__.py` so it's importable.
3. Add it to list of packages in `setup.py` 

## Shared code

Any variables, functions, or other code that is shared across modules should be placed in the parent module, which is held in the [gitk](gitk) folder.


