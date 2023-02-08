# Genomic interval toolkit

## Introduction

This repository is a suite of tools related to applying machine learning approaches in understanding genomic interval data.

## Repository organization

This repo is divided into modules:

- [gitk/hmm](gitk/hmm) - Building HMMs
- [gitk/assess](gitk/assess) - Assess universe fit
- [gitk/likelihood](gitk/likelihood) - Calculate likelihood of universe

Each module corresponds to a CLI command (*e.g* `gitk hmm` or `gitk likelihood`), with the corresponding code contained within a sub-folder named after the model. Inside each sub-folder is also:

- `cli.py` - defines the command-line interface and provides a subparser for this module's CLI command.
- `README.md` - describes how to use the code
- `<module>.py`, and other `.py` files - functions that provide utility for this module.

## Install

```
pip install --user --upgrade .
```

## Run

You can run the modules then with

```
gitk <module> ...
```

## Shared code

Any variables, functions, or other code that is shared across modules should be placed in the parent module, which is held in the [gitk](gitk) folder.

## Using from within Python

All the functions should be written in a way that they can be used either through the CLI, or via an import, using `gitk.<module>.<function>`, like:

```
import gitk

gitk.hmm.function()
```
