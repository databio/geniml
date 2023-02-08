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

## Using the tools

The functionality should be written in a way that it provides utility as a Python library. Then, in addition, some subset of the utility can be provided by CLI-access.

### From within Python

All the functions should be written to be useful via import, calling with `gitk.<module>.<function>`. For example:

```
import gitk

gitk.hmm.function()
```

### From the CLI

For the modules where it makes sense, they should also be runnable on the CLI, like this:

```
gitk <module> ...
```

## Shared code

Any variables, functions, or other code that is shared across modules should be placed in the parent module, which is held in the [gitk](gitk) folder.


