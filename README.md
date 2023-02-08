# Genomic interval toolkit

## Introduction

This repository is a suite of tools related to applying machine learning approaches in understanding genomic interval data.

## Install

```
pip install --user --upgrade .
```

## Repository organization

This repo is divided into sub-modules, each with a sub-folder:

- [gitk/hmm](gitk/hmm) - Building HMMs
- [gitk/assess](gitk/assess) - Assess universe fit
- [gitk/likelihood](gitk/likelihood) - Calculate likelihood of universe

Each subfolder has:

- a `cli.py` file that defines the command-line interface and provides an argparser.
- a `README.md` file that describes how to use the code
