# The universe module

## Introduction

This module provides multiple ways to build a genomic region universe 



## Method 1: HMM

This will use an HMM to create a flexible segment universe, given an input of several bed files.

## How to use

Where can you find a very small example dataset?

```
gitk hmm --out_file tests/consesnus/universe/hmm_norm.bed --cov_folder tests/consesnus/coverage/
```


```
gitk hmm --out_file tests/consesnus/universe/hmm_norm.bed --cov_folder tests/consesnus/coverage/ --normalize
```