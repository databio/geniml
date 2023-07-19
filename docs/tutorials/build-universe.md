# How to build a new universe

Where can you find a very small example dataset?

```console
gitk hmm \
  --out_file tests/consensus/universe/hmm_norm.bed \
  --cov_folder tests/consensus/coverage/ 
```

```console
gitk hmm \
  --out_file tests/consensus/universe/hmm_norm.bed \
  --cov_folder tests/consensus/coverage/ \
  --normalize
```