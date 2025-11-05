# hmm module

## Introduction


## How to use



```
geniml lh --model-file tests/consesnus/lh_model.tar \
          --coverage-folder tests/consesnus/coverage/ \
          --file-no 4
```

Note: The `lh` command builds a likelihood model. To build universes using the model, use:

```
geniml build-universe ml --model-file tests/consesnus/lh_model.tar \
                         --coverage-folder tests/consesnus/coverage/ \
                         --output-file tests/consesnus/universe/ML.bed
```