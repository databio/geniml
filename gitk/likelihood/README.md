# hmm module

## Introduction


## How to use



```
gitk lh build_model --model_folder tests/consesnus/lh_model \
              --file_list tests/consesnus/file_list.txt \
              --coverage_folder tests/consesnus/coverage/
```

```
 gitk lh universe_hard --coverage_file tests/consesnus/coverage/all_core.bw  \
                       --fout tests/consesnus/universe/ML_hard.bed

```

```
gitk lh universe_flexible --model_folder test/data/lh_model \
                          --output_file test/results/universe/ML_flexible.bed

```