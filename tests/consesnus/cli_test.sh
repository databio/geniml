#!/bin/bash

rm -r tests/consesnus/results
mkdir tests/consesnus/results

# make lh model
gitk lh build_model --model-file tests/consesnus/results/lh_model.tar \
              --file-no 4 \
              --coverage-folder tests/consesnus/coverage/

mkdir tests/consesnus/results/universe/
# make cut-off universe
gitk lh universe_hard --coverage-file tests/consesnus/coverage/all_core.bw  \
                       --fout tests/consesnus/results/cut_off.bed

gitk lh universe_hard --coverage_file tests/consesnus/coverage/all_core.bw  \
                      --cut_off 2 \
                       --fout tests/consesnus/results/universe/cut_off_c2.bed

gitk lh universe_hard --coverage_file tests/consesnus/coverage/all_core.bw  \
                      --cut_off 1 \
                      --merge 100 \
                      --filter_size 300 \
                       --fout tests/consesnus/results/universe/cut_off_c1_m100_f300.bed
# make ML flexible universe

gitk lh universe_flexible --model-file tests/consesnus/results/lh_model.tar \
                          --output-file tests/consesnus/results/ML_flexible.bed \
                          --cov-folder tests/consesnus/coverage/

# make HMM universe
gitk hmm --out-file tests/consesnus/results/hmm_raw.bed --cov-folder tests/consesnus/coverage/

gitk hmm --out_file tests/consesnus/results/universe/hmm_norm.bed --cov_folder tests/consesnus/coverage/ --normlaize --save_max_cove

# assessment
 gitk assess distance --raw-data-folder tests/consesnus/raw/\
  --file-list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/ML_flexible.bed \
  --save-to-file --folder-out tests/consesnus/results/distance/ \
  --pref test --no-workers 1 --save-each

 gitk assess distance --raw_data_folder tests/consesnus/raw/\
  --file_list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/universe/ML_flexible.bed \
  --save_to_file --folder_out tests/consesnus/results/distance/ \
  --pref test_flex --npool 1 --save_each --flexible

gitk assess intersection --raw-data-folder tests/consesnus/raw/\
  --file-list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/ML_flexible.bed \
  --save-to-file --folder-out tests/consesnus/results/intersection/ \
  --pref test --no-workers 1
