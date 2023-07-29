#!/bin/bash

rm -r tests/consesnus/results
mkdir tests/consesnus/results

# make lh model
geniml lh build_model --model-file tests/consesnus/results/lh_model.tar \
              --file-no 4 \
              --coverage-folder tests/consesnus/coverage/

mkdir tests/consesnus/results/universe/
# make cut-off universe
geniml universe cc --coverage-folder tests/consesnus/coverage/  \
                       --output-file tests/consesnus/results/universe/cc_universe.bed

geniml universe ccf --coverage-folder tests/consesnus/coverage/  \
                       --output-file tests/consesnus/results/universe/ccf_universe.bed

geniml universe ml --model-file tests/consesnus/results/lh_model.tar \
                          --output-file tests/consesnus/results/universe/ml_universe.bed\
                          --coverage-folder tests/consesnus/coverage/

# make HMM universe
geniml universe hmm --output-file tests/consesnus/results/universe/hmm_universe.bed --coverage-folder tests/consesnus/coverage/


# assessment
 geniml assess --raw-data-folder tests/consesnus/raw/\
  --file-list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/universe/cc_universe.bed\
  --save-to-file --folder-out tests/consesnus/results/ \
  --pref cc --no-workers 1 \
  --overlap \
  --distance \
  --distance-universe-to-file

 geniml assess --raw-data-folder tests/consesnus/raw/\
  --file-list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/universe/ccf_universe.bed \
  --save-to-file --folder-out tests/consesnus/results/ \
  --pref ccf --no-workers 1 \
  --overlap \
  --distance \
  --distance-universe-to-file\
  --distance-flexible\
  --distance-flexible-universe-to-file

 geniml assess --raw-data-folder tests/consesnus/raw/\
  --file-list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/universe/ml_universe.bed \
  --save-to-file --folder-out tests/consesnus/results/ \
  --pref ml --no-workers 1 \
  --overlap \
  --distance \
  --distance-universe-to-file\
  --distance-flexible\
  --distance-flexible-universe-to-file

 geniml assess --raw-data-folder tests/consesnus/raw/\
  --file-list tests/consesnus/file_list.txt \
  --universe tests/consesnus/results/universe/hmm_universe.bed \
  --save-to-file --folder-out tests/consesnus/results/ \
  --pref hmm --no-workers 1 \
  --overlap \
  --distance \
  --distance-universe-to-file\
  --distance-flexible\
  --distance-flexible-universe-to-file

