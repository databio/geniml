# How to create consensus peaks from a set of BED files

## Data preprocessing

1. install [uniwig](https://github.com/databio/uniwig/tree/smoothing), make sure to use branch smoothing
2. use [create_unsorted.sh](https://github.com/databio/uniwig/blob/smoothing/create_unsorted.sh) to make three bigWig from your files

## Cut-off universe
Make cut-off universe from coverage using:
```
 gitk lh universe_hard --coverage_file coverage.bw  \
                       --fout universe.bed

```  
Where:
- ```--coverage_file```, takes the path to bigWig file with cverage track
- ```--fout```, takes the path to output file 

## Maximum likelihood universe
Make likelihood model from coverage tracks using:

```
gitk lh build_model --model_folder model.tar \
              --file_no x \
              --coverage_folder coverage/
```
Where:
- ```--model_folder```, takes the name of tar archive that will contain the likelihood model
- ```--file_no```, number of files used in analysis
- ```--coverage_folder``` path to folder with coverage tracks
- ```--coverage_prefix``` prefix used in uniwig for making files 

Use likelihood model to make a maximum likelihood universe
```
gitk lh universe_flexible --model_folder model.tar \
                          --output_file universe.bed
```
Where:
-  ```--model_folder```, takes the name of tar archive that contains the likelihood model
- ```--output_file```, takes the path to output file 