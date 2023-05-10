

path_universe='/project/shefflab/data/StarSpace/universe/tiles1000.hg19.bed'
path_output='../outputs/bedembed_output'
assembly='hg19'

path_meta_test='../tests/test_file_meta.csv'
path_data='/project/shefflab/data/encode/'
labels="cell_type,target"

python ./bedembed_test.py -data_path $path_data -meta $path_meta_test -univ $path_universe -o $path_output \
-l $labels 
