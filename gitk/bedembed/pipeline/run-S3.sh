
path_starspace='../tools/Starspace/starspace'
path_meta='../tests/test_file_meta.csv'
path_universe='/project/shefflab/data/StarSpace/universe/tiles1000.hg19.bed'
path_output='../outputs/bedembed_output'
assembly='hg19'
path_data='/project/shefflab/data/encode/'
path_meta_test='../tests/test_file_meta.csv'
labels="cell_type,target"

python ./bedembed_queryDBsim.py -data_path $path_data -db_path $path_meta -query_path $path_meta_test -univ $path_universe -o $path_output \
-l $labels 
