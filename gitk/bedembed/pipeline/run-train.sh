
path_starspace='../tools/StarSpace/starspace'
path_meta='../tests/test_file_meta.csv'
path_universe='/project/shefflab/data/StarSpace/universe/tiles1000.hg19.bed'
path_output='../outputs/bedembed_output'
assembly='hg19'
path_data='/project/shefflab/data/encode/'
labels="cell_type,target"

no_files=10
start_line=0
dim=50
epochs=20
learning_rate=0.001

python ./bedembed_train.py -star $path_starspace -i $path_data -g $assembly -meta $path_meta -univ $path_universe \
-l $labels -nof $no_files -o $path_output -startline $start_line -dim $dim -epochs $epochs -lr $learning_rate

