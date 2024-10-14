# INPUT PREPROCESSING
input='/project/shefflab/brickyard/results_pipeline/gomez_atac/results_pipeline/differential/gitk_univ/pseudobulk_data/cell_barcodes/'
train_meta='/home/bx2ur/code/reninness_score/tests/renin_singlecell_data_with_label.csv'
universe='/home/bx2ur/code/reninness_score/data/universe/universe.bed'
output='/home/bx2ur/code/reninness_score/test_outputs/gitk_univ_d10/'
label='label'

geniml bedspace preprocess -i $input -m $train_meta -u $universe -o $output -l $label

# MODEL TRAINING
path_starspace='/home/bx2ur/code/reninness_score/tools/Starspace/'
preprocessed_input='/home/bx2ur/code/reninness_score/test_outputs/gitk_univ_d10/train_input.txt'
epochs=50
dim=10 
learning_rate=0.0001
output='/home/bx2ur/code/reninness_score/test_outputs/gitk_univ_d10/'

geniml bedspace train -s $path_starspace -i $preprocessed_input -n $epochs -d $dim -l $learning_rate -o $output

# DISTANCE CALCULATION 
model='/home/bx2ur/code/reninness_score/test_outputs/gitk_univ_d10/starspace_trained_model'
path_starspace='/home/bx2ur/code/reninness_score/tools/Starspace/starspace'
train_meta='/home/bx2ur/code/reninness_score/tests/renin_singlecell_data_with_label.csv'
test_meta='/home/bx2ur/code/reninness_score/tests/ctrl_6mo.csv'
universe='/home/bx2ur/code/reninness_score/data/universe/universe.bed'
project_name='test'
test_file_path='/project/shefflab/brickyard/results_pipeline/gomez_atac/results_pipeline/differential/gitk_univ/pseudobulk_data/cell_barcodes/captopril_trt/ctrl_6mo/' 
label='label'
output='/home/bx2ur/code/reninness_score/test_outputs/gitk_univ_d10/'
threshold=0.8

geniml bedspace distances -i $model -s $path_starspace --metadata-train $train_meta --metadata-test $test_meta -u $universe -p $project_name -f $test_file_path -l $label -o $output -t $threshold

