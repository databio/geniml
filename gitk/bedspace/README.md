# BEDSpace

Uses the StarSpace method to embed the bed files and the meta data.

The resulted embeddings are used in the following scenarios:

* Scenario 1: Retrieve region sets for a metadata query
* Scenario 2: Annotate unlabeled region sets
* Scenario 3: Retrieve region sets for a query region set



## Installation instructions

1. Clone this repository

## How to use

### 1. Create an output directory

Create a directory to store all the outputs files 

```
export BEDEMBED=`pwd`
mkdir -p outputs/bedspace_output/figures/S1
mkdir -p outputs/bedspace_output/figures/S2
mkdir -p outputs/bedspace_output/figures/S3
module load anaconda gcc/9.2.0 boost
```

### 3. Run the bedspace_train pipeline to train the StarSpace model

This model will be use in all the scenarios. 

```
cd pipeline
python ./bedspace_train.py 

```

#### 3.1 To generate the UMAP-plot of the label embeddings use:
```
cd pipeline
Visualization.py
```


### 4. Run the bedspace_test pipeline for Scenario-1 and Scenario-2

This code calculate the similarity between the test file embedding and label embedding 
```
cd pipeline
python ./bedspace_test.py

```

#### 4.1 To generate and save the plots for Scenario-1 use:
```
cd pipeline
Visualization.py
```

#### 4.2 To generate and save the plots for Scenario-2 use:
```
cd pipeline
Visualization.py
```


### 5. Run the bedspace_test for pipeline Scenario-3

This code calculate the similarity between the embedding of the query(test) files and database(train) files

```
cd pipeline
python ./bedspace_queryDBsim.py

```
#### 5.1 To generate and save the plots for Scenario-3 use:
```
cd pipeline
Visualization.py
```







