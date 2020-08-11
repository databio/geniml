# Using bedshift to create an evaluation dataset for similarity scores

## Generate different files

Bedshift perturbations include add, drop, shift, cut, and merge. Using any of these perturbations, or combinations of them, you can generate a set of files that are slightly perturbed from the original file. Assuming that the original file is called `original.bed`, and you want 100 files of added regions and 100 files of dropped regions:

```
bedshift -b original.bed -l hg38.chrom.sizes -a 0.1 -r 100
bedshift -b original.bed -d 0.3 -r 100
```

Don't forget the add and shift operations require a chrom.sizes file. The output file will be in `bedshifted_original.bed`.


## Evaluating a similarity score

This is when the bedshifted file will be put to use. The 100 repetitions of add and drop will be compared against the original file using the similarity score of your choice. The output of the similarity score should reflect the degree of change specified to bedshift. In very general terms, the pseudocode should be like this:

```
for each bedshift_file in folder:
	score = SimilarityScore(bedshift_file, original_file, ...)
	add score to score_list
avg_similarity_score = mean(score_list)
```

You can repeat this for each of the similarity scores and each of the perturbation combinations, and then compare the results. This way, you can get an accurate understanding of whether your similarity score reflects added regions, dropped regions, and more.


## Using a PEP to quickly submit multiple bedshift jobs

Using a [Portable Encapsulated Project](http://pep.databio.org/en/latest/) (PEP), creating multiple combinations of bedshift files becomes faster and more organized. The PEP consists of a sample table containing the perturbation parameters and a config file. Here is what the `sample_table.csv` may look like. Each row specifies the arguments for a bedshift command.

| sample_name | add | drop | shift | cut | merge |
|-------------|-----|------|------|------|-------|
| add1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| add2 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| add3 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| drop-shift1 | 0.0 | 0.1 | 0.2 | 0.0 | 0.0 |
| drop-shift2 | 0.0 | 0.2 | 0.2 | 0.0 | 0.0 |
| drop-cut | 0.0 | 0.3 | 0.0 | 0.4 | 0.0 |
| shift-merge | 0.0 | 0.0 | 0.4 | 0.0 | 0.4 |

And here is what the `project_config.yaml` file looks like:

```
pep_version: 2.0.0
sample_table: "sample_table.csv"
sample_modifiers:
  append:
    file: "original.bed"
    repeat: 100
```

Now the project is described neatly in two files. The `sample_modifiers` in the config file just adds extra columns to the sample table in post-processing and makes the project more configurable, instead of having to repeat the same parameter in the `sample_table.csv`. In this example, the `sample_modifiers` append two columns with the file which bedshift is to be performed on, and the number of repetitions that bedshift should create.

The PEP describes the project, but the tool that submits the project jobs is called [looper](http://looper.databio.org/en/latest/). In one line of code, it will interpret the PEP and form commands to be submitted to your processor or computing cluster. To use looper, you will need to add a few lines to your `project_config.yaml`:

```
pep_version: 2.0.0
sample_table: "sample_table.csv"
looper:
  output_dir: "looper_output/"
sample_modifiers:
  append:
    pipeline_interfaces: "pipeline_interface.yaml"
    file: "original.bed"
    repeat: 100
```

You will also need to create a `pipeline_interface.yaml` that describes how to form commands:

```
pipeline_name: bedshift_run
pipeline_type: sample
command_template: >
    bedshift -b {sample.file} -l hg38.chrom.sizes -a {sample.add} -d {sample.drop} -s {sample.shift} -c {sample.cut} -m {sample.merge} -r {sample.repeat} -o {sample.sample_name}.bed
compute:
  mem: 4000
  cores: 1
  time: "00:10:00"
```

After all of this, the command to run looper and submit the jobs is:

```
looper run project_config.yaml
```

Soon, you should see bedshift files appear in the `looper_output` folder. The BED file names will correspond to the sample names from `sample_table.csv`.

