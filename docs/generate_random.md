# Generate a random BED file

To generate a random BED file, you need to start with is a BED file with one region. On the command line:

```
touch random.bed
```

Add in any region, which you will delete later. It is easier to add a region on chromosome 1 so it will be at the top of the BED file after perturbations.

```
chr1	1	100
```

The next step is to construct a bedshift command to add regions to this file. We will need to specify the rate of add, which is going to be the number of the new regions we want in this case. Let's try generating 1,000 new regions. Don't forget to have a chromosome sizes file, which is required for adding regions.

```
bedshift -b random.bed -l hg38.chrom.sizes -a 1000
```

The printed output should say that 1000 regions changed. Finally, go into the output at bedshifted_random.bed and delete the original region. If you specified repeats and have many output files that need to have the original region deleted, here is a handy command to delete the first line of every BED file.

```
find . -name "*.bed" -exec sed -i '.bak' '1d' {} \;
```

This `find` command will find all BED files and execute a `sed` command to remove the first line. The `sed` command will operate in place and create `.bak` backup files, which can be removed later.

