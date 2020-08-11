# Generate a random BED file

To generate a random BED file, you need to start with is a BED file with one region, which you will delete later. (It will appear at the top of the BED files, so it will be easier to delete later.) On the command line:

```
echo "chr1\t1\t1000" > random.bed
```

The next step is to construct a bedshift command to add regions to this file. We will need to specify the rate of add, which in this case is going to be the number of the new regions we want. Let's try generating 1,000 new regions. Don't forget to specify a chromosome sizes file, which is required for adding regions.

```
bedshift -b random.bed -l hg38.chrom.sizes -a 1000
```

The printed output should say that 1000 regions changed. Finally, go into the output at bedshifted_random.bed and delete the original region. If you specified repeats and have many output files that need to have the original region deleted, here is a handy command to delete the first line of every BED file. (Warning: make sure there are no other BED files in the folder before using this command.)

```
find . -name "*.bed" -exec sed -i '.bak' '1d' {} \;
```

This `find` command will find all BED files and execute a `sed` command to remove the first line. The `sed` command will operate in place and create `.bak` backup files, which can be removed later.

