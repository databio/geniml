# Shift, Add, and Drop From File

"From file" means that the regions selected to shift, add, or drop are specified from a provided file. These features provide the ability to finely control what regions are perturbed. For example, if you have a BED file specifying exon regions and you want to add only exons, you can use `--addfile`.

## Add from file example

```
bedshift -b mydata.bed -a 0.07 --addfile exons.bed
```

Specifying `--addfile` with `-a` add rate will increase the size of `mydata.bed` by 7% with new regions selected from `exons.bed`.

## Shift from file example

Shift from file first calculates which regions overlap between the specified `--shiftfile` and `--bedfile`, then selects which regions to shift among those overlaps.

```
bedshift -b mydata.bed -s 0.42 --shiftmean 5 --shiftstdev 5 --shiftfile snp.bed
```

In this example, we only want to shift regions that are SNPs. The number of shifted regions is 42% of the total regions in `mydata.bed`. Notice here that unlike `--addfile`, we still have to specify the shift mean and standard deviation. This is because `--shiftfile` tells which regions to shift, but not by how much.

## Drop from file example

Drop from file, like shift from file, calculates overlaps between the specified `--dropfile` and `--bedfile`, then selects regions from those overlaps to drop.

```
bedshift -b mydata.bed -d 0.4 -dropfile snp.bed
```

This command will drop regions that overlap with SNPs. The number of dropped regions is 40% of the total regions in `mydata.bed`.
