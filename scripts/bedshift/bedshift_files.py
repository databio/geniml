import bedshift
import os


datafolder = "."

files = os.listdir()
for file in files:
    if file.endswith(".bed"):
        # you may also pass in a chrom.sizes file as the
        # second argument if you are adding or shifting regions
        b = bedshift.Bedshift(file)
        b.all_perturbations(cutrate=0.3, droprate=0.2)
        b.to_bed("bedshifted_" + file)
