library("data.table")
library("ggplot2")

# This script produces visualizations of bedshift results (perturbed bed files) using R.

# Load in the files and process them, returning a table with all the regions.
bedshiftread = function(startfile, randfiles){
	files = c(randfiles, startfile)
	nfiles = length(files)
	regionslist = lapply (files, fread)
	rowsperfile = sapply(regionslist, NROW)
	regionstable = rbindlist(regionslist, fill=TRUE)
	regionstable[,fileid:=rep(seq_len(nfiles), rowsperfile)]
	regionstable[,file:="random"]

	starfileregions = seq(from=NROW(regionstable)+1-rowsperfile[length(rowsperfile)], to=NROW(regionstable))
	regionstable[starfileregions,file:="original"]
	return(regionstable)
}

# Plot the results of the bedshiftread function
bedshiftplot = function(regionstable) {
	ggplot(regionstable, 
		aes(xmin=V2, xmax=V3, ymin=fileid, ymax=fileid+0.75, fill=file)) + 
		geom_rect() + 
		theme_classic() + 
		scale_fill_manual(values=c("black", "gray")) + 
		xlab("Genome") + 
		ylab("Files") + 
		theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
}

# Provide the original file (the one that's being perturbed)
# and the filenames of all randomized files.

# Run the randomization with a command like this:
# bedshift --verbosity 5 -b tests/simple_1.bed -d .3 -l tests/chrom_sizes_1 -r 10

startfile = "tests/simple_1.bed"
randfiles =  paste0("rep", 1:10, "_bedshifted_simple_1.bed")

pdf("drop_H.pdf", width=6, height=2)
regionstable = bedshiftread(startfile, randfiles)
bedshiftplot(regionstable)
dev.off()
