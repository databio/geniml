packages <- c("optparse", "GenomicDistributions", "foreach", "doParallel")
install.packages(setdiff(packages, rownames(installed.packages())))

suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("GenomicDistributions"))
suppressPackageStartupMessages(library(foreach))
suppressPackageStartupMessages(library(doParallel))

get_cluster_median <- function(fpath){
    query_data = rtracklayer::import(fpath)
    queryList = GRangesList(cluster=query_data)
    TSSdist = calcFeatureDistRefTSS(queryList, "hg19")
    abs_dist = lapply(TSSdist,abs)[['cluster']]
    return (c(median(abs_dist), length(query_data)))
}
lappend <- function (lst, ...){
lst <- c(lst, list(...))
  return(lst)
}
get_cluster_tss <- function(cluster_folder, assembly="hg19"){
    cluster_files = list.files(cluster_folder, pattern="cluster.*bed$")
    num_clusters = length(cluster_files)
    tss_arr = list()
    
    for (i in 1:num_clusters){
        queryFile = file.path(cluster_folder,cluster_files[i])
        query_data = rtracklayer::import(queryFile)
        queryList = GRangesList(cluster=query_data)
        TSSdist = calcFeatureDistRefTSS(queryList, assembly)
        abs_dist = lapply(TSSdist,abs)[['cluster']]
        tss_arr <- lappend(tss_arr,abs_dist)
    }
    return (tss_arr)
} 
random_median <- function(array, sample_size){
    sample_arr = sample(array,sample_size,FALSE)
    return (median(sample_arr))
}

perm_test <- function(perm_num, array, size, pos){
    median_arr = replicate(perm_num,random_median(array,size))
    pval = sum(median_arr < pos)/perm_num
    return (pval)
}

find_peak <- function(tss_arr, ldist, rdist){
    a = density(unlist(tss_arr))
    x = unlist(a[1])
    y = unlist(a[2])
    idx = which.max(y)
    peak_pos = x[idx]
    if (peak_pos > ldist & peak_pos < rdist){
        return (TRUE)
    } else {
        return (FALSE)
    }
}


get_significance_vals <- function(path, assembly, num_replicates=10000){
    # calculate the tss of all regions in all the clusters
    cluster_tss = get_cluster_tss(path, assembly)
    num_clusters <- length(cluster_tss)
    # get cluster size
    csize_arr = rep(0,times=num_clusters)
    for (i in 1:num_clusters){
        s = length(unlist(cluster_tss[i]))
        csize_arr[i] <- s
    }
    # merge tss from all clusters
    all_tss = c()
    for (i in 1:num_clusters){
        c_tss = unlist(cluster_tss[i])
        all_tss = c(all_tss,c_tss)
    }
    pval_arr = rep(0, num_clusters)
    for (i in 1:num_clusters){
        median_tss = median(unlist(cluster_tss[i]))
        pval = perm_test(num_replicates, all_tss, csize_arr[i], median_tss)
        pval_arr[i] = pval
    }
    return (pval_arr)
}

option_list = list(
  make_option("--path", type="character", default="/bigtemp/gz5hp/genomes/tfbs_experiments/tbfs_clustering_results/KMeans_20K_seed0", 
              help="dataset file name", metavar="character"),
    make_option("--assembly", type="character", default="hg19", 
              help="hg19 or hg38", metavar="character"),
    make_option("--num-workers", type="integer",default=10,
              help="number of parallel processes", metavar="number of processes"),
    make_option("--num-samples", type="integer",default=1000,
              help="number of samples", metavar="number of samples")
); 
 
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

registerDoParallel(opt$num_workers)

pattern = file.path(opt$path, 'Kmeans_*')
paths = Sys.glob(pattern)
num_path <- length(paths)
out <- foreach (i=1:num_path) %dopar% {
    folder <- paths[i]
    save_path <- file.path(folder, 'pvals.txt')
    # if (!file.exists(save_path)){
    pvals <- get_significance_vals(folder, opt$assembly, opt$num_samples)
    cat(format(pvals, nsmall=6),sep='\n',file=save_path)
    # }
}
