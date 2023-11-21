# BED File Retrieval and Processing

This client downloads, processes, and caches BED files and BED sets from the BEDbase API and converts them into a GenomicRanges or GenomicRangesList object. It provides various commands to interact with BED files, including downloading individual files, downloading BEDsets, processing local BED files, and processing BED file identifiers.

## Usage

### Create an Instance of the BBClient Class:

```python
from geniml.bbclient import BBClient

bbclient = BBClient(cache_folder="cache", bedbase_api="https://api.bedbase.org
```

### Download and cache a BED file from BEDbase

```python
bedfile_id = "...."  # find interesting bedfile on bedbase
bedfile = bbclient.load_bed(bedfile_id)  # download, cache and return a RegionSet object
gr = bedfile.to_granges()  # return a GenomicRanges object
```

### Cache a local BED file
```python
from geniml.io import RegionSet

bedfile = RegionSet("path/to/bedfile")
gr = bedfile.to_granges()  # should return a GenomicRanges object
bedfile_id = bbclient.add_bed_to_cache(bedfile) # compute its ID and add it to the cache
```

### Download cache a BEDset from BEDbase

```python
bedset_identifier = "xyz" # find some interesting bedset on bedbase.org
bedset = bbclient.load_bedset(bedset_identifier)  # download, cache and return a BedSet object
grl = bedset.to_granges_list()  # return a GenomicRangesList object
```

### Cache a local BEDset

```python
from geniml.io import BedSet

bedset_folder = "path/to/bed/files/folder"
bedset = BedSet(
    [os.path.join(bedset-folder, file_name) for file_name in os.listdir(bedset_folder)]
)
bedset_id = bbclient.add_bedset_to_cache(bedset)
```

## For command line usage, run the CLI with appropriate subcommands and arguments as described below:

### Cache BED file

```bash
geniml bbclient cache-bed --input-identifier <path_to_BED_file_or_identifier_or_url>
```

Replace  <path_to_BED_file_or_identifier_or_url> with the path to the local BED file or BED file's identifier or url.

### Cache BED set

```bash
geniml bbclient cache-bedset --input-identifier <path_to_BED_files_folder_or_identifier>
```

Replace  <path_to_BED_files_folder_or_identifier> with path to the folder of BED files or the BEDset identifier

### Seek the path of a BED file or BEDset in cache folder

```bash
geniml bbclient seek --input-identifier <identifier>
```

Replace <bedset_identifier> with the identifier of the BED file or BEDset you want to seek.

### List and count the subdirectories and files in cache  folder

```bash
geniml bbclient tree
```

`tree` command may need [installing](https://www.geeksforgeeks.org/tree-command-unixlinux/)

### Command-Line Arguments

    cache-bed: cache a BED file from BEDbase (with given identifier), url, or local file 
    cache-bedset: cache a BEDset from BEDbase (with given identifier) or local folder 
    seek: return the path of a BED file or BEDset in cache folder
    tree: 

### Cache Folder

By default, the downloaded and processed BED files are cached in the bed_cache folder. You can specify a different cache folder using the --cache-folder argument.
The cache folder has this structure:
```
cache_folder
  bedfiles
    a/b/ab1234xyz.bed.gz
    ..
  bedsets
    c/d/cd123hij.txt
```


### Dependencies
    requests: For making HTTP requests.
    pandas: For data manipulation.
    genomicranges: For processing BED files and creating GenomicRanges objects.
