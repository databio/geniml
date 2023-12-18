# BED file caching and loading from BEDbase

The BEDbase client command `bbclient` downloads, processes, and caches BED files and BED sets from the BEDbase API and converts them into GenomicRanges or GenomicRangesList objects.
It provides various commands to interact with BED files, including downloading individual files, downloading BEDsets, processing local BED files, and processing BED file identifiers.

This document provides tutorials for using `bbclient` via either:

1. the [Python interface](#getting-started-python-interface), or
2. the [command-line interface](#command-line-interface).

## Getting started: Python interface

### Create an instance of the BBClient Class:

```python
from geniml.bbclient import BBClient

bbclient = BBClient(cache_folder="cache", bedbase_api="https://api.bedbase.org
```

### Download and cache a remote BED file from BEDbase 

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

### Load a BED file from cache into Python session

```python
bedfile_id = "...."  # get the identifier
bedfile = bbclient.load_bed(bedfile_id)  # the same function can also load BED files that have already been cached
```


### Download and cache a BEDset from BEDbase

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

## Command line interface

### Cache BED file

```bash
geniml bbclient cache-bed <BED_file_or_identifier_or_url>
```

The `<BED_file_or_identifier_or_url>` variable can be one of 3 things:

1. a path to a local BED file;
2. a BED record identifier from BEDbase; or,
3. a URL to a BED file hosted anywhere.

### Cache BEDset

```bash
geniml bbclient cache-bedset <BED_files_folder_or_identifier>
```

The `<BED_files_folder_or_identifier>` variable may be:

1. local path to a folder containing BED files; or,
2. a BEDbase BEDset identifier

### Seek the path of a BED file or BEDset in cache folder

To retrieve the local file path to a BED file stored locally,

```bash
geniml bbclient seek <identifier>
```

Replace <identifier> with the identifier of the BED file or BEDset you want to seek.

### Count the subdirectories and files in `bedfiles` & `bedsets` folder

```bash
geniml bbclient inspect
```

`inspect` command may need installing [`tree`](https://www.geeksforgeeks.org/tree-command-unixlinux/)

### Remove a BED file or BEDset from the cache folder 

```bash
geniml bbclient rm <identifier>
```

Replace <identifier> with the identifier of the BED file or BEDset you want to remove.

### Cache Folder

By default, the downloaded and processed BED files are cached in the bed_cache folder. You can specify a different cache folder using the --cache-folder argument, or set the environment variable `BBCLIENT_CACHE`.
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
