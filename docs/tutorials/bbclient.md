# BED File Retrieval and Processing

This client downloads, processes, and caches BED files and BED sets from the BEDbase API and converts them into a GenomicRanges or GenomicRangesList object. It provides various commands to interact with BED files, including downloading individual files, downloading BEDsets, processing local BED files, and processing BED file identifiers.

## Usage

###
```python
from geniml.bbclient import BBClient
from geniml.io import RegionSet
```

### Create an Instance of the BBClient Class:

```python
bbc = BBClient(cache_folder="<cache_folder_path>", bedbase_api="<bedbase_api>")
```

### Cache a Local BED File
```python
bbclient.add_bed_to_cache(RegionSet("<local_bed_file_path>"))
```

### Download a BEDset
```python
bedset = bbclient.load_bedset("<bed_identifier>")
```

### Download and Process BED File Identifiers
```python
regionset = bbclient.load_bed(<input_identifier(s)>)
```

## For command line usage, run the CLI with appropriate subcommands and arguments as described below:

### Download and Process BED File Identifiers

```bash
geniml bbclient identifiers --input-identifier <bed_identifier>
```

Replace <bed_identifier> with either the BED file identifier or a file containing a list of BED file identifiers.

### Download and Process a BED Region

```bash
geniml bbclient region --input-identifier <bed_identifier> --chr <chromosome_number> --start <start_position> --end <end_position>
```

Replace <bed_identifier> with the identifier of the BED file, <chromosome_number> with the chromosome number, <start_position> with the start position, and <end_position> with the end position of the region you want to download and process.

### Process a Local BED File

```bash
geniml bbclient local --input-identifier <local_bed_file_path>
```

Replace <local_bed_file_path> with the path to the local BED file you want to process.

### Download a BEDset

```bash
geniml bbclient bedset --bedset <bedset_identifier>
```

Replace <bedset_identifier> with the identifier of the BEDset you want to download.

### Command-Line Arguments

    bedset: Download a BEDset.
    region: Download and process a BED region.
    local: Process a local BED file.
    identifiers: Process BED file identifiers.

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
