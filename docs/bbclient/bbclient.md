# BBClient

The BEDbase client command `bbclient` downloads, processes, and caches BED files and BED sets from the BEDbase API and converts them into RegionSet, GenomicRanges or GenomicRangesList objects.
It provides various commands to interact with BED files, including downloading individual files, downloading BEDsets, processing local BED files, and processing BED file identifiers.

This document provides tutorials for using `bbclient` via either:

1. the 🐍 [Python interface](#python-interface),
2. the 🖥️ [command-line interface](#command-line-interface), or 
3. the 🦀 [Rust interface](#rust-interface).

## 🐍 Python interface

#### Instance of BBClient:
First, we need to create an instance of the `BBClient` class. Optionally, you can specify a custom cache folder to store downloaded BED files. If not specified, it defaults to `./.bb_cache`.
```python
from geniml.bbclient import BBClient

bbclient = BBClient(cache_folder="cache")
```

#### Load a BED file from cache, or download and cache a remote BED file from BEDbase
THis function will try to find the BED file in the cache first. If not found, it will download it from BEDbase, process it, cache it, and return a `RegionSet` object.
```python
bedfile_id = "...."  # find interesting bedfile on bedbase e.g. "3f0aaadf6e8854282f21ea19ab5c061a"
bedfile = bbclient.load_bed(bedfile_id)  # download, cache and return a RegionSet object
```

#### Cache a local BED file
You can just provide a URL, Path or RegionSet object and it will add to cache for you:
```python
from gtars.models import RegionSet

bedfile = RegionSet("path/to/bedfile")

# Alternatively, you can provide a URL:
# bedfile = "https://example.com/path/to/bedfile.bed.gz"
# Or a local file path:
# bedfile = "path/to/bedfile"
bedfile_id = bbclient.add_bed_to_cache(bedfile)
```

#### Download and cache a **BEDset** from BEDbase

```python
bedset_identifier = "xyz" # find some interesting bedset on bedbase.org
bedset = bbclient.load_bedset(bedset_identifier)  # download, cache and return a BedSet object
grl = bedset.to_granges_list()  # return a GenomicRangesList object
```

#### Cache a local BEDset

```python
from geniml.io import BedSet

bedset_folder = "path/to/bed/files/folder"
bedset = BedSet(
    [os.path.join(bedset-folder, file_name) for file_name in os.listdir(bedset_folder)]
)
bedset_id = bbclient.add_bedset_to_cache(bedset)
```

---
### 🖥️ Command line interface

#### Cache BED file

```bash
geniml bbclient cache-bed <BED_file_or_identifier_or_url>
```
The `<BED_file_or_identifier_or_url>` variable can be one of 3 things:

1. a path to a local BED file;
2. a BED record identifier from BEDbase; or,
3. a URL to a BED file hosted anywhere.

#### Cache BEDset

```bash
geniml bbclient cache-bedset <BED_files_folder_or_identifier>
```
The `<BED_files_folder_or_identifier>` variable may be:

1. local path to a folder containing BED files; or,
2. a BEDbase BEDset identifier


#### Seek the path of a BED file or BEDset in cache folder
To retrieve the local file path to a BED file stored locally,

```bash
geniml bbclient seek <identifier>
```
where <identifier> is the BED file or BEDset identifier.


#### Count BED files that are cached

```bash
geniml bbclient inspect-bedfiles
```

#### Count BEDsets that are cached

```bash
geniml bbclient inspect-bedsets
```

#### Remove a BED file or BEDset from the cache folder 

```bash
geniml bbclient rm <identifier>
```
where <identifier> is the BED file or BEDset identifier.

---

## 🦀 Rust interface

####  Install gtars

```bash
cargo install gtars
```

#### Cache a BED file

```rust
use gtars::bbclient::BBClient;

let mut bbc = BBClient::new(Some(cache_folder.clone()), None).expect("Failed to create BBClient");
```

where first argument is an optional cache folder path, and the second argument is an optional BEDbase API URL.

#### Add local BED file to cache

```rust
let bed_id: String = bbc
            .add_local_bed_to_cache(PathBuf::from(_path/to.bed.gz), None)
            .unwrap();
```

#### Add RegionSet object to cache

```rust
use gtars::models::RegionSet;
let rs: RegionSet = RegionSet::try_from("path/to/bedfile").unwrap();
let bed_id: String = bbc
            .add_regionset_to_cache(rs, None)
            .unwrap();
```

#### Delete a BED file from cache

```rust
let result: Result<()> = bbc.remove("bed_id")
            .expect("Failed to remove bedset file and its bed files");
            
```


--- 

## 📂 Cache Folder

By default, the downloaded and cached BED files are cached in the `./.bbcache` folder. 
You can specify a different cache folder using the --cache-folder argument, or set the environment variable `BBCLIENT_CACHE`.

```bash
export BBCLIENT_CACHE="/path/to/cache/folder"
```

## 🔗 BEDbase API
By default the `bbclient` connects to the public BEDbase instance at `https://api.bedbase.org`.
To set a different BEDbase instance set the environment variable `BEDBASE_API`:
```bash
export BEDBASE_API="https://your.bedbase.instance/api"
```