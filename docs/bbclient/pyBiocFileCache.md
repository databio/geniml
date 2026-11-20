# How to use bbclient cache in R

The bbclient caching system is supported in both Python and R. 
Both of these caching systems are compatible with `BiocFileCache` for BED files and `Zarr` with BED tokens.


## Bed caching

### Step 1: Cache BED File Using bbclient CLI
```terminal
geniml bbclient cache-bed bbad85f21962bb8d972444f7f9a3a932
```

The cached file is saved in the default bbclient cache folder, or a user-provided cache folder. 
Since we didn't provide a caching folder, the default folder will be in our home directory.

### Step 2: Get Path to the BED File Using Python

#### a. Using bbclient in Python

Here's an example of how you can retrieve the path to the BED file in Python:
```python
from geniml.bbclient import BBClient

bbc = BBClient()
print(bbc.seek("bbad85f21962bb8d972444f7f9a3a932"))
```

And it will print path to our file:
```text
'/home/bnt4me/.bbcache/bedfiles/b/b/bbad85f21962bb8d972444f7f9a3a932.bed.gz'
```

b. Using pyBiocFileCache

```python
import os

from pybiocfilecache import BiocFileCache

# get cache folder
bbcache_folder = os.path.join(os.path.expanduser("~"), ".bbcache")

# get bedfile cache folder 
bedfile_cache_folder = os.path.join(bbcache_folder, "bedfiles")

bio_cache = BiocFileCache(bedfile_cache_folder)

bed_cache_obj = bio_cache.get("bbad85f21962bb8d972444f7f9a3a932")

print(bed_cache_obj.fpath)
```
And it will print path to our file:
```text
'/home/bnt4me/.bbcache/bedfiles/b/b/bbad85f21962bb8d972444f7f9a3a932.bed.gz'
```

3. Get path using R caching system - BiocFileCache

```R
# Install necessary package
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install("BiocFileCache")

# Load necessary libraries
library(BiocFileCache)

# Get cache folder
bbcache_folder <- file.path(Sys.getenv("HOME"), ".bbcache")

# Get bedfile cache folder
bedfile_cache_folder <- file.path(bbcache_folder, "bedfiles")

# Create a BiocFileCache object
bio_cache <- BiocFileCache(bedfile_cache_folder)

# Get the cached file using the identifier
bed_cache_obj <- bfcrpath(bio_cache, "bbad85f21962bb8d972444f7f9a3a932")

# Print the file path
print(bed_cache_obj)
```
And we will get this message printed:
```text
[1] "/home/bnt4me/.bbcache/bedfiles/b/b/bbad85f21962bb8d972444f7f9a3a932.bed.gz"
```


This R script will perform the same tasks as the Python script: handling the cache and retrieving the specified BED file from the cache.

By following these steps, you can efficiently manage and retrieve cached BED files using the bbclient caching system in both Python and R.

P.S. All links will be different on your machine, as they are generated based on your local home directory.

____


## Caching of tokenized bed files

To store tokenized BED files, we use the Zarr format. 
BBClient saves tokenized files in the zarr folder within the bbcache folder (which is located in our home directory if the bbcache folder is not specified).

Here's an example of how you can download and cache tokenized bed file using bbclient:
```python
from geniml.bbclient import BBClient

bbc = BBClient()
bbc.add_bed_tokens_to_cache( bed_id= '0dcdf8986a72a3d85805bbc9493a1302', universe_id= '58dee1672b7e581c8e1312bd4ca6b3c7')
```
If user didn't get any error, the tokenized file is saved in the default bbclient cache folder, or a user-provided cache folder.


### Step 1: Get zarr tokenized bed file using bbclient in Python
```python
from geniml.bbclient import BBClient

bbc = BBClient()
tokens_arr = bbc.load_bed_tokens(bed_id= '0dcdf8986a72a3d85805bbc9493a1302', universe_id= '58dee1672b7e581c8e1312bd4ca6b3c7')

print(tokens_arr)
```
Result is a zarr array object:
```text
<zarr.core.Array '/58dee1672b7e581c8e1312bd4ca6b3c7/0dcdf8986a72a3d85805bbc9493a1302' (29438,) int64>
```

### Step 2: Get zarr tokenized bed file using Python zarr library
```python
import os 
import zarr


bbcache_folder = os.path.join(os.path.expanduser("~"), ".bbcache")

# get zarr cache folder 
bedfile_cache_folder = os.path.join(bbcache_folder, "tokens.zarr")

zarr_cache = zarr.group(bedfile_cache_folder)

universe_id = "58dee1672b7e581c8e1312bd4ca6b3c7"
bed_id = "0dcdf8986a72a3d85805bbc9493a1302"

tokens_arr = zarr_cache[universe_id][bed_id]
print(tokens_arr)
```
Result is a zarr array object:
```text
<zarr.core.Array '/58dee1672b7e581c8e1312bd4ca6b3c7/0dcdf8986a72a3d85805bbc9493a1302' (29438,) int64>
```

### Step 3: Get zarr tokenized bed file using R zarr library
```R
## we need BiocManager to perform the installation
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
## install Rarr
BiocManager::install("Rarr")

library(Rarr)

```
🚧 Use `Rarr` library to open zarr file: https://github.com/grimbough/Rarr?tab=readme-ov-file
