# BBClient practical example 

This tutorial walks through a practical Python workflow using `bbclient` to download, cache, and work with BED files from BEDbase.

## Installation

```bash
pip install geniml
```

## 1. Create a BBClient instance

```python
from geniml.bbclient import BBClient

# Use the default cache folder (~/.bbcache), or specify a custom path
bbc = BBClient(cache_folder="my_cache")
```

## 2. Load a BED file from BEDbase

`load_bed` checks the local cache first. If the file is not cached, it downloads it from BEDbase, caches it, and returns a `RegionSet` object.

```python
bed_id = "dcc005e8761ad5599545cc538f6a2a4d"

regionset = bbc.load_bed(bed_id)
print(regionset)
```

```
RegionSet with 42193 regions.
```

## 3. Inspect the RegionSet

Once you have a `RegionSet`, you can inspect its contents and compute basic statistics.

```python
# Number of regions
print(len(regionset))

# Mean width of regions
print(regionset.mean_region_width())

# Total nucleotide length covered
print(regionset.get_nucleotide_length())

# Last base pair position per chromosome
print(regionset.get_max_end_per_chr())

# Unique identifier (digest) of the region set
print(regionset.identifier)
```

## 4. Iterate over regions

You can iterate over individual regions in the `RegionSet`:

```python
for region in regionset:
    print(region.chr, region.start, region.end)
    break  # print just the first region
```

```
chr1 778544 778794
```

## 5. Find the cached file path

After loading, the file is stored locally. Use `seek` to get its path:

```python
path = bbc.seek(bed_id)
print(path)
```

```
/home/user/.bbcache/bedfiles/d/c/dcc005e8761ad5599545cc538f6a2a4d.bed.gz
```

## 6. Save the RegionSet to a local file

```python
# Save as plain BED
regionset.to_bed("output.bed")

# Save as gzipped BED
regionset.to_bed_gz("output.bed.gz")
```

## 7. Cache a local BED file

If you have a BED file on disk and want to add it to the cache:

```python
local_id = bbc.add_bed_to_cache("path/to/local.bed.gz")
print(local_id)  # returns the computed identifier
```

## 8. Remove a file from cache

```python
bbc.remove_bedfile_from_cache(bed_id)
```

## Full example

```python
from geniml.bbclient import BBClient

# Initialize client
bbc = BBClient(cache_folder="my_cache")

bed_id = "dcc005e8761ad5599545cc538f6a2a4d"

# Download from BEDbase and cache locally
regionset = bbc.load_bed(bed_id)

# Inspect
print(f"Regions:          {len(regionset)}")
print(f"Mean width:       {regionset.mean_region_width():.1f} bp")
print(f"Total coverage:   {regionset.get_nucleotide_length()} bp")
print(f"Identifier:       {regionset.identifier}")

# Find cached file
print(f"Cached at:        {bbc.seek(bed_id)}")

# Save locally
regionset.to_bed_gz("dcc005e8761ad5599545cc538f6a2a4d.bed.gz")
```

!!! info
    Full bbclient reference is available in the [BBClient documentation](../bbclient/bbclient.md).
