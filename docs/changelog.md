# Changelog

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format. 


# [0.8.1] -- 2025-09-23

- Fixed bedspace 
- Fixed tests in bbclient 
- Removed unused tests and files


# [0.8.0] -- 2025-09-23

- Added Atacformer

## [0.7.0] -- 2025-04-15

- Improved encoding (indexing) of the bed file (region2vec)
- Updated code due to recent changes in the gtars
- Switched RegionSet from io module to Gtars regionSet

## [0.6.0] -- 2025-01-03

- Fixed: biocfilecache (upgraded to v0.6.0)
- Added: inspect functions that lists all cached beds and bedsets

## [0.5.2] -- 2024-12-03

- Switched bivec search to fastembed
- Fixed: incorrect caching of the files

## [0.5.1] -- 2024-10-18

- Fixed: Fix search score. #187
- Fixed: Batch search for Qdrant - Speed up search #186
- Fixed: Some packages not in requirements #189

## [0.5.0] -- 2024-10-15

- Added: new bi-vector search

## [0.4.3] -- 2024-12-09

- Fixed: required torch version


## [0.4.2] -- 2024-10-07

- Cleaned dependencies and separated into 2 groups: basic and ml
- Improved efficiency of CLI
- Fixed bbcache incorrectly saving files from url #175


## [0.4.1] -- 2024-09-18

- Fixed: bbclient ignores --cache-folder on CLI #164
- Fixed bug in opening bed files


## [0.4.0] -- 2024-06-04

- Added bed tokens caching to bbclient [bbclient] Add tokenized file cache and download #153
- Added pyBiocFileCache for bedfiles to support R caching [bbclient] Integrate bedbase caching with R #151
- Added support of Python3.12
- Optimized encoding of regions for Region2Vec models
- Added updates to the new Atacformer
- Renamed tokenizers to TreeTokenizer and AnnDataTokenizer
- Bump genimtools which includes performance upgrades and stability updates to the tokenizers

## [0.3.0] -- 2024-04-04

- Added S3 uploading to bbclient
- Added and cleaned tests
- Added bed2bed search interface

## [0.2.0] -- 2024-02-20

- Fixed a bug with pydantic
- Integrate `lightning` for easier training of models with SLURM and the [DDP framework](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- New datasets for streaming `.gtok` files to models
- New tutorials for updated ScEmbed
- Start working on `Atacformer`

## [0.1.0] -- 2023-12-18

- First official released version of geniml
- Integrated bedshift 1.1.1 into geniml (excluding intersection support)


## [bedshift 1.1.1] - 2021-04-15

- Updated documentation
- Fixed dependencies packaging for building documentation

## [bedshift 1.1.0] - 2021-04-02

- Added ability to specify chrom sizes file, or refgenie genome.
- Add perturbation will create regions proportional to chromosome size
- Improve performance of perturbations
- Add --dropfile, --addfile, and --shiftfile options
- Add --add_valid option
- Add --yaml-config option
- Improved testing framework
- Improved logging messages

## [bedshift 1.0.0] - 2020-05-20

- Add, shift, drop, cut, and merge perturbations.
- Basic documentation
- Basic tests
