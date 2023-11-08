# Module overviews

`geniml` is organized into modules. Each module groups together related tasks. This document provides an overview of each module.

## Module `assess-universe`

Many genomic interval analysis methods, particularly those used by `geniml` require that regions be re-defined in terms of a consensus region set, or universe. However, a universe may not be a good fit to a collection of files. This module assesses that fit. Given a collection of genomic interval sets, and a proposed universe, we can assess how well the universe fits the genomic interval sets. This module provides several complementary methods to assess fit.

## Module `bbclient`

The `bbclient` module can download BED files or BED sets from [BEDbase](https://bedbase.org/) and cache them into local folders.

## Module `bedspace`

The `bedspace` module uses the StarSpace method (Wu et al., 2018) to jointly embed genomic interval regions sets with associated metadata into a shared latent embedding space. This facilitates fast search and retrieval of similar region sets and their associated metadata. 

## Module `build-universe`

This module provides multiple ways to build a genomic region universe. These include: 1. **HMM**: uses an HMM to create a flexible segment universe, given an input of several bed files.

## Module `evaluation`

Once a `geniml` region embedding model is trained, we may want to evaluate the embeddings. The `evaluation` module provides several functions for that. These include statistical tests, like the Cluster Tendency Test (CTT) and the Reconstruction Test (RCT), and biological tests, the Genome Distance Scaling Test (GDST) and the Neighborhood Preserving Test (NPT). These evaluation metrics can be helpful to determine if your models are working well, optimize training parameters, etc.

## Module `region2vec`

`Region2Vec` is an unsupervised method for creating embeddings for genomic regions and region sets from a set of raw BED files. The program uses a variation of the word2vec algorithm by building shuffled context windows from BED files. The co-occurence statistics of genomic regions in a collection of BED files allow the model to learn region embeddings.

## Module `scembed`

`scEmbed` is a single-cell implementation of `region2Vec`: a method to represent genomic region sets as vectors, or embeddings, using an adapted word2vec approach. `scEmbed` allows for dimensionality reduction and feature selection of single-cell ATAC-seq data; a notoriously sparse and high-dimensional data type. We intend for `scEmbed` to be used with the [`scanpy`](https://scanpy.readthedocs.io/en/stable/) package. As such, it natively accepts `AnnData` objects as input and returns `AnnData` objects as output.

## Module `search`

The search module provides a generic interface for vector search. Several geniml packages (such as `region2vec`), will create embeddings for different entities. The search module provides interfaces that store vectors and perform fast k-nearest neighbors (KNN) search with a given query vector.  Options include a database backend (using [`qdrant-client`](https://github.com/qdrant/qdrant-client)) and  local file backend (using [`hnswlib`](https://github.com/nmslib/hnswlib).

## Module `text2bednn`

`Vec2Vec` is a feedforward neural network that maps vectors from the embedding space of natural language (such as embeddings created by [`SentenceTransformers`](https://www.sbert.net/)) to the embedding space of BED (such as embeddings created by `Region2Vec`). By mapping the embedding of natural language query strings to the space of BED files, `Vec2Vec` can perform natural language search of BED files. 

## Module `tokenization`

In NLP, training word embeddings requires first tokenizing words such that words in different forms are represented by one word. For example, "orange", "oranges" and "Orange" are all mapped to "orange" since they essentially convey the same meaning. This reduces the vocabulary size and improves the quality of learned embeddings. Similary, many `geniml` modules (such as `region2vec`) require first tokenizating regions.

To tokenize reigons, we need to provide a universe, which specifies the "vocabulary" of genomic regions. The universe is a BED file, containing representative regions. With the given universe, we represent (tokenize) raw regions into the regions in the universe.

Different strategies can be used to tokenize. The simplest case we call *hard tokenization*, which means if the overlap between a raw region in a BED file and a region in the universe exceeds a certain amount, then we use the region in the universe to represent this raw region; otherwise, we ignore this raw region. This is a "zero or one" process. After hard tokenization, each BED file will contain only regions from the universe, and the number of regions will be smaller or equal to the original number.



