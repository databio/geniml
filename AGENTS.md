# geniml -- Genomic Interval Machine Learning

Builds vector embeddings and ML models from genomic interval data (BED files), enabling similarity search, clustering, and classification of genomic region sets.

## Install

```
pip install geniml          # base (no ML deps)
pip install geniml[ml]      # torch, transformers, gensim
pip install geniml[sc]      # scanpy, anndata
pip install geniml[search]  # qdrant-client, fastembed
pip install geniml[all]     # everything
```

## Quick start

```python
from geniml.region2vec import Region2VecExModel

model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38")
vec = model.encode("peaks.bed")
```

## Submodules

- **region2vec** -- Embed BED files into vectors (Region2VecExModel)
- **scembed** -- Single-cell embeddings (wraps region2vec for AnnData)
- **search** -- Vector similarity search (BED2BED, Text2BED; HNSW/Qdrant backends)
- **bbclient** -- Download BED files from BEDbase (BBClient)
- **atacformer** -- Transformer for scATAC-seq
- **geneformer** -- Transformer for scRNA-seq
- **craft** -- Contrastive model for gene activity
- **bedspace** -- StarSpace-based BED embedding (requires external binary)
- **assess/likelihood** -- Region set overlap, distance, and likelihood stats
- **io** -- BedSet class. For single region sets, prefer `gtars.models.RegionSet`

## Dependency gating

Heavy submodules (region2vec, scembed, atacformer, etc.) are NOT imported by `import geniml`. Import them directly: `from geniml.region2vec import Region2VecExModel`. They will fail with ImportError if the matching optional dep group is not installed.

## Deprecated -- do not use

- `geniml.io.Region`, `geniml.io.RegionSet` -- use `gtars.models.Region/RegionSet`
- `geniml.region2vec.main_legacy` -- replaced by Region2VecExModel
- `geniml.text2bednn` -- use `search.Text2BEDSearchInterface`
- `geniml.nn` -- internal utilities, not public API
