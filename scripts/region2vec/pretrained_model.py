import os
import scanpy as sc
from geniml.io import RegionSet
from geniml.region2vec import Region2VecExModel

# Load the pretrained model
model = Region2VecExModel("databio/r2v-ChIP-atlas-hg38")

# Load the data
regions = RegionSet("/path/to/regions.bed")


# Embed the data
embeddings = model.encode(regions)

# TODO: need to finish with actual be files
