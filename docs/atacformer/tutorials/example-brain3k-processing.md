# Example Brain3k Processing
This tutorial demonstrates how to process a small dataset from 10X Genomics' Brain3k multiome dataset using Atacformer. 

To start, we need to download the matrix files:

```bash
wget "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/human_brain_3k/human_brain_3k_filtered_feature_bc_matrix.tar.gz" -O  "brain3k.tar.gz"

tar -xzf brain3k.tar.gz
```

There are three files in the extracted directory: `barcodes.tsv`, `features.tsv`, and `matrix.mtx`. These files contain the cell barcodes, the feature names, and the matrix data, respectively. We will use these files to build an `AnnData` object, which is a common format for storing single-cell data. The `AnnData` object will contain the genomic regions of interest in the `.var` dataframe:

<img align="center" src="../../img/making_anndata.svg" style="margin-left: auto; margin-right: auto; display: block;">

```python
import pandas as pd
from anndata import AnnData
from scipy.io import mmread

# read in the data
mtx = mmread("matrix.mtx.gz").T.tocsr()
barcodes = pd.read_csv("barcodes.tsv.gz", header=None, names=["barcode"], sep="\t")
features = pd.read_csv("features.tsv.gz", header=None, names=["feature_id", "feature_name", "type", "chr", "start", "end"], sep="\t")

# create the AnnData object, write it to disk
adata = AnnData(
    X=mtx,
    obs=barcodes,
    var=features.set_index("feature_id"),
)
adata.write_h5ad("brain3k.h5ad")
```

We need to subset the data to only include the ATAC-seq regions.

```python
atac = adata[:, adata.var["type"] == "Peaks"]
```

Next, we will pre-tokenize the data using the Atacformer tokenizer. This step converts the genomic regions into input IDs that the Atacformer model can understand.

```python
from geniml.atacformer import AtacformerForCellClustering
from geniml.tokenization import tokenize_anndata
from gtars.tokenizers import Tokenizer

model = AtacformerForCellClustering.from_pretrained("databio/atacformer-base-hg38")

tokenizer = Tokenizer.from_pretrained("databio/atacformer-base-hg38")
tokens = tokenize_anndata(atac, tokenizer)
tokens = [t["input_ids"] for t in tokens]
```

Now we can generate single-cell embedding and attach them back to the `AnnData` object. This will allow us to use the embeddings for downstream tasks such as clustering or visualization.

```python
embeddings = model.encode_tokenized_cells(tokens)
atac.obsm["X_atacformer"] = embeddings.detach().cpu().numpy()
```

Finally, we can generate clusters and visualize the results using UMAP. 

```python
import scanpy as sc

sc.pp.neighbors(atac, use_rep="X_atacformer")
sc.tl.leiden(atac)

sc.pl.umap(atac, color=["leiden", "chr", "start", "end"], frameon=False, wspace=0.4, hspace=0.4)
```
