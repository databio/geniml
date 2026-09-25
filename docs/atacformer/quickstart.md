# Quickstart - generate single-cell embeddings with Atacformer
This quickstart guide will help you generate single-cell embeddings using Atacformer.

## Installation
To install Atacformer, you need to have the `geniml` package installed. You can do this using pip:

```bash
pip install geniml[ml] # install with machine learning dependencies
```

Test the installation by importing Atacformer in Python:

```bash
python -c "from geniml import __version__; print(__version__)"
```

## Loading a Pre-trained Model
You can easily load a pre-trained Atacformer model from the Hugging Face Hub. For example, you can load the base model described in our paper:

```python
from geniml.atacformer import AtacformerForCellClustering

model = AtacformerForCellClustering.from_pretrained("databio/atacformer-base-hg38")
model = model.to("cuda")  # move the model to GPU (...if available)
```

## Tokenize your data
To generate embeddings, you need to tokenize your genomic interval data. Frequently, this data is stored in AnnData format. You can use the `geniml` package to tokenize your AnnData object.

```python
import scanpy as sc

from gtars.tokenizers import Tokenizer
from geniml.tokenization.utils import tokenize_anndata

tokenizer = Tokenizer.from_pretrained("databio/atacformer-base-hg38")
adata = sc.read_h5ad("path/to/your/anndata.h5ad")

tokens = tokenize_anndata(adata, tokenizer)

input_ids = [t["input_ids"] for t in tokens]
```

Alternatively, you can tokenize `.fragments.tsv.gz` files directly:

```python
from tqdm import tqdm
from gtars.tokenizers import tokenize_fragment_file

tokenized_data = tokenize_fragment_file("path/to/your/fragments.tsv.gz", tokenizer)

# qc 
MIN_COUNT = 500
MAX_COUNT = 100_000

# filter down low and high count cells
tokens_filtered = [(barcode, ids) for barcode, ids in tqdm(tokens, desc="Filtering tokens", total=len(tokens)) if len(ids) >= MIN_COUNT and len(ids) <= MAX_COUNT]
tokens_filtered = [(barcode, list(set(ids))) for barcode, ids in tqdm(tokens_filtered, desc="Removing duplicates", total=len(tokens_filtered))]

input_ids = [ids for _, ids in tokens_filtered]
```

## Generate Embeddings
Once you have your tokenized data, you can generate embeddings using the model:

```python
cell_embeddings = model.encode_tokenized_cells(
    input_ids=input_ids,
    batch_size=32,  # adjust based on your memory capacity
)
```

## Downstream Tasks
You can use the generated embeddings for various downstream tasks such as clustering, classification, or visualization.