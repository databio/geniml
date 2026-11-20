# Generating a UMAP from fragments files
One of the most common single-cell ATAC-seq data formats is the fragments file format [from 10X Genomics](https://www.10xgenomics.com/support/software/cell-ranger-arc/latest/analysis/outputs/fragments-file). These files contain information about the genomic regions that were accessible in individual cells during the ATAC-seq experiment.

## Getting the data
To start, lets grab an example fragments file:
```bash
wget "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/human_brain_3k/human_brain_3k_atac_fragments.tsv.gz" -O human_brain_3k_atac_fragments.tsv.gz
```

## Tokenize first
Remember that we always **tokenize first, then infer after.** We will start by tokenizing the fragments file:
```python
from gtars.tokenizers import tokenize_fragments_file, Tokenizer

tokenizer = Tokenizer.from_pretrained("databio/atacformer-base-hg38")
tokens = tokenize_fragments_file("human_brain_3k_atac_fragments.tsv.gz", tokenizer)
```

`tokens` is now a list of dictionaries, where each key is a unique cell barcode, and then each value is a list of `input_ids` for the corresponding cell.

## Basic QC
Before inferring, lets remove cells with very low and very high fragment counts:
```python
min_fragments = 200
max_fragments = 10_000

filtered_tokens = {k: v for k, v in tokens.items() if min_fragments <= len(v) <= max_fragments}
```

## Infer embeddings
Now that we have **tokenized** the fragments file and performed basic QC, we can infer embeddings for the filtered tokens:
```python
from gtars.models import Atacformer

model = Atacformer.from_pretrained("databio/atacformer-base-hg38")
embeddings = model.encode_tokenized_cells(
    input_ids=filtered_tokens.values(),
    batch_size=32
)

# detach embeddings from the computation graph and convert to numpy
embeddings = embeddings.detach().cpu().numpy()
```

## Generate UMAP
Now that we have the embeddings, we can generate a UMAP:
```python
from umap import UMAP
import matplotlib.pyplot as plt

reducer = UMAP(n_components=2, random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])
plt.show()
```