<p align="center">
<h1><img align="center" src="img/atacformer_logo.svg" class="img-header" height="100"></h1>
</p>


<p align="center">
<a href="https://pypi.org/project/geniml"><img src="https://img.shields.io/pypi/v/geniml" alt=""></a>
<a href="https://github.com/databio/geniml"><img src="https://img.shields.io/badge/source-github-354a75?logo=github"></a>
</p>

Atacformer is a general-purpose transformer-based foundation model for genomic interval data, specifically designed for ATAC-seq data. It is part of the Geniml toolkit, which provides a suite of tools for genomic interval machine learning. Trained on a large corpus of ATAC-seq data, Atacformer can be fine-tuned for various downstream tasks such as classification, regression, and sequence generation.

All pre-trained models are available on the [Hugging Face Hub](https://huggingface.co/databio), and you can easily load them using the `geniml` package:

```python
from geniml.atacformer import AtacformerForCellClustering

model = AtacformerForCellClustering.from_pretrained("databio/atacformer-base-hg38")
```