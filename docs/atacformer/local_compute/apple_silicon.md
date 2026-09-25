Apple's M-series chips have become increasingly popular for local-first machine learning tasks. Conveniently, PyTorch has [first-class support](https://docs.pytorch.org/docs/stable/notes/mps.html) for Apple Silicon, making it a great choice for running Atacformer models on these devices.

In most cases, it could be as simple as moving your model to the `mps` device:

```python
from geniml.atacformer import AtacformerForCellClustering

model = AtacformerForCellClustering.from_pretrained("databio/atacformer-base-hg38")
model = model.to("mps")
```

However, at the moment, [not all tensor operations are supported on the `mps` device](https://github.com/pytorch/pytorch/issues/77764). One of these is the `aten::_nested_tensor_from_mask_left_aligned` operation inside the `TransformerEncoder` class, which is used in the Atacformer model.

To work around this, we've provided a patch that modifies the `TransformerEncoder` to use a different operation that is supported on the `mps` device. You can use it like so:

```python
from geniml.atacformer import AtacformerForCellClustering, patch_atacformer_model_for_mps

model = AtacformerForCellClustering.from_pretrained("databio/atacformer-base-hg38")
model = model.to("mps")

patch_atacformer_model_for_mps(model)

# use the model as usual
# ...
```