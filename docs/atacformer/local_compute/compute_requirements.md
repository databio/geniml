# Compute requirements
Atacformer is a transformer model that leverages a large context-window, due to the nature of scATAC-seq data. A typical Atacformer context-window size is 8,192 tokens. To that end, Atacformer requires a significant amount of GPU memory to use in its full capacity.

Atacformer processes data in chunks (batches). This `batch_size` is configurable and can be adjusted based on the available hardware resources and the specific requirements of the task at hand. You can use the following table as a general rule of thumb for how much GPU VRAM you need:

| Batch Size | GPU VRAM Required | Example GPUs               |
|------------|-------------------|-----------------------------|
| 4          | 2GB               | NVIDIA RTX 2080 Ti         |
| 8          | 4GB               | NVIDIA RTX 3080            |
| 16         | 8GB               | NVIDIA RTX 3090            |
| 32         | 16GB              | NVIDIA RTX 4070 Ti        |
| 64         | 48GB              | NVIDIA A40/A6000                |
