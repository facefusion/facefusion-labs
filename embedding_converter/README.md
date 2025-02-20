Embedding Converter
===================

> Convert face embeddings between various models.

![License](https://img.shields.io/badge/license-OpenRAIL--MS-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion-labs/next/.github/previews/embedding_converter.png?sanitize=true)


Installation
------------

```
pip install -r requirements.txt
```


Setup
-----

This `config.ini` utilizes the MegaFace dataset to train the Embedding Converter for SimSwap.

```
[preparing.dataset]
dataset_path = .datasets/images
image_pattern = {}/*.*g
```

```
[training.loader]
split_ratio = 0.8
batch_size = 256
num_workers = 8
```

```
[training.model]
source_path = .models/arcface_w600k_r50.pt
target_path = .models/arcface_simswap.pt
```

```
[training.trainer]
learning_rate = 0.001
max_epochs = 4096
```

```
[training.output]
directory_path = .outputs
file_pattern = arcface_converter_simswap_{epoch}_{step}
resume_file_path = .outputs/last.ckpt
```

```
[exporting]
directory_path = .exports
source_path = .outputs/last.ckpt
target_path = .exports/arcface_converter_simswap.onnx
ir_version = 10
opset_version = 15
```


Training
--------

Train the Embedding Converter model.

```
python train.py
```

Launch the TensorBoard to monitor the training.

```
tensorboard --logdir=.logs
```


Exporting
---------

Export the model to ONNX.

```
python export.py
```
