ArcFace Converter
=================

> Convert face embeddings between various ArcFace models.

![License](https://img.shields.io/badge/license-MIT-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion-labs/master/.github/preview_arcface_converter.png?sanitize=true)


Installation
------------

```
pip install -r requirements.txt
```


Setup
-----

This `config.ini` utilizes the MegaFace dataset to train an ArcFace Converter for SimSwap.

```
[preparing.dataset]
dataset_path = .datasets/megaface/train.rec
crop_size = 112
process_limit = 650000
```

```
[preparing.model]
source_path = .models/arcface_w600k_r50.onnx
target_path = .models/arcface_simswap.onnx
```

```
[preparing.input]
directory_path = .inputs
source_path = .inputs/arcface_w600k_r50.npy
target_path = .inputs/arcface_simswap.npy
```

```
[training.loader]
split_ratio = 0.8
batch_size = 51200
num_workers = 8
```

```
[training.trainer]
max_epochs = 4096
```

```
[training.output]
directory_path = .outputs
file_pattern = arcface_converter_simswap_{epoch:02d}_{val_loss:.4f}
```

```
[exporting]
directory_path = .exports
source_path = .outputs/last.ckpt
target_path = .exports/arcface_converter_simswap.onnx
opset_version = 15
```

```
[execution]
providers = CUDAExecutionProvider
```


Preparing
---------

Prepare the face embedding pairs.

```
python prepare.py
```


Training
--------

Train the ArcFace converter model.

```
python train.py
```


Exporting
---------

Export the model to ONNX.

```
python export.py
```
