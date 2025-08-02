CrossFace
=========

> Seamless face embedding across various models.

![License](https://img.shields.io/badge/license-OpenRAIL--MS-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion-labs/master/.github/previews/crossface.png?sanitize=true)


Installation
------------

```
pip install -r requirements.txt
```


Setup
-----

This `config.ini` utilizes the MegaFace dataset to train the CrossFace model for SimSwap.

```
[training.dataset]
file_pattern = .datasets/megaface/**/*.jpg
```

```
[training.loader]
batch_size = 256
num_workers = 8
split_ratio = 0.95
```

```
[training.model]
source_path = .models/arcface_w600k_r50.pt
target_path = .models/arcface_simswap.pt
```

```
[training.trainer]
max_epochs = 4096
compile_mode = reduce-overhead
strategy = auto
precision = 16-mixed
```

```
[training.optimizer]
learning_rate = 0.001
```

```
[training.logger]
logger_path = .logs
logger_name = crossface_simswap
```

```
[training.output]
directory_path = .outputs
file_pattern = crossface_simswap_{epoch}_{step}
resume_path = .outputs/last.ckpt
```

```
[exporting]
directory_path = .exports
source_path = .outputs/last.ckpt
target_path = .exports/crossface_simswap.onnx
ir_version = 10
opset_version = 15
```


Training
--------

Train the model.

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
