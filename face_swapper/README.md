Face Swapper
============

> Face shape and feature aware identity transfer.

![License](https://img.shields.io/badge/license-ResearchRAIL--MS-red)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion-labs/next/.github/previews/face_swapper.png?sanitize=true)


Installation
------------

```
pip install -r requirements.txt
```


Setup
-----

This `config.ini` utilizes the MegaFace dataset to train the Face Swapper model.

```
[preparing.dataset]
dataset_path = .datasets/train
directory_pattern = {}/*
image_pattern = {}/*.*g
same_person_probability = 0.2
```

```
[training.loader]
batch_size = 8
num_workers = 8
```

```
[training.model]
id_embedder_path = .models/id_embedder.pt
landmarker_path = .models/landmarker.pt
motion_extractor_path = .models/motion_extractor.pt
```

```
[training.model.generator]
num_blocks = 2
id_channels = 512
```

```
[training.model.discriminator]
input_channels = 3
num_filters = 64
num_layers = 5
num_discriminators = 3
kernel_size = 4
```

```
[training.losses]
weight_adversarial = 1
weight_identity = 20
weight_attribute = 10
weight_reconstruction = 10
weight_pose = 100
weight_gaze = 10
```

```
[training.trainer]
learning_rate = 0.0004
max_epochs = 50
precision = 16-mixed
automatic_optimization = false
```

```
[training.output]
directory_path = .outputs
file_path = .outputs/last.ckpt
file_pattern = 'checkpoint-{epoch}-{step}-{l_G:.4f}-{l_D:.4f}'
preview_frequency = 250
validation_frequency = 1000
```

```
[exporting]
directory_path = .exports
source_path = .outputs/last.ckpt
target_path = .exports/face_swapper.onnx
ir_version = 10
opset_version = 15
```

```
[inferencing]
generator_path = .outputs/last.ckpt
id_embedder_path = .models/id_embedder.pt
source_path = .assets/source.jpg
target_path = .assets/target.jpg
output_path = .outputs/output.jpg
```


Training
--------

Train the Face Swapper model.

```
python train.py
```


Exporting
---------

Export the model to ONNX.

```
python export.py
```


Inferencing
-----------

Inference the model.

```
python infer.py
```
