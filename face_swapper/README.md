Face Swapper
=================

> Swap one face over another face.

![License](https://img.shields.io/badge/license-MIT-green)


Preview
-------

![Preview]()


Installation
------------

```
pip install -r requirements.txt
```


Example
-------

This example utilizes the MegaFace dataset to train an ArcFace Converter for SimSwap.

```
[preparing.dataset]
dataset_path = datasets/train
folder_pattern = {}/*
image_pattern = {}/*.*g
same_person_probability = 0.2

[training.loader]
batch_size = 24
num_workers = 12

[training.model]
id_embedder_path = assets/models/id_embedder.pt
landmarker_path = assets/models/landmarker.pt
motion_extractor_path = assets/models/motion_extractor.pt

[training.model.generator]
num_blocks = 2
id_channels = 512

[training.model.discriminator]
input_channels = 3
num_filters = 64
num_layers = 5
num_discriminators = 3
kernel_size = 4

[training.losses]
weight_adversarial = 1
weight_id = 20
weight_attribute = 10
weight_reconstruction = 10
weight_pose = 100

[training.trainer]
max_epochs = 50
learning_rate = 0.0004
precision = 16-mixed
automatic_optimization = false

[training.output]
checkpoint_path = outputs/last.ckpt
directory_path = outputs
file_pattern = 'checkpoint-{epoch}-{step}-{l_G:.4f}-{l_D:.4f}'
preview_frequency = 250
validation_frequency = 1000

[exporting]
directory_path = export
source_path = outputs/last.ckpt
target_path = export/face_swapper.onnx
opset_version = 15

[inference]
generator_path = outputs/last.ckpt
id_embedder_path = assets/models/id_embedder.pt
source_path = assets/images/source.jpg
target_path = assets/models/target.jpg
output_path = outputs/output.jpg
```


Training
--------

Train the Face swapper model.

```
python train.py
```


Exporting
---------

Export the model to ONNX.

```
python export.py
```
