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
[training.dataset]
file_pattern = .datasets/vggface2/**/*.jpg
batch_ratio = 0.2
```

```
[training.loader]
batch_size = 8
num_workers = 8
split_ratio = 0.9995
```

```
[training.model]
id_embedder_path = .models/id_embedder.pt
landmarker_path = .models/landmarker.pt
motion_extractor_path = .models/motion_extractor.pt
```

```
[training.model.generator]
encoder_type = unet-pro
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
weight_adversarial = 1.5
weight_identity = 15
weight_attribute = 10
weight_reconstruction = 15
weight_pose = 0
weight_gaze = 0
```

```
[training.trainer]
learning_rate = 0.0004
max_epochs = 50
precision = 16-mixed
automatic_optimization = false
preview_frequency = 250
```

```
[training.output]
directory_path = .outputs
file_pattern = face-swapper_{epoch}_{step}
resume_path = .outputs/last.ckpt
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


Inferencing
-----------

Inference the model.

```
python infer.py
```
