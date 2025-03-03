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
warp_template = vgg_face_hq_to_arcface_128_v2
batch_mode = equal
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
embedder_path = .models/arcface.pt
landmarker_path = .models/landmarker.pt
motion_extractor_path = .models/motion_extractor.pt
```

```
[training.model.generator]
encoder_type = unet-pro
identity_channels = 512
output_channels = 4096
num_blocks = 2
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
adversarial_weight = 1.5
attribute_weight = 10
reconstruction_weight = 20
identity_weight = 20
pose_weight = 0
gaze_weight = 0
```

```
[training.trainer]
learning_rate = 0.0004
max_epochs = 50
precision = 16-mixed
preview_frequency = 250
```

```
[training.output]
directory_path = .outputs
file_pattern = face_swapper_{epoch}_{step}
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
embedder_path = .models/arcface.pt
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
