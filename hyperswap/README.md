HyperSwap
=========

> Hyper accurate face swapping for everyone.

![License](https://img.shields.io/badge/license-ResearchRAIL--MS-orange)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion-labs/master/.github/previews/hyperswap.png?sanitize=true)


Installation
------------

```
pip install -r requirements.txt
```


Setup
-----

This `config.ini` utilizes the VGGFace2 dataset to train the HyperSwap model.

```
[training.dataset]
file_pattern = .datasets/vggface2/**/*.jpg
convert_template = vggfacehq_512_to_arcface_128
multiplier = 1
transform_size = 256
usage_mode = both
batch_mode = same
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
generator_embedder_path = .models/blendface.pt
loss_embedder_path = .models/arcface.pt
gazer_path = .models/gazer.pt
face_masker_path = .models/face_masker.pt
```

```
[training.model.generator]
source_channels = 512
output_channels = 4096
output_size = 256
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
[training.model.masker]
input_channels = 67
output_channels = 1
num_filters = 16
```

```
[training.losses]
adversarial_weight = 1.0
cycle_weight = 1.0
feature_weight = 10.0
reconstruction_weight = 10.0
identity_weight = 20.0
gaze_weight = 0.05
mask_weight = 5.0
```

```
[training.trainer]
accumulate_size = 4
gradient_clip = 20.0
noise_factor = 0.05
max_epochs = 50
strategy = auto
precision = 16-mixed
preview_frequency = 100
```

```
[training.logger]
logger_path = .logs
logger_name = hyperswap
```

```
[training.generator]
learning_rate = 0.0004
momentum = 0.5
scheduler_factor = 0.7
scheduler_patience = 2000
```

```
[training.discriminator]
learning_rate = 0.0002
momentum = 0.5
scheduler_factor = 0.7
scheduler_patience = 2000
```

```
[training.output]
directory_path = .outputs
file_pattern = hyperswap_{epoch}_{step}
resume_path = .outputs/last.ckpt
```

```
[exporting]
directory_path = .exports
source_path = .outputs/last.ckpt
target_path = .exports/face_swapper.onnx
target_size = 256
ir_version = 10
opset_version = 15
precision = full
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


Inferencing
-----------

Inference the model.

```
python infer.py
```
