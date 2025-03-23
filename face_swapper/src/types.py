from typing import Any, Dict, Literal, Tuple, TypeAlias

from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]
BatchMode = Literal['equal', 'same', 'different']

Feature : TypeAlias = Tensor
Embedding : TypeAlias = Tensor
Mask : TypeAlias = Tensor
Loss : TypeAlias = Tensor

Padding : TypeAlias = Tuple[int, int, int, int]

GeneratorModule : TypeAlias = Module
EmbedderModule : TypeAlias = Module
GazerModule : TypeAlias = Module
MotionExtractorModule : TypeAlias = Module
FaceMaskerModule : TypeAlias = Module

OptimizerSet : TypeAlias = Any

WarpTemplate = Literal['vgg_face_hq_to_arcface_128_v2', 'ffhq_to_arcface_128_v2', 'arcface_128_v2_to_arcface_112_v2']
WarpTemplateSet : TypeAlias = Dict[WarpTemplate, Tensor]
