from typing import Any, Dict, Tuple, TypeAlias, Type, Literal

from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]

Attributes : TypeAlias = Tuple[Tensor, ...]
Embedding : TypeAlias = Tensor
FaceLandmark203 : TypeAlias = Tensor

Padding : TypeAlias = Tuple[int, int, int, int]

GeneratorModule : TypeAlias = Module
EmbedderModule : TypeAlias = Module
LandmarkerModule : TypeAlias = Module
MotionExtractorModule : TypeAlias = Module

OptimizerConfig : TypeAlias = Any

WarpMatrix = Literal['vgg_face_hq_to_arcface_128_v2', 'arcface_128_v2_to_arcface_112_v2']
WarpMatrixSet : TypeAlias = Dict[WarpMatrix, Tensor]
