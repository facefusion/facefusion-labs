from typing import Any, Dict, Literal, Tuple, TypeAlias

from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]
BatchMode = Literal['equal', 'same']

Attributes : TypeAlias = Tuple[Tensor, ...]
Embedding : TypeAlias = Tensor
Gaze : TypeAlias = Tuple[Tensor, Tensor]

Padding : TypeAlias = Tuple[int, int, int, int]

GeneratorModule : TypeAlias = Module
EmbedderModule : TypeAlias = Module
GazerModule : TypeAlias = Module
MotionExtractorModule : TypeAlias = Module
ParserModule : TypeAlias = Module

OptimizerSet : TypeAlias = Any

WarpTemplate = Literal['vgg_face_hq_to_arcface_128_v2', 'arcface_128_v2_to_arcface_112_v2']
WarpTemplateSet : TypeAlias = Dict[WarpTemplate, Tensor]
