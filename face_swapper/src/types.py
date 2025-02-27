from typing import Any, Tuple, TypeAlias

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
