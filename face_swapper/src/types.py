from typing import Any, Tuple, TypeAlias

from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]

Attributes : TypeAlias = Tuple[Tensor, ...]
Embedding : TypeAlias = Tensor
FaceLandmark203 : TypeAlias = Tensor

Padding : TypeAlias = Tuple[int, int, int, int]

VisionFrame : TypeAlias = NDArray[Any]

GeneratorModule : TypeAlias = Module
EmbedderModule : TypeAlias = Module
