from typing import Tuple, TypeAlias

from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]

Attribute : TypeAlias = Tensor
Attributes : TypeAlias = Tuple[Tensor, ...]
Embedding : TypeAlias = Tensor
FaceLandmark203 : TypeAlias = Tensor

Padding : TypeAlias = Tuple[int, int, int, int]

GeneratorModule : TypeAlias = Module
EmbedderModule : TypeAlias = Module
