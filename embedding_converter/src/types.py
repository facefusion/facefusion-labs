from typing import Any, List, TypeAlias

from numpy.typing import NDArray
from torch import Tensor

Batch : TypeAlias = Tensor
Embedding : TypeAlias = Tensor
VisionFrame : TypeAlias = NDArray[Any]
