from typing import Any, Tuple

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

Batch = Tuple[Tensor, Tensor, int]
Loader = DataLoader[Tuple[Tensor, ...]]
UNetAttributes = Tuple[Tensor, ...]

Embedding = NDArray[Any]
VisionFrame = NDArray[Any]
