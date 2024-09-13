from typing import Any, Tuple

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

Batch = Tuple[Tensor, Tensor]
Loader = DataLoader[Tuple[Tensor, ...]]

Embedding = NDArray[Any]
FaceLandmark5 = NDArray[Any]
VisionFrame = NDArray[Any]
