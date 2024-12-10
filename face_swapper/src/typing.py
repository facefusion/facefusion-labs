from typing import Any, Tuple, List, Dict, Optional

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

Batch = Tuple[Any, Any, Any]
Loader = DataLoader[Tuple[Tensor, ...]]
TargetAttributes = Tuple[Tensor, ...]
DiscriminatorOutputs = List[List[Tensor]]
LossDict = Dict[str, Tensor]

Embedding = NDArray[Any]
VisionFrame = NDArray[Any]
