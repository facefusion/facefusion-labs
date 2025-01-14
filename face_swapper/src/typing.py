from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

Batch = Tuple[Any, Any, Any]
Loader = DataLoader[Tuple[Tensor, ...]]
TargetAttributes = Tuple[Tensor, ...]
DiscriminatorOutputs = List[List[Tensor]]
LossDict = Dict[str, Tensor]
IDEmbedding = Tensor
StateDict = OrderedDict[str, Any]
Embedding = NDArray[Any]
VisionFrame = NDArray[Any]
