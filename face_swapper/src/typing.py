from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader


Batch = Tuple[Any, Any, Any]
Loader = DataLoader[Tuple[Tensor, ...]]
TargetAttributes = Tuple[Tensor, ...]
DiscriminatorOutputs = List[List[Tensor]]
IdEmbedding = Tensor
SourceEmbedding = IdEmbedding
StateDict = OrderedDict[str, Any]
Padding = Tuple[int, int, int, int]
FaceLandmark203 = Tensor
VisionTensor = Tensor
LossTensor = Tensor
GeneratorLossSet = Dict[str, Tensor]
DiscriminatorLossSet = Dict[str, Tensor]
VisionFrame = NDArray[Any]
