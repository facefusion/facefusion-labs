from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch.nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

Batch = Tuple[Any, Any, Any]
Loader = DataLoader[Tuple[Tensor, ...]]
ImagePathList = List[str]
ImagePathSet = Dict[str, ImagePathList]

SwapAttributes = Tuple[Tensor, ...]
TargetAttributes = Tuple[Tensor, ...]
DiscriminatorOutputs = List[List[Tensor]]

IdEmbedding = Tensor
SourceEmbedding = IdEmbedding
FaceLandmark203 = Tensor

StateSet = OrderedDict[str, Any]
Padding = Tuple[int, int, int, int]

LossTensor = Tensor
VisionTensor = Tensor
VisionFrame = NDArray[Any]

GeneratorLossSet = Dict[str, Tensor]
DiscriminatorLossSet = Dict[str, Tensor]

Generator = torch.nn.Module
IdEmbedder = torch.nn.Module
