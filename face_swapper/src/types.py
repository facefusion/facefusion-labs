from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeAlias

from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

ImagePathList : TypeAlias = List[str]
ImagePathSet : TypeAlias = Dict[str, ImagePathList]

SwapAttributes : TypeAlias = Tuple[Tensor, ...]
TargetAttributes : TypeAlias = Tuple[Tensor, ...]
DiscriminatorOutputs : TypeAlias = List[List[Tensor]]

Embedding : TypeAlias = Tensor
FaceLandmark203 : TypeAlias = Tensor

StateSet : TypeAlias = OrderedDict[str, Any]
Padding : TypeAlias = Tuple[int, int, int, int]

VisionFrame : TypeAlias = NDArray[Any]
LossTensor : TypeAlias = Tensor
VisionTensor : TypeAlias = Tensor

Batch : TypeAlias = Tuple[VisionTensor, VisionTensor, Tensor]

GeneratorLossSet : TypeAlias = Dict[str, Tensor]
DiscriminatorLossSet : TypeAlias = Dict[str, Tensor]

Generator : TypeAlias = Module
Embedder : TypeAlias = Module
