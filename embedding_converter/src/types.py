from typing import Any, Tuple

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

Embedding = NDArray[Any]
EmbeddingDataset = NDArray[Embedding]
FaceLandmark5 = NDArray[Any]

VisionFrame = NDArray[Any]
VisionTensor = Tensor

Batch = Tuple[VisionTensor, VisionTensor]
