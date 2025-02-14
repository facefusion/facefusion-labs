from typing import Any, Tuple, TypeAlias

from numpy.typing import NDArray
from torch import Tensor

Batch : TypeAlias = Tuple[Tensor, Tensor]

Embedding : TypeAlias = NDArray[Any]
EmbeddingDataset : TypeAlias = NDArray[Embedding]
FaceLandmark5 : TypeAlias = NDArray[Any]

VisionFrame : TypeAlias = NDArray[Any]
