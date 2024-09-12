from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader

Batch = Tuple[Tensor, Tensor]
Loader = DataLoader[Tuple[Tensor, ...]]
