from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader

DataLoaderSet = DataLoader[Tuple[Tensor, ...]], DataLoader[Tuple[Tensor, ...]]

