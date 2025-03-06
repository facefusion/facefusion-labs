from typing import Any, TypeAlias, Dict

from torch import Tensor

Batch : TypeAlias = Tensor
Embedding : TypeAlias = Tensor

Config : TypeAlias = Dict[str, Any]
OptimizerConfig : TypeAlias = Any
