from typing import Any, Dict, TypeAlias

from torch import Tensor

Batch : TypeAlias = Tensor
Embedding : TypeAlias = Tensor

Config : TypeAlias = Dict[str, Any]
ConfigSet : TypeAlias = Dict[str, Config]

OptimizerSet : TypeAlias = Any
