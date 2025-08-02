from typing import Any, Literal, TypeAlias

from torch import Tensor

Batch : TypeAlias = Tensor
Embedding : TypeAlias = Tensor

OptimizerSet : TypeAlias = Any

TrainerCompileMode = Literal['default', 'reduce-overhead', 'max-autotune']
TrainerStrategy = Literal['auto', 'ddp', 'ddp_spawn', 'ddp_find_unused_parameters_true']
TrainerPrecision = Literal['64-true', '32-true', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', 'transformer-engine', 'transformer-engine-float16']
