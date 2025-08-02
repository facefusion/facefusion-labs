from typing import Any, Dict, Literal, Tuple, TypeAlias

from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]
BatchMode = Literal['equal', 'same', 'different']
UsageMode = Literal['source', 'target', 'both']

ConvertTemplate = Literal['arcface_128_to_arcface_112_v2', 'ffhq_512_to_arcface_128', 'vggfacehq_512_to_arcface_128']
ConvertTemplateSet : TypeAlias = Dict[ConvertTemplate, Tensor]

Feature : TypeAlias = Tensor
Embedding : TypeAlias = Tensor
Mask : TypeAlias = Tensor
Loss : TypeAlias = Tensor

Padding : TypeAlias = Tuple[int, int, int, int]

GeneratorModule : TypeAlias = Module
EmbedderModule : TypeAlias = Module
GazerModule : TypeAlias = Module
FaceMaskerModule : TypeAlias = Module

OptimizerSet : TypeAlias = Any

TrainerCompileMode = Literal['default', 'reduce-overhead', 'max-autotune']
TrainerStrategy = Literal['auto', 'ddp', 'ddp_spawn', 'ddp_find_unused_parameters_true']
TrainerPrecision = Literal['64-true', '32-true', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', 'transformer-engine', 'transformer-engine-float16']
