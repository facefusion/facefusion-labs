from typing import Any, Dict, List, Literal, Tuple, TypeAlias, TypedDict

from torch import Tensor
from torch.nn import Module

Batch : TypeAlias = Tuple[Tensor, Tensor]
BatchMode = Literal['equal', 'same', 'different']
UsageMode = Literal['source', 'target', 'both']

FileSet = TypedDict('FileSet',
{
	'dataset_name' : str,
	'usage_mode' : UsageMode,
	'file_paths' : List[str]
})

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

ConvertTemplate = Literal['arcface_128_to_arcface_112_v2', 'ffhq_512_to_arcface_128', 'vggfacehq_512_to_arcface_128']
ConvertTemplateSet : TypeAlias = Dict[ConvertTemplate, Tensor]
