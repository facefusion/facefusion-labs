import torch
from torch import Tensor, nn

from .types import WarpMatrixSet, EmbedderModule, Embedding, Padding

WARP_MATRIX_SET : WarpMatrixSet =\
{
	'vgg_face_hq_to_arcface_128_v2': torch.tensor(
	[
		[
			1.01305414,
			-0.00140513,
			-0.00585911
		],
		[
			0.00140513,
			1.01305414,
			0.11169602
		]
	], dtype = torch.float32),
	'arcface_128_v2_to_arcface_112_v2': torch.tensor(
	[
		[
			8.75000016e-01,
			-1.07193451e-08,
			3.80446920e-10
		],
		[
			1.07193451e-08,
			8.75000016e-01,
			-1.25000007e-01
		]
	], dtype = torch.float32)
}


def warp_tensor(input_tensor : Tensor, alignment_matrix : str) -> Tensor:
	matrix = WARP_MATRIX_SET.get(alignment_matrix).repeat(input_tensor.shape[0], 1, 1)
	grid = nn.functional.affine_grid(matrix.to(input_tensor.device), list(input_tensor.shape))
	output_tensor = nn.functional.grid_sample(input_tensor, grid, align_corners = False, padding_mode = 'reflection')
	return output_tensor


def calc_embedding(embedder : EmbedderModule, input_tensor : Tensor, padding : Padding) -> Embedding:
	crop_tensor = warp_tensor(input_tensor, 'arcface_128_v2_to_arcface_112_v2')
	crop_tensor = nn.functional.interpolate(crop_tensor, size = (112, 112), mode = 'area')
	crop_tensor[:, :, :padding[0], :] = 0
	crop_tensor[:, :, 112 - padding[1]:, :] = 0
	crop_tensor[:, :, :, :padding[2]] = 0
	crop_tensor[:, :, :, 112 - padding[3]:] = 0
	embedding = embedder(crop_tensor)
	embedding = nn.functional.normalize(embedding, p = 2)
	return embedding
