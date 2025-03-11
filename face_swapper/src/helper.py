import torch
from torch import Tensor, nn

from .types import EmbedderModule, Embedding, Padding, WarpTemplate, WarpTemplateSet

WARP_TEMPLATE_SET : WarpTemplateSet =\
{
	'vgg_face_hq_to_arcface_128_v2': torch.tensor(
	[
		[ 1.01305414, -0.00140513, -0.00585911 ],
		[ 0.00140513, 1.01305414, 0.11169602 ]
	]),
	'arcface_128_v2_to_arcface_112_v2': torch.tensor(
	[
		[ 8.75000016e-01, -1.07193451e-08, 3.80446920e-10 ],
		[ 1.07193451e-08, 8.75000016e-01, -1.25000007e-01 ]
	])
}


def warp_tensor(input_tensor : Tensor, warp_template : WarpTemplate) -> Tensor:
	normed_warp_template = WARP_TEMPLATE_SET.get(warp_template).repeat(input_tensor.shape[0], 1, 1)
	affine_grid = nn.functional.affine_grid(normed_warp_template.to(input_tensor.device), list(input_tensor.shape))
	output_tensor = nn.functional.grid_sample(input_tensor, affine_grid, align_corners = False, padding_mode = 'reflection')
	return output_tensor


def calc_embedding(embedder : EmbedderModule, input_tensor : Tensor, padding : Padding) -> Embedding:
	crop_tensor = warp_tensor(input_tensor, 'arcface_128_v2_to_arcface_112_v2')
	crop_tensor = nn.functional.interpolate(crop_tensor, size = 112, mode = 'area')
	crop_tensor[:, :, :padding[0], :] = 0
	crop_tensor[:, :, 112 - padding[1]:, :] = 0
	crop_tensor[:, :, :, :padding[2]] = 0
	crop_tensor[:, :, :, 112 - padding[3]:] = 0

	embedding = embedder(crop_tensor)
	embedding = nn.functional.normalize(embedding, p = 2)
	return embedding


def overlay_mask(target_tensor : Tensor, mask_tensor : Tensor) -> Tensor:
	color_tensor = torch.zeros(*target_tensor.shape, dtype = target_tensor.dtype, device = target_tensor.device)
	color_tensor[:, 2, :, :] = 1
	mask_tensor = mask_tensor.repeat(1, 3, 1, 1).clamp(0, 0.8)
	output_tensor = target_tensor * (1 - mask_tensor) + color_tensor * mask_tensor
	return output_tensor
