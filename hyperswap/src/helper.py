import torch
from torch import Tensor, nn

from .types import ConvertTemplate, ConvertTemplateSet, EmbedderModule, Embedding, Mask, Padding

CONVERT_TEMPLATE_SET : ConvertTemplateSet =\
{
	'arcface_128_to_arcface_112_v2': torch.tensor(
	[
		[ 8.75000016e-01, -1.07193451e-08, 3.80446920e-10 ],
		[ 1.07193451e-08, 8.75000016e-01, -1.25000007e-01 ]
	]),
	'ffhq_512_to_arcface_128': torch.tensor(
	[
		[ 8.50048894e-01, -1.29486822e-04, 1.90956388e-03 ],
		[ 1.29486822e-04, 8.50048894e-01, 9.56254653e-02 ]
	]),
	'vggfacehq_512_to_arcface_128': torch.tensor(
	[
		[ 1.01305414, -0.00140513, -0.00585911 ],
		[ 0.00140513, 1.01305414, 0.11169602 ]
	])
}


def convert_transform(input_tensor : Tensor, convert_template : ConvertTemplate) -> Tensor:
	convert_matrix = CONVERT_TEMPLATE_SET.get(convert_template).repeat(input_tensor.shape[0], 1, 1)
	affine_grid = nn.functional.affine_grid(convert_matrix.to(input_tensor.device), list(input_tensor.shape))
	output_tensor = nn.functional.grid_sample(input_tensor, affine_grid, padding_mode = 'reflection')
	return output_tensor


def calc_embedding(embedder : EmbedderModule, input_tensor : Tensor, padding : Padding) -> Embedding:
	crop_tensor = convert_transform(input_tensor, 'arcface_128_to_arcface_112_v2')
	crop_tensor = nn.functional.interpolate(crop_tensor, size = 112, mode = 'area')
	crop_tensor[:, :, :padding[0], :] = 0
	crop_tensor[:, :, 112 - padding[1]:, :] = 0
	crop_tensor[:, :, :, :padding[2]] = 0
	crop_tensor[:, :, :, 112 - padding[3]:] = 0

	embedding = embedder(crop_tensor)
	embedding = nn.functional.normalize(embedding, p = 2)
	return embedding


def overlay_mask(input_tensor : Tensor, input_mask : Mask) -> Tensor:
	overlay_tensor = torch.zeros(*input_tensor.shape, dtype = input_tensor.dtype, device = input_tensor.device)
	overlay_tensor[:, 2, :, :] = 1
	input_mask = input_mask.repeat(1, 3, 1, 1).clamp(0, 0.8)
	output_tensor = input_tensor * (1 - input_mask) + overlay_tensor * input_mask
	return output_tensor


def inject_noise(input_tensor : Tensor, factor : float) -> Tensor:
	noise = torch.randn_like(input_tensor) * factor
	output_tensor = input_tensor + noise
	return output_tensor
