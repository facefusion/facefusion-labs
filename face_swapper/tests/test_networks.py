import sys

import pytest
import torch

sys.path.append('..')

from face_swapper.src.networks.aad import AAD
from face_swapper.src.networks.unet import UNet


@pytest.mark.parametrize('output_size', [ 256 ])
def test_aad_with_unet(output_size : int) -> None:
	identity_channels = 512
	if output_size == 256:
		output_channels = 4096
	if output_size == 512:
		output_channels = 8192
	num_blocks = 2

	generator = AAD(identity_channels, output_channels, output_size, num_blocks).eval()
	encoder = UNet(output_size).eval()

	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, output_size, output_size)

	target_attributes = encoder(target_tensor)
	output_tensor = generator(source_tensor, target_attributes)

	assert output_tensor.shape == (1, 3, output_size, output_size)
