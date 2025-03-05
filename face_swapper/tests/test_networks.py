import pytest
import torch

from face_swapper.src.networks.aad import AAD
from face_swapper.src.networks.unet import UNet


@pytest.mark.parametrize('output_size', [ 256 ])
def test_aad_with_unet(output_size : int) -> None:
	generator = AAD(512, 4096, output_size, 2).eval()
	encoder = UNet(output_size).eval()

	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, output_size, output_size)

	target_attributes = encoder(target_tensor)
	output_tensor = generator(source_tensor, target_attributes)

	assert output_tensor.shape == (1, 3, output_size, output_size)
