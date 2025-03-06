from configparser import ConfigParser

import pytest
import torch

from face_swapper.src.networks.aad import AAD
from face_swapper.src.networks.unet import UNet


@pytest.mark.parametrize('output_size', [ 128, 256, 512 ])
def test_aad_with_unet(output_size : int) -> None:
	config_parser = ConfigParser()
	config_parser['training.model.generator'] =\
	{
		'identity_channels': '512',
		'output_channels': str(output_size * 16),
		'output_size': str(output_size),
		'num_blocks': '2'
	}

	generator = AAD(config_parser).eval()
	encoder = UNet(config_parser).eval()

	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, output_size, output_size)

	target_attributes = encoder(target_tensor)
	output_tensor = generator(source_tensor, target_attributes)

	assert output_tensor.shape == (1, 3, output_size, output_size)
