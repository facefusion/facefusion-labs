from configparser import ConfigParser

import pytest
import torch

from hyperswap.src.models.generator import Generator
from hyperswap.src.networks.aad import AAD
from hyperswap.src.networks.masknet import MaskNet
from hyperswap.src.networks.unet import UNet
from hyperswap.tests.helper import create_test_config_parser


@pytest.mark.parametrize('output_size', [ 256, 512, 1024 ])
def test_aad_with_unet(output_size : int) -> None:
	config_parser = ConfigParser()
	config_parser.read_dict(
	{
		'training.model.generator':
		{
			'source_channels': '512',
			'output_size': str(output_size),
			'num_blocks': '2'
		}
	})

	encoder = UNet(config_parser).eval()
	generator = AAD(config_parser).eval()

	source_tensor = torch.randn(1, 512)
	target_tensor = torch.randn(1, 3, output_size, output_size)

	target_features = encoder(target_tensor)
	output_tensor = generator(source_tensor, target_features)

	assert output_tensor.shape == (1, 3, output_size, output_size)


@pytest.mark.parametrize('output_size', [ 256, 512, 1024 ])
def test_mask_net(output_size : int) -> None:
	config_parser = ConfigParser()
	config_parser.read_dict(
	{
		'training.model.generator':
		{
			'output_size': str(output_size)
		},
		'training.model.masker':
		{
			'input_channels': '67',
			'output_channels': '1',
			'num_filters': '16'
		}
	})

	masker = MaskNet(config_parser).eval()

	target_tensor = torch.randn(1, 3, output_size, output_size)
	target_feature = torch.randn(1, 64, output_size, output_size)

	output_mask = masker(target_tensor, target_feature)

	assert output_mask.shape == (1, 1, output_size, output_size)


@pytest.mark.parametrize('output_size', [ 512, 1024 ])
def test_generator_with_initial_state(output_size : int) -> None:
	source_generator = Generator(create_test_config_parser(256))
	target_generator = Generator(create_test_config_parser(output_size))
	source_state = source_generator.state_dict()
	target_state = target_generator.state_dict()
	initial_keys = []

	for state_key, state_value in target_state.items():
		if state_key in source_state and state_value.shape == source_state.get(state_key).shape:
			initial_keys.append(state_key)

	assert 'generator.base_layers.0.primary_layers.0.conv1.weight' in initial_keys
	assert 'generator.output_layer.primary_layers.0.conv1.weight' in initial_keys
