import configparser
from itertools import chain
from typing import List

import numpy
import torch.nn
import torch.nn as nn
from torch import Tensor

from face_swapper.src.types import DiscriminatorOutputs

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class MultiscaleDiscriminator(nn.Module):
	def __init__(self) -> None:
		super(MultiscaleDiscriminator, self).__init__()
		self.input_channels = CONFIG.getint('training.model.discriminator', 'input_channels')
		self.num_filters = CONFIG.getint('training.model.discriminator', 'num_filters')
		self.kernel_size = CONFIG.getint('training.model.discriminator', 'kernel_size')
		self.num_layers = CONFIG.getint('training.model.discriminator', 'num_layers')
		self.num_discriminators = CONFIG.getint('training.model.discriminator', 'num_discriminators')

		self.downsample = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = [ 1, 1 ], count_include_pad = False) # type:ignore[arg-type]
		self.prepare_discriminators()

	def prepare_discriminators(self) -> None:
		for discriminator_index in range(self.num_discriminators):
			single_discriminator = NLayerDiscriminator(self.input_channels, self.num_filters, self.num_layers, self.kernel_size)
			setattr(self, 'discriminator_layer_{}'.format(discriminator_index), single_discriminator.model)

	def forward(self, input_tensor : Tensor) -> DiscriminatorOutputs:
		discriminator_outputs = []
		temp_tensor = input_tensor

		for discriminator_index in range(self.num_discriminators):
			model_layers = getattr(self, 'discriminator_layer_{}'.format(self.num_discriminators - 1 - discriminator_index))
			discriminator_outputs.append([ model_layers(temp_tensor) ])

			if discriminator_index < (self.num_discriminators - 1):
				temp_tensor = self.downsample(temp_tensor)

		return discriminator_outputs


class NLayerDiscriminator(nn.Module):
	def __init__(self, input_channels : int, num_filters : int, num_layers : int, kernel_size : int) -> None:
		super(NLayerDiscriminator, self).__init__()
		self.num_layers = num_layers
		model_layers = self.prepare_model_layers(input_channels, num_filters, num_layers, kernel_size)
		self.model = nn.Sequential(*list(chain(*model_layers)))

	def prepare_model_layers(self, input_channels : int, num_filters : int, num_layers : int, kernel_size : int) -> List[List[torch.nn.Module]]:
		padding_size = int(numpy.ceil((kernel_size - 1.0) / 2))

		model_layers =\
		[
			[
				nn.Conv2d(input_channels, num_filters, kernel_size = kernel_size, stride = 2, padding = padding_size),
			 	nn.LeakyReLU(0.2, True)
			]
		]
		current_filters = num_filters

		for layer_index in range(1, num_layers):
			previous_filters = current_filters
			current_filters = min(current_filters * 2, 512)
			model_layers +=\
			[
				[
					nn.Conv2d(previous_filters, current_filters, kernel_size = kernel_size, stride = 2, padding = padding_size),
					nn.InstanceNorm2d(current_filters), nn.LeakyReLU(0.2, True)
				]
			]
		previous_filters = current_filters
		current_filters = min(current_filters * 2, 512)
		model_layers +=\
		[
			[
				nn.Conv2d(previous_filters, current_filters, kernel_size = kernel_size, padding = padding_size),
				nn.InstanceNorm2d(current_filters),
				nn.LeakyReLU(0.2, True)
			],
			[
				nn.Conv2d(current_filters, 1, kernel_size = kernel_size, padding = padding_size)
			]
		]
		return model_layers

	def forward(self, input_tensor : Tensor) -> Tensor:
		return self.model(input_tensor)
