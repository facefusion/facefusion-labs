import configparser
from typing import List

import numpy
import torch.nn as nn

from face_swapper.src.types import VisionTensor

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class MultiscaleDiscriminator(nn.Module):
	def __init__(self) -> None:
		super(MultiscaleDiscriminator, self).__init__()
		self.downsample = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = (1, 1), count_include_pad = False)
		self.discriminators = self.create_discriminators()

	def create_discriminators(self) -> nn.ModuleList:
		num_discriminators = CONFIG.getint('training.model.discriminator', 'num_discriminators')
		input_channels = CONFIG.getint('training.model.discriminator', 'input_channels')
		num_filters = CONFIG.getint('training.model.discriminator', 'num_filters')
		kernel_size = CONFIG.getint('training.model.discriminator', 'kernel_size')
		num_layers = CONFIG.getint('training.model.discriminator', 'num_layers')
		discriminators = nn.ModuleList()

		for _ in range(num_discriminators):
			discriminator = NLayerDiscriminator(input_channels, num_filters, num_layers, kernel_size).discriminator
			discriminators.append(discriminator)

		return discriminators

	def forward(self, input_tensor : VisionTensor) -> List[List[VisionTensor]]:
		temp_tensor = input_tensor
		output_tensors = []

		for discriminator in self.discriminators:
			output_tensors.append([ discriminator(temp_tensor) ])
			temp_tensor = self.downsample(temp_tensor)

		return output_tensors


class NLayerDiscriminator(nn.Module):
	def __init__(self, input_channels : int, num_filters : int, num_layers : int, kernel_size : int) -> None:
		super(NLayerDiscriminator, self).__init__()
		layers = self.create_layers(input_channels, num_filters, num_layers, kernel_size)
		self.discriminator = nn.Sequential(*layers)

	@staticmethod
	def create_layers(input_channels : int, num_filters : int, num_layers: int, kernel_size : int) -> List[nn.Module]:
		padding = int(numpy.ceil((kernel_size - 1) / 2))
		current_filters = num_filters
		layers =\
		[
			nn.Conv2d(input_channels, current_filters, kernel_size = kernel_size, stride = 2, padding = padding),
			nn.LeakyReLU(0.2, True)
		]

		for _ in range(1, num_layers):
			previous_filters = current_filters
			current_filters = min(current_filters * 2, 512)
			layers +=\
			[
				nn.Conv2d(previous_filters, current_filters, kernel_size = kernel_size, stride = 2, padding = padding),
				nn.InstanceNorm2d(current_filters),
				nn.LeakyReLU(0.2, True)
			]

		previous_filters = current_filters
		current_filters = min(current_filters * 2, 512)
		layers +=\
		[
			nn.Conv2d(previous_filters, current_filters, kernel_size = kernel_size, padding = padding),
			nn.InstanceNorm2d(current_filters),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(current_filters, 1, kernel_size = kernel_size, padding = padding)
		]
		return layers

	def forward(self, input_tensor : VisionTensor) -> VisionTensor:
		return self.discriminator(input_tensor)
