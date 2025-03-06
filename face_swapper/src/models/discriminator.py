import configparser
from typing import List

from torch import Tensor, nn

from ..networks.nld import NLD

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')


class Discriminator(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.avg_pool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = (1, 1), count_include_pad = False)
		self.discriminators = self.create_discriminators()

	@staticmethod
	def create_discriminators() -> nn.ModuleList:
		num_discriminators = CONFIG.getint('training.model.discriminator', 'num_discriminators')
		input_channels = CONFIG.getint('training.model.discriminator', 'input_channels')
		num_filters = CONFIG.getint('training.model.discriminator', 'num_filters')
		kernel_size = CONFIG.getint('training.model.discriminator', 'kernel_size')
		num_layers = CONFIG.getint('training.model.discriminator', 'num_layers')
		discriminators = nn.ModuleList()

		for _ in range(num_discriminators):
			discriminator = NLD(input_channels, num_filters, num_layers, kernel_size).sequences
			discriminators.append(discriminator)

		return discriminators

	def forward(self, input_tensor : Tensor) -> List[Tensor]:
		temp_tensor = input_tensor
		output_tensors = []

		for discriminator in self.discriminators:
			output_tensors.append(discriminator(temp_tensor))
			temp_tensor = self.avg_pool(temp_tensor)

		return output_tensors
