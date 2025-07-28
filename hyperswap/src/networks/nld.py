import math
from configparser import ConfigParser

from torch import Tensor, nn


class NLD(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_input_channels = config_parser.getint('training.model.discriminator', 'input_channels')
		self.config_num_filters = config_parser.getint('training.model.discriminator', 'num_filters')
		self.config_kernel_size = config_parser.getint('training.model.discriminator', 'kernel_size')
		self.config_num_layers = config_parser.getint('training.model.discriminator', 'num_layers')
		self.layers = self.create_layers()
		self.sequences = nn.Sequential(*self.layers)

	def create_layers(self) -> nn.ModuleList:
		padding = math.ceil((self.config_kernel_size - 1) / 2)
		current_filters = self.config_num_filters
		layers = nn.ModuleList(
		[
			nn.Conv2d(self.config_input_channels, current_filters, kernel_size = self.config_kernel_size, stride = 2, padding = padding),
			nn.LeakyReLU(0.2)
		])

		for _ in range(1, self.config_num_layers):
			previous_filters = current_filters
			current_filters = min(current_filters * 2, 512)
			layers +=\
			[
				nn.Conv2d(previous_filters, current_filters, kernel_size = self.config_kernel_size, stride = 2, padding = padding),
				nn.InstanceNorm2d(current_filters),
				nn.LeakyReLU(0.2)
			]

		previous_filters = current_filters
		current_filters = min(current_filters * 2, 512)
		layers +=\
		[
			nn.Conv2d(previous_filters, current_filters, kernel_size = self.config_kernel_size, padding = padding),
			nn.InstanceNorm2d(current_filters),
			nn.LeakyReLU(0.2),
			nn.Conv2d(current_filters, 1, kernel_size = self.config_kernel_size, padding = padding)
		]
		return layers

	def forward(self, input_tensor : Tensor) -> Tensor:
		return self.sequences(input_tensor)
