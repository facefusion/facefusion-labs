from configparser import ConfigParser
from typing import List

from torch import Tensor, nn

from ..networks.nld import NLD


class Discriminator(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_num_discriminators = config_parser.getint('training.model.discriminator', 'num_discriminators')
		self.config_parser = config_parser
		self.avg_pool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = (1, 1), count_include_pad = False)
		self.discriminators = self.create_discriminators()

	def create_discriminators(self) -> nn.ModuleList:
		discriminators = nn.ModuleList()

		for _ in range(self.config_num_discriminators):
			discriminator = NLD(self.config_parser).sequences
			discriminators.append(discriminator)

		return discriminators

	def forward(self, input_tensor : Tensor) -> List[Tensor]:
		temp_tensor = input_tensor
		output_tensors = []

		for discriminator in self.discriminators:
			output_tensor = discriminator(temp_tensor)
			output_tensors.append(output_tensor)
			temp_tensor = self.avg_pool(temp_tensor)

		return output_tensors
