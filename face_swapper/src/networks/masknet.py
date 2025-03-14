from configparser import ConfigParser

import torch
from torch import Tensor, nn

from ..types import Attribute, Mask


class MaskNet(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_input_channels = config_parser.getint('training.model.masker', 'input_channels')
		self.config_output_channels = config_parser.getint('training.model.masker', 'output_channels')
		self.sequences = self.create_sequences()

	def create_sequences(self) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(self.config_input_channels, self.config_output_channels, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(self.config_output_channels),
			nn.ReLU(inplace = True),
			nn.Conv2d(self.config_output_channels, self.config_output_channels, kernel_size = 1),
			nn.Sigmoid()
		)

	def forward(self, input_tensor : Tensor, input_attribute : Attribute) -> Mask:
		output_mask = torch.cat([ input_tensor, input_attribute ], dim = 1)
		output_mask = self.sequences(output_mask)
		return output_mask
