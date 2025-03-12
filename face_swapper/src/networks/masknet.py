from configparser import ConfigParser

import torch
from torch import Tensor, nn

from ..types import Attribute


class MaskNet(nn.Module):
	def __init__(self, config_parser : ConfigParser) -> None:
		super().__init__()
		self.config_input_channels = config_parser.getint('training.model.masker', 'input_channels')
		self.config_output_channels = config_parser.getint('training.model.masker', 'output_channels')
		self.config_num_filters = config_parser.getint('training.model.masker', 'num_filters')
		self.down_samples = self.create_down_samples(self.config_input_channels, self.config_num_filters)
		self.up_samples = self.create_up_samples(self.config_num_filters)
		self.bottleneck = BottleNeck(self.config_num_filters * 2)
		self.conv = nn.Conv2d(self.config_num_filters, self.config_output_channels, kernel_size = 1)
		self.sigmoid = nn.Sigmoid()

	@staticmethod
	def create_down_samples(input_channels : int, num_filters : int) -> nn.ModuleList:
		return nn.ModuleList(
		[
			DownSample(input_channels, num_filters),
			DownSample(num_filters, num_filters * 2)
		])

	@staticmethod
	def create_up_samples(num_filters : int) -> nn.ModuleList:
		return nn.ModuleList(
		[
			UpSample(num_filters * 2, num_filters),
			UpSample(num_filters, num_filters)
		])

	def forward(self, target_tensor : Tensor, target_attribute : Attribute) -> Tensor:
		output_tensor = torch.cat([ target_tensor, target_attribute ], dim = 1)

		for down_sample in self.down_samples:
			output_tensor = down_sample(output_tensor)

		output_tensor = self.bottleneck(output_tensor)

		for up_sample in self.up_samples:
			output_tensor = up_sample(output_tensor)

		output_tensor = self.conv(output_tensor)
		output_tensor = self.sigmoid(output_tensor)
		return output_tensor


class BottleNeck(nn.Module):
	def __init__(self, num_filters : int):
		super().__init__()
		self.sequences = self.create_sequences(num_filters)
		self.relu = nn.ReLU(inplace = True)

	@staticmethod
	def create_sequences(num_filters : int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False),
			nn.BatchNorm2d(num_filters),
			nn.ReLU(inplace = True),
			nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False),
			nn.BatchNorm2d(num_filters),
			nn.ReLU(inplace = True)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor) + input_tensor
		output_tensor = self.relu(output_tensor)
		return output_tensor


class UpSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 2, stride = 2),
			nn.ReLU(inplace = True)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor)
		return output_tensor


class DownSample(nn.Module):
	def __init__(self, input_channels : int, output_channels : int) -> None:
		super().__init__()
		self.sequences = self.create_sequences(input_channels, output_channels)

	@staticmethod
	def create_sequences(input_channels : int, output_channels : int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1, bias = False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(2)
		)

	def forward(self, input_tensor : Tensor) -> Tensor:
		output_tensor = self.sequences(input_tensor)
		return output_tensor
